# 提案：通用 Workload 资源

状态：草案，阶段 0 POC 已完成
目标：模型实例、基准测试、缓存服务实例三类负载统一编译到一个 Workload 资源
影响范围：新增 `workloads` 表；三类负载的用户面 API **保持不变**

## 1. 目标形态

引入通用 `Workload` 资源，只承载执行语义——规格（镜像、命令、端口、资源、重启策略）、绑定结果（worker、加速器）、执行状态。worker 只 watch 这一种资源，维护一套状态机。

领域语义留在各自的 server 侧 controller：模型部署、基准测试、缓存服务仍是用户面 API，由 controller 把副本数、逐节点扇出、run-to-completion 等语义「编译」为 Workload，并把执行状态聚合回领域状态。**用户不直接创建 Workload。**

```
Model ──────────┐
CacheService ───┼── controller 编译 ──> Workload ──> worker 控制回路 ──> 容器
Benchmark ──────┘         ▲                  │
                          └── 状态聚合 ───────┘
```

## 2. 三类负载的字段映射

这一节是提案的核心：资源模型必须同时容纳三类，只按其中一类设计必然要返工。

### 2.1 已经是 Workload 的部分

**`ModelInstanceSubordinateWorker`（`schemas/models.py:678`）字面上就是一个内嵌进 JSON 列的 Workload**：

```python
worker_id, worker_name, worker_ip, worker_ifname
gpu_type, gpu_indexes, gpu_addresses, total_gpus
computed_resource_claim
pid, ports, arguments
state, state_message
```

分布式实例的「一实例多 Workload + 分组标签」不是新发明，而是把这个内嵌列表提升成行。收益是直接的：`serve_manager.py` 里散布的 `sw_pos` 查找和 `distributed_servers.subordinate_workers.{i}` 路径 patch 全部消失。

`CacheServiceInstanceBase`（`schemas/cache_services.py:222`）13 个字段里，只有 `cache_service_id` 是领域字段，其余全是通用执行状态。

### 2.2 字段对照

| Workload 字段 | 模型实例 | 基准测试 | 缓存服务实例 |
|---|---|---|---|
| `owner_kind` / `owner_id` | `model_instance_id` | `benchmark_id` | `cache_service_id` |
| `owner_principal_id` | ✓ 有 | ✓ 有 | **无**（靠父服务间接过滤） |
| `cluster_id` | ✓ | ✓ | ✓ |
| `worker_id` | ✓（主/从各一） | ✓ | ✓ |
| `role` / `group_key` / `group_index` | **需要**（leader / worker） | — | — |
| `start_after` | **需要**，两种门（见下） | — | — |
| `managed` | **需要**（DELEGATED 只预留资源） | — | — |
| `gpu_indexes` / `gpu_addresses` / `gpu_type` | **需要** | — | —（吃全部 GPU） |
| `computed_resource_claim` | **需要**（调度器的资源决策） | — | — |
| `ports`（命名） | `port` + `ports: List[int]`，二者是同一个端口 | — | `port` + `metrics_port` |
| `progress` | 模型文件下载进度 | — | — |
| `pid` | ✓ | ✓ | **缺**（当前只在内存里） |
| `arguments` | 从属侧有 | — | — |
| `state` / `state_message` | ✓ | ✓ | ✓ |
| `restart_count` / `last_restart_time` | ✓ | — | ✓ |
| `restart_policy` | `always` | `never` | `always` |
| `active_deadline_seconds` | — | **需要**（`_handle_benchmark_timeout`） | — |
| `spec_digest` | — | — | ✓ |
| `healthy` / `last_check_at` | — | — | ✓ |

**只按缓存服务设计会漏掉的三样**：加速器绑定（`gpu_indexes` / `computed_resource_claim`）、分组（`group_key` / `role`）、`arguments`。这三样恰好是最难事后补的——它们决定了行的粒度。

**只按缓存服务设计会误加的一样**：`healthy` / `last_check_at` 是缓存服务的探针语义；模型实例的健康检查是业务层的（`sync_model_instances_inference_health` 发真实推理请求），不该由 Workload 承载。建议保留在 Workload 上但标注为可选，由 restart_policy 之外的探针配置驱动。

### 2.3 状态机是两套，不是一套

**这一条是只看缓存服务时最容易搞错的地方。**

`CacheServiceStateEnum` 和它的实例状态几乎一一对应，容易误以为领域状态 = Workload 状态。但 `ModelInstanceStateEnum`（`schemas/models.py:636`）不是：

```
PENDING → ANALYZING → SCHEDULED → INITIALIZING → DOWNLOADING → STARTING → RUNNING
          └ 调度器 ┘              └────── 模型文件准备 ──────┘
```

`ANALYZING` / `SCHEDULED` 发生在容器存在之前，`DOWNLOADING` 是模型文件准备。这些都**不是 Workload 的状态**。

所以：

- `WorkloadStateEnum` 只覆盖执行子集：`pending` / `starting` / `running` / `unreachable` / `succeeded` / `error`
- 领域资源保留自己更丰富的生命周期，controller 负责映射
- worker 侧已有的 `WorkloadPhase`（`worker/controlloop/workload_state.py`）是**容器运行时状态**的中性分类，与上面两者都不同——它读 `WorkloadStatus`，不落库。三者不要合并

### 2.4 `download_progress` 的归属（POC 已结论）

放在 Workload 上（`progress`）。从属节点的下载进度天然是**逐 workload** 的状态，留在领域侧就得重新引入一个按 worker 索引的列表，正是要消除的那种结构。

代价是一个需要接受的推论：**Workload 行在绑定时创建，不是在容器启动时创建**——下载发生在容器之前，行必须先于容器存在。这与 K8s 一致（Pod 先于其容器存在），也与 §2.3 自洽：实例处于 `DOWNLOADING` 时，它的 workload 处于 `pending`。

## 3. 表定义

```
workloads
  id, name, created_at, updated_at

  -- 归属
  owner_kind, owner_id, owner_principal_id, cluster_id

  -- 分组（分布式实例）
  group_key, group_index, role

  -- 绑定结果
  worker_id
  gpu_type, gpu_indexes, gpu_addresses          -- JSON
  computed_resource_claim                        -- JSON

  -- 规格
  restart_policy                                 -- always | on_failure | never
  active_deadline_seconds
  spec_digest
  labels                                         -- JSON，展示性标签

  -- 执行状态
  state, state_message
  ports                                          -- JSON: {"service": 40001, "metrics": 40002}
  pid, arguments                                 -- JSON
  restart_count, last_restart_time
  healthy, last_check_at
  progress                                       -- 供给进度，见 §2.4
```

`ports` 是唯一的端口载体，**不要另设 `port` 列**：`_assign_ports` 里 `mi.ports = [mi.port]` 然后 extend，两者恒为同一个端口，拆成两列只会让它们漂移。第一个端口通用命名为 `service`，其余由各 backend 自己的 compiler 命名——`ports[1:]` 的布局取决于 backend 与 executor（vLLM/mp 是 DP-RPC + master-port + VLLM_PORT，vLLM/ray 仅 dp>1 时有 DP-RPC，其他 backend 没有），且 connecting port 恒在末位，通用层猜不了。

`managed: bool` 区分「gpustack 要跑的容器」和「只做了资源预留」。见 §4 阶段 0 的结论 5。

### 租户作用域

缓存服务实例没有 `owner_principal_id`，列表接口靠父服务间接过滤（`routes/cache_service_instances.py:96` 的 `cache_service_id.in_(visible_service_ids)`）。通用表对每种 owner 做不了这种间接，所以 `owner_principal_id` 必须是列，由 controller 创建时从 owner 抄下来。加了之后 `tenant_list_conditions`（`api/tenant.py:496`）的默认分支直接生效。

### 索引与唯一约束

数据库支持面为 PostgreSQL 13+ / MySQL 8.0.36+（含 openGauss、OceanBase），以下在两者上行为一致。

```
INDEX  (worker_id)                            -- worker 对账 + orphan 清理，最热
INDEX  (owner_kind, owner_id)                 -- controller 扇出与状态聚合
INDEX  (owner_kind, owner_id, state)          -- 端点解析，见下
INDEX  (cluster_id)
INDEX  (group_key)                            -- 分布式实例的组内查询
UNIQUE (owner_kind, owner_id, worker_id, group_index)
```

唯一约束取代 `uix_cache_service_instances_service_worker`（「一个缓存服务在一个 worker 上只有一个实例」）。没有它，controller 扇出并发跑两遍就会建出重复行，这类问题只在高负载时出现。加 `group_index` 是因为分布式模型实例可能在同一 worker 上放多个从属 workload。

`(owner_kind, owner_id, state)` 服务于 `_resolve_managed_endpoint`（`server/cache_services.py:148`）——**每次模型实例调度都要跑**，是频率最高的查询。现在是全取回来在 Python 里筛 RUNNING，迁移时一并下推到 SQL。

## 4. 分阶段路径

三类都切，但分步。每一步都可独立发布、独立回滚。

### 阶段 0：POC —— 已完成 ✅

代码在 `hack/poc-workload/`（31 个用例，全部对着生产代码断言，不进 CI），完整结论见 `hack/poc-workload/FINDINGS.md`。

**模型立得住，六项修改全是加减字段，不动结构**，已回写进 §2、§3：

1. 字段覆盖完整——`ModelInstanceSubordinateWorker` 的 15 个字段全部有去处，只有 `total_gpus` 不带（可推出）
2. 容器名与 `get_deployment_metadata` 逐个相等，迁移不会孤立在跑的容器
3. `port` 必须与 `ports[0]` 合并；`ports[1:]` 无法通用命名（§3）
4. 启动次序可干净上移到 controller，但需要**两种门**：`STARTED`（follower 等 leader）与 `READY`（follower 等前一个 follower）
5. `DELEGATED` 把「要跑的容器」和「只是资源预留」混在了一起，需要 `managed` 字段
6. leader 与 follower 可能同机，唯一约束必须含 `group_index`
7. `should_update` 是调用方策略，不属于聚合的返回值
8. `download_progress` 归 Workload（§2.4）

**顺带证实两个与本提案无关的既存问题**（无论提案做不做都成立）：

- DELEGATED 的从属节点会被对账循环写成 ERROR——事件路径不给它建容器，对账循环却查不到 workload 就判失败。链路已用探针打通到生产代码；可达性（v2 的 GGUF 分布式能否走到多 worker 调度）待产品确认。
- 与 leader 同机的 follower 永远不被管理——`get_deployment_metadata` 按 worker 查、只返回一个结果。

**顺带发现**：`RUN_FIRST` 在生产路径上是死的。调度器只对 vLLM / SGLang / MindIE 设 `INITIALIZE_LATER`，其余留默认 `DELEGATED`，无处设 `RUN_FIRST`。迁移时可确认是否保留。

### 阶段 1：缓存服务实例（免迁移窗口）

缓存服务尚未发布，**这是唯一不需要数据迁移的窗口**——建表 + 删表，无回填、无双写期。

- 建 `workloads` 表，删 `cache_service_instances`
- `CacheServiceController` 改为编译 Workload + 聚合状态
- `cache_service_manager` 改为 watch `workloads`（filter `owner_kind='cache_service'`），**这一步之后它就是 Workload 控制回路的雏形**
- 前端零改动（见 §6）

迁移链是单线的（36 个 revision，单 head `b7e2c4d15a80`，无分支），`cache_service_instances` 由 `d5e8f0a1b2c3` 创建，必然在新 revision 之前执行，所以 `drop_table` 无需条件判断。

### 阶段 2：基准测试

天然的 Job 形态，验证 `restart_policy=never` + `active_deadline_seconds`。

`Benchmark` 表**不减字段**，只是执行部分（`state` / `pid` / `worker_id`）改为从子 Workload 聚合。API 不变。

明确**不下沉**的两样：

- **结果收集** —— `_collect_results`（`benchmark_manager.py:767` 起约 400 行）从容器写出的 point 文件读、聚合、回写 metrics。领域逻辑。
- **串行队列** —— `_active_benchmark_id` 保证同 worker 同时只跑一个压测（独占 GPU）。这是**准入/调度语义**，按「调度是上层职责」应留在 server 侧：同一 worker 上只把一个基准测试置为 assigned。Workload 层不引入队列或并发上限概念。

### 阶段 3：模型实例

爆炸半径最大，涉及调度器与分布式拓扑。前置条件是阶段 0 的结论已在阶段 1、2 上验证过。

- 调度器的绑定结果直接写进 Workload（`worker_id` + `gpu_indexes` + `computed_resource_claim`）
- `distributed_servers.subordinate_workers[]` 提升为多行，按 `group_key` 关联
- `ModelInstance` 保留自己的完整生命周期（含 `ANALYZING` / `SCHEDULED` / `DOWNLOADING`），controller 映射执行子集

这一阶段有数据迁移（模型实例已发布），需要回填 + 回滚方案。

### 与 worker 控制回路抽取的关系

worker 侧共享库（`worker/controlloop/`）已抽出 `watcher` / `workload_state` / `writeback`，这三块与资源模型正交。**剩下的 `ports` 和 `reaper` 应该等阶段 1 之后再抽**：

- `ports` —— 统一需要命名端口模型，那正是 §3 的一部分
- `reaper` —— 如果 workload 成为单一资源，`workload_cleaner.py` 从「按 type label 分三支、每支一次 API list」退化成「列本 worker 的 workloads，与运行时对账」，注册制那套设计不需要存在

`backoff` 与资源模型无关，可随时抽（顺带修掉 serve 的退避指数只存内存、worker 重启后归零、且无重启上限的问题）。

## 5. 级联删除：从外键降级为 controller GC

**这是本提案风险最高的一处，三个阶段都受影响。**

缓存服务实例现在靠数据库外键 `ForeignKey("cache_services.id", ondelete="CASCADE")`。通用 `workloads` 表挂不上——`owner_id` 指向哪张表由 `owner_kind` 决定。

替代方案需要三层，缺一不可：

1. **删除时同步删** —— owner 的删除路径显式删掉自己的 workloads。正常路径，延迟最低。
2. **孤儿对账** —— controller 周期扫 workloads，owner 已不存在的删掉。覆盖第 1 步中途失败（进程崩溃、事务回滚）。`CacheServiceController._resync_loop`（`controllers.py:609`，60 秒）可以挂这里。
3. **worker 侧容器清理** —— 已存在（`workload_cleaner.py`），行 GC 之后由它兜底容器。

**第 2 步不能按「owner 不存在就删」一刀切**：owner 表读失败、或 owner 正在创建但 workload 先落库，都会被误判成孤儿而删掉正在跑的容器。需要宽限期（参考现有的 `WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD`），且只在 owner 表**成功读到**且确认不存在时才删。

本期共享缓存联测中的「级联删除回收滞后」就在这条链路上。**换掉外键之前，应先确认现有 GC 路径已经稳定**，否则是在已知不稳的机制上再加一层依赖。

## 6. 调用点清单（阶段 1）

阶段 2、3 的清单待各自立项时补。行号为撰写时的位置。

### 需要重写

| 位置 | 改动 |
|---|---|
| `server/controllers.py:368-860` `CacheServiceController` | 三个 watch + 扇出（`_reconcile_service:665`、`_desired_worker_ids:730`）+ 聚合（`_sync_service_aggregate:782`）改按 Workload + owner 过滤，语义不变 |
| `worker/cache_service_manager.py` | 改为 watch `workloads` + `owner_kind` 过滤 |
| `routes/cache_service_instances.py` | 整个 router 由 workloads 取代；`routes/routes.py:196` 同步改 |

### 需要改查询

| 位置 | 改动 |
|---|---|
| `server/cache_services.py:148` `_resolve_managed_endpoint` | 换类型；RUNNING 筛选下推 SQL |
| `exporter/exporter.py:476` | 换类型；**必须补 `owner_kind` 限定** |
| `routes/cache_services.py:314` `/{id}/instances` | 换类型，响应模型改名 |
| `routes/cache_services.py:398/420` 日志代理 | 换类型 |
| `worker/workload_cleaner.py` | 净删代码，见 §4 |

`exporter.py:476` 现在按 `state=RUNNING` **全局**取再在内存里分组。迁到 workloads 后不补 `owner_kind` 限定就会扫到模型实例和基准测试的行——**查询不报错，只是悄悄多返回数据**，是最容易漏的一处。

### 机械改动

`schemas/__init__.py:121-125, 338-342` 导出名；`codegen/generate.py` 的 `class_names` 加 `Workload`；重新生成 client。

### 前端（`gitlawr/gpustack-ui`）

只通过嵌套路由消费，**不碰顶层 instance 接口**：

- `src/pages/kv-cache/apis/index.ts:73` — `GET /cache-services/{id}/instances`
- `src/pages/kv-cache/apis/index.ts:86` — `DELETE /cache-services/{id}/instances/{instanceId}`（recreate）
- `src/pages/kv-cache/detail.tsx` + `components/service-instances`、`hooks/use-recreate-instance`

保持这两个嵌套接口的路径和响应字段名不变则**前端零改动**。唯一破坏性变化是 `port`/`metrics_port` 合并成 `ports`，在 `CacheServiceInstancePublic` 视图模型里拆回两个字段兼容即可——不值得为它改前端。

## 7. 风险与取舍

1. **缓存服务刚联测稳定。** 本期修的那批问题（事件丢失、写回竞态、级联删除回收滞后、非优雅终止）都验证在当前行模型上。换模型等于把那批验证重做一遍，且 §5 动的正是其中一条已知不稳的链路。
2. **模型实例的映射是纸面上定不下来的。** 加速器绑定与分布式分组只由它独占，阶段 0 的 POC 就是为此存在。跳过 POC 直接进阶段 1，等于让缓存服务替一个未经验证的模型背书。
3. **阶段 3 有真实数据迁移。** 模型实例已发布，需要双写、回填、回滚方案。

对冲这三条的是一条硬事实：**缓存服务是唯一不需要数据迁移的窗口**，发布之后阶段 1 的成本会与阶段 3 相同。

## 8. 待决

- 阶段 1 是否现在做（§7 的取舍；POC 已把「模型是猜的」这条风险消掉）
- §5 的 GC 三层是否先于阶段 1 单独加固
- `DELEGATED` 的资源预留：用 `managed` 字段，还是把多节点 claim 并到 leader workload
- 两个既存问题（DELEGATED 从属节点被判 ERROR、同机 follower 不被管理）是否独立立项修复
