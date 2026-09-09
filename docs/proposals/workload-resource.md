# 提案：通用 Workload 资源

状态：实施中 —— 阶段 0/1/2 完成，阶段 3 的折叠已在真实环境生效（见 §4）
目标：模型实例、基准测试、缓存服务实例三类负载统一编译到一个 Workload 资源
影响范围：新增 `workloads` 表；三类负载的用户面 API **保持不变**

> 实施中新增的结论散在 §2.3、§2.6、§3、§4 各阶段小节里，都标了「实施中发现」。它们是这份提案里唯一不来自设计推演、而来自把代码写出来的部分——其中分量最重的几条来自真实环境，不是测试（§4 阶段 3）。

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
| `gpu_indexes` / `gpu_addresses` / `gpu_type` | **需要** | — | —（吃全部 GPU） |
| `computed_resource_claim` | **需要**（调度器的资源决策） | — | — |
| `reserved_claims` | **需要**（DELEGATED 的多节点预留，见 §2.5） | — | — |
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

#### 映射是两张表，反向是偏函数（**实施中发现**）

写这一节时以为映射可以按名字走：两个枚举都有 `STARTING`，看起来对应。**它们指的不是同一个时刻**：

| 实例状态 | 谁写 | 含义 | → workload |
|---|---|---|---|
| `INITIALIZING` | worker | 已拉起供给进程（同时写 `pid`） | `starting` |
| `STARTING` | **服务端** | 模型文件就绪，还没启动任何东西 | `pending` |

按名字映射的后果是 `INITIALIZING`（容器在跑）被映成 `pending`（无容器），而 `STARTING → starting` 这条**永远不会发生**——写它的是服务端，不经过 worker 的镜像。leader 的 workload 于是只有 `pending → running` 两个状态。

现在是两张独立的表（`_TO_WORKLOAD_STATE` / `_TO_INSTANCE_STATE`），任一方向都不从另一方推导，并有覆盖全部执行态的往返测试。

更要紧的推论：**正向是全映射，反向只有 3/6**。多个领域状态塌缩到同一个 workload 状态，反向就不是函数。折叠在这些位置**弃权**而不是猜：

```
pending / starting → 弃权（两个实例状态都映到这里）
running / unreachable / error → 映射回去
succeeded → 弃权（服务型实例没有这个概念）
```

代价是折叠对启动全过程沉默，实例的 `INITIALIZING` 和 `STARTING` 仍由领域侧自己写。

### 2.4 `download_progress` 的归属（POC 已结论）

放在 Workload 上（`progress`）。从属节点的下载进度天然是**逐 workload** 的状态，留在领域侧就得重新引入一个按 worker 索引的列表，正是要消除的那种结构。

代价是一个需要接受的推论：**Workload 行在绑定时创建，不是在容器启动时创建**——下载发生在容器之前，行必须先于容器存在。这与 K8s 一致（Pod 先于其容器存在），也与 §2.3 自洽：实例处于 `DOWNLOADING` 时，它的 workload 处于 `pending`。

### 2.5 DELEGATED 的资源预留并入 leader（已决定）

阶段 0 的 POC 发现 `subordinate_workers[]` 混了两件事：**gpustack 要跑的容器**（INITIALIZE_LATER / RUN_FIRST），和**只做了资源预留**（DELEGATED，容器归别的框架管）。POC 里用 `managed: bool` 区分，同时保留了两种行。

**决定不走这条**：一个 Workload 行就是一个 gpustack 要跑的容器，不留「存在但不跑」的行。DELEGATED 的从属节点不产生 Workload，它们的资源占用并入 leader workload 的 `reserved_claims`——一个「(worker_id, gpu_indexes, computed_resource_claim)」列表，表示这个 workload 替别的框架在别的节点上占住了什么。

代价是资源核算要同时读 `computed_resource_claim`（本行自己的）和 `reserved_claims`（替别人占的），调度器那侧要一并改。换来的是 Workload 的语义不打折：**有行就有容器**，worker 不需要一个「这行你别管」的分支。

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
  reserved_claims                                -- JSON，见 §2.5

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
  started_at                                     -- 容器何时开始，见下
  healthy, last_check_at
  progress                                       -- 供给进度，见 §2.4
```

`started_at`（**实施中发现**）：`active_deadline_seconds` 必须从某个时刻起算，而行的 `created_at` 是它**被创建**的时刻——对排队等前一个跑完的基准测试来说，两者差着整段排队时间。对应 Kubernetes Job 的 `status.startTime`，由 worker 在容器启动时写入。

**枚举必须自定义 `__str__` 返回 value**（**实施中发现**）：生成的客户端用 `str(属性)` 比对 `str(查询值)` 过滤 watch 缓存，而裸的 `class X(str, Enum)` 渲染成 `"X.MEMBER"`——**任何按枚举字段的缓存过滤会静默返回空列表，不是报错**。缓存服务的对账循环正是这样读实例的，读到空的含义是「服务器不再报告这些」，也就是回收容器的输入。`CacheServiceStateEnum` 早就为此定义了 `__str__`。

**读缓存时不要传 `page` 参数**（**实施中发现**）：客户端只要看到 page 就整个跳过缓存直接打 API。高频读（每次状态写回）传 `page=-1` 会变成每次一个全量 HTTP 请求。

`ports` 是唯一的端口载体，**不要另设 `port` 列**：`_assign_ports` 里 `mi.ports = [mi.port]` 然后 extend，两者恒为同一个端口，拆成两列只会让它们漂移。第一个端口通用命名为 `service`，其余由各 backend 自己的 compiler 命名——`ports[1:]` 的布局取决于 backend 与 executor（vLLM/mp 是 DP-RPC + master-port + VLLM_PORT，vLLM/ray 仅 dp>1 时有 DP-RPC，其他 backend 没有），且 connecting port 恒在末位，通用层猜不了。

`reserved_claims` 承载 DELEGATED 的多节点资源预留，见 §2.5。

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

### 阶段 1：缓存服务实例 —— 已完成 ✅

免迁移窗口用掉了：建 `workloads` 表、删 `cache_service_instances`，不是数据迁移。

**唯一完整迁移的一类**——Workload 就是权威，没有影子阶段。controller 编译并聚合状态，`cache_service_manager` watch `workloads`（filter `owner_kind`），前端零改动。

兑现的收益与预测一致：租户过滤不再对父表子查询（workload 自带 `owner_principal_id`）；端点解析的 RUNNING 筛选下推到查询（它每次模型实例调度都跑）；exporter 必须声明 `owner_kind`，否则会连另外两类的行一起收走——**这是整个迁移里唯一不会大声失败的错误**。

容器名从 `cache-svc-{服务}-i{实例}` 变成 `cache-svc-{服务}-w{worker}`：name 就是容器名、必须先于行存在，所以不能用行 id，而 (服务, worker) 本来就是身份。

### 阶段 2：基准测试 —— 已完成 ✅

**实际在阶段 3 之后做**，导致 `restart_policy` 和 `active_deadline_seconds` 一度没有消费者。

`restart_policy = never` 第一次有了填写者:基准测试退出是**完成**不是故障。三个状态没有容器可描述——`pending` / `queued` / `stopped`——都留在 workload 的 pending；`queued` 尤其说明问题,那是服务端决定 worker 下一个跑哪个,不是执行。

**`active_deadline_seconds` 的消费顺带修掉一个洞**(**实施中发现**):worker 原本从内存时间戳起算超时,worker 中途重启后时间戳为 None,`_is_benchmark_timed_out` 从此永远返回 False,跑飞的基准测试一直占着 GPU。deadline 改从 workload 行取即可幸存。这也是 `started_at` 被识别出来的原因(见 §3)。

状态镜像已补齐,与模型实例同形。仍未做:折叠回 Benchmark 行(它仍是权威)。

**明确不下沉的两样**保持原判:结果收集(领域逻辑)、串行队列(`_active_benchmark_id` 是准入语义,留在 server 侧)。

### 阶段 3：模型实例 —— 折叠已生效 🔶

提案原本只有三条要求,实施时拆成四步:

| 步骤 | 状态 |
|---|---|
| 1. 编译成 Workload 行 | ✅ 写入,无人消费 |
| 2a. 执行状态镜像到 Workload | ✅ 双写,实例行仍权威 |
| 2b. 折叠回实例 | ✅ **已翻开关,单机与分布式均验证** |
| 3. 调度器直接写绑定、`subordinate_workers[]` 变成行 | ⬜ 未开始 |
| 4. 从 ModelInstance 摘字段(真实数据迁移) | ⬜ 未开始 |

**为什么 2b 不直接翻**:它一旦生效就是实例状态的唯一来源,而折叠错了实例永远出不了 STARTING——第一现场会是生产环境。所以折叠照常运行但只记录分歧,**沉默就是可以翻转的证据**。

折叠逻辑对着 `_get_main_worker_distributed_state` 逐项验证:两个 follower 的 16 种状态组合全部与生产实现比对,失败消息则由一个同时驱动两边、比对输出的测试钉住。

`GPUSTACK_MODEL_INSTANCE_STATE_FROM_WORKLOADS=true` 是开关,它随步骤 4 一起消失。

步骤 3 的规模需要预先知道:**68 个文件引用 `ModelInstance`**,`distributed_servers` 散在 20 个文件里,含调度器、全部候选选择器、放置打分器、资源核算和四个 backend。

#### 这套比对装置抓到了什么

真实环境跑下来,闸门在翻开关前拦下 **5 个会造成事故的缺陷**。单测一个都没抓到——它们全部来自"两个写者、不同时刻"这个结构,而不是某段逻辑本身:

| 分歧 | 若直接翻开关 |
|---|---|
| 实例 `SCHEDULED` vs 折叠 `ERROR` | 失败后重新调度被**撤销**(workload 还留着上一次运行的 ERROR) |
| 实例 `UNREACHABLE` vs 折叠 `RUNNING` | worker 失联标记被**撤销** |
| 实例 `STARTING` vs 折叠 `INITIALIZING` | 见 §2.3,映射按名字走 |
| `state_message` `''` vs `None` | 每个健康实例都报分歧,闸门永远过不去 |
| 标记失联时多标了 `STARTING` 的 leader | 修第 2 条时引入的,标记范围比服务端既有规则宽 |

**规律**:凡是 **worker 不参与**的状态变更(重新调度、worker 失联),镜像就跟不上——镜像只在 worker 写回时触发。这类只能由服务端在写实例的同时一并写 workload 行,而且范围要逐字照搬既有规则(最后一行就是没照搬的后果)。第 4 步之后问题消失:那时 workload 是执行状态的唯一去处。

还有一个不是这套装置抓的,但它的后果正是**让这套装置失效**:`.value` 作用在 ORM 读回的字符串上会抛 `AttributeError`(列声明成 `String`,读回是 `str`,而 API 校验回来的是枚举)。折叠对每个实例崩溃,分歧日志恒空——**闸门会读成通过**。根因已在 `EnumString` 里消除(读出时转回枚举),但仓库里另有 19 列同样的不对称。

#### 仪表本身的三次修正

闸门的判据是"日志为空",所以**空日志必须只有一种解释**。三次发现它不是:

1. **只在分歧时打日志** —— 空日志同时意味着"处处一致""控制器没跑""事件没到"。加了计数 tally。
2. **弃权不计入 tally** —— 启动全程弃权时仍然一行不打,和控制器停摆输出相同。弃权计入节奏。
3. **一致只有总数,不分状态** —— 只见过 `running` 的一轮,和覆盖了 error/unreachable 的一轮读起来一样;分布式和单机也都只贡献 `running`。改成按状态计数,并单列 `of which distributed`。

**判据本身也改过一次。** 起初是"等 N 秒后是否仍有差异"——但比较两个最终一致的写者,任何在窗口内自行消解的东西都能通过,包括两个写者来回翻转这种真缺陷,调大窗口只会让盲区更大。改成看**收敛到哪里**:实例最终是否到达折叠提出的那个值。到了就是折叠对且更早(读到失联的瞬间就报,不必等主 worker 那一轮);没到就是提议从未成真,单列为 `overtaken`——旧判据会把它当成干净的 settle 放过。时间参数还在,但降级成"隔多久再问",不再是判据。

### 与 worker 控制回路抽取的关系 —— 已完成 ✅

8 块全部抽出(`gpustack/worker/controlloop/`,约 1100 行):

`watcher` / `workload_state` / `writeback` / `backoff` / `container_logs` / `launcher` / `reaper` / `ports`

后两块原本卡在资源模型,阶段 1 之后解锁。`reaper` 最终没有退化成单一对账——只迁了缓存服务,三个 kind 的分支仍是实的,所以做成了注册制。

抽取过程中修掉的既有缺陷:serve 的重启退避指数只存内存(worker 重启后归零)、孤儿回收在活跃集读取失败时会当成「没有活跃的」、三个供给子进程的前导重复了三遍。

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

1. ~~**缓存服务刚联测稳定。**~~ 已迁移并在真实环境验过:删除的回收从最多约 7 分钟降到秒级,拉镜像过程第一次可见。原风险(那批验证要重做)兑现为实际重做,代价符合预期。
2. **模型实例的映射是纸面上定不下来的。** 加速器绑定与分布式分组只由它独占，阶段 0 的 POC 就是为此存在。跳过 POC 直接进阶段 1，等于让缓存服务替一个未经验证的模型背书。
3. **阶段 3 有真实数据迁移。** 模型实例已发布，需要双写、回填、回滚方案。

对冲这三条的是一条硬事实：**缓存服务是唯一不需要数据迁移的窗口**，发布之后阶段 1 的成本会与阶段 3 相同。

## 8. 待决

**已决定的**（记录在此,免得重新讨论）:

- 阶段 1 现在做 —— 已完成
- GC 三层先于阶段 1 单独加固 —— 已完成(§5)
- `DELEGATED` 的资源预留并入 leader 的 `reserved_claims`,不用 `managed` 字段 —— 见 §2.5

**待决的**:

- **绑定是否应当不可变** —— 现在重新调度是在同一个 ModelInstance 上改绑定(`scheduler.py:376`),`sync_model_instance_workloads` 随之**原地更新已有的行**。k8s 的 Pod 不是这样:绑定不可变,重新调度建新 Pod。若 workload 行也如此(重绑=新建行、旧行回收),"两个写者改同一行"这一类竞争在构造上消失,也和后续 Pod 生命周期映射一致。代价是行 id 会变(worker 需重新解析)、回收面变大、日志文件命名要跟着调整。**这个方向是评审中提出的,值得在阶段 4 之后单独立项。**

- **`INITIALIZING` 是否可以从用户可见的生命周期里消失** —— 第 4 步之后 worker 不再写实例,而 `INITIALIZING` 正是它写的,序列会变成 `DOWNLOADING → STARTING → RUNNING`。倾向接受(`STARTING` 已经表达"正在起",两者语义重叠),但这是产品可见的变化,需要在第 4 步之前定。若不能少,得让服务端在 workload 转 `starting` 时补写。
- **`PATCH /workloads/{id}/status`** —— 服务端写 spec、worker 写 status 这条边界目前只是**约定**:worker 的写回是 GET 整行 / 改字段 / PUT 整行,会覆盖服务端并发写入的 spec(重新调度换了 GPU 就会丢)。要求走生成的客户端,所以需要新端点 + 重新生成客户端。这是阶段 3 之后依然存在的缺陷,不是过渡期产物。
- **基准测试的折叠**是否也做(目前 Benchmark 行仍权威)
- **两个既存问题是否独立立项**:DELEGATED 从属节点被对账循环写成 ERROR(**可达性待产品确认**:v2 的 `BackendEnum` 里没有 llama-box,GGUF 分布式能否真走到多 worker 调度,代码判断不了);与 leader 同机的 follower 永远不被管理
- **serve 无重启上限**:缓存服务 5 次后 park 到 ERROR,模型实例永远重启。判断是产品语义不是缺陷(推理服务的失败常常是外部的,且用户能用 `restart_on_error` 关掉),未改
- **serve 的退避不持久化**:`restart_count` 是日志文件的代号必须单调,不能兼任连续崩溃计数,所以持久化需要另开一列。当期明确不做

## 9. 验证清单

### 已在真实环境验过

| 项 | 结果 |
|---|---|
| 缓存服务删除 | 容器秒级消失(原来最多约 7 分钟) |
| 缓存服务拉镜像 | 供给日志可见——这是统一启动模型的直接收益 |
| 折叠比对(单机) | `running` / `error` / `unreachable` 全部一致 |
| 折叠比对(双机分布式) | 同上,`_distributed_override` 三个分支都走到 |
| 翻开关后 | 单机与分布式均能正常到 RUNNING,不卡 STARTING |

判读用的一行:

```
Workload fold [authoritative corrected=0]: agreed={running=6, error=1, unreachable=4}
  of which distributed={...} declined={instance_not_executing=20, leader_starting=33}
```

模式写在最前面——两种模式在一切正常时计数相同,不标出来就无法判断折叠是在决定还是只在旁观(第一次读的时候就误判了)。

### 尚未验证

1. **基准测试超时** —— 设了 `benchmark_max_duration_seconds` 后中途重启 worker,超时应仍触发
2. **容器日志 `previous=true`** —— 能看到上一次的容器日志
3. **回归面** —— 基准测试行为应完全不变(权威数据源没动)
