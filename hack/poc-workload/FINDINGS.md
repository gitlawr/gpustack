# 阶段 0 POC 结论

对应 `docs/proposals/workload-resource.md` 的阶段 0。目的是在纸面之外验证：提出的 Workload 资源模型能不能装下一个分布式模型实例。

运行：`pytest hack/poc-workload`（31 个用例）。这里的代码是废弃件，不进 CI（`pytest.ini` 的 `testpaths = tests`）。

方法：把真实的 `ModelInstance` 编译成 Workload 组，再把组的执行状态聚合回去，两个方向都对着**生产代码**断言——`get_deployment_metadata`、`_dispatch_model_instance_event`、`_get_main_worker_distributed_state`、`_assign_ports`，其中一个探针直接调用 `ServeManager.sync_model_instances_state`。

## 结论一：模型立得住，但要改四处

### 1. 字段覆盖完整 ✅

`ModelInstanceSubordinateWorker` 声明的 15 个字段全部有去处，只有 `total_gpus` 不带（可由 `gpu_indexes` 推出）。没有装不下的东西。

`test_every_subordinate_field_lands_somewhere` 用 `model_fields` 反射做断言，以后加字段忘了映射会直接红。

### 2. 容器名精确对齐 ✅

编译出的 workload 名与 `get_deployment_metadata` 逐个相等（leader 用实例名，follower 用 `<name>-f<i>`）。**迁移不会孤立任何在跑的容器**——这是整件事的前提，先验证掉。

### 3. `port` 必须与 `ports[0]` 合并 ⚠️ 需改 schema

`_assign_ports` 里 `mi.ports = [mi.port]` 然后 extend，所以 `port` 恒等于 `ports[0]`，**它们是一个端口不是两个**。

第一版编译器把两者分别命名，产出 `{"service": 8000, "port0": 8000, ...}`——同一个号码挂在两个名字下，迟早漂移。已修，`test_leader_port_is_not_duplicated_across_names` 钉住。

**更重要的是：`ports[1:]` 无法通用命名。** 布局取决于 backend 与 executor：

| 场景 | ports 布局 |
|---|---|
| vLLM + mp | `[HTTP, DP-RPC, master-port, VLLM_PORT, ..., connecting]` |
| vLLM + ray | `[HTTP, DP-RPC(仅 dp>1), connecting]` |
| 其他 backend 分布式 | `[HTTP, connecting]` |
| 非分布式 | `[HTTP]` |

connecting port 恒在**末位**，所以按位置命名必错。

> **给 schema 的结论**：Workload 只有 `ports` 一个命名映射，不要单独的 `port` 列。第一个端口通用命名为 `service`；其余由**各 backend 自己的 compiler** 命名，通用编译层不猜。

### 4. 启动次序可以干净地上移到 controller ✅

现在的次序是每个 worker 在每个事件里重新推导（`_dispatch_model_instance_event` 读整个实例）。改成 workload 上的依赖后，controller 决定一次，worker 只会看到「已经允许启动」的 workload。

需要**两种门**，一种不够：

- `STARTED`（到达 STARTING 或更后）—— INITIALIZE_LATER 里 follower 等 leader
- `READY`（到达 RUNNING，或失败告终）—— follower 等前一个 follower（即 `_dispatch_model_instance_event` 里那段 phantom-read 保护）

`test_blocked_workloads_release_in_dependency_order` 验证了逐级放行。

**顺带发现：`RUN_FIRST` 在生产路径上是死的。** 调度器只对 vLLM / SGLang / MindIE 设 `INITIALIZE_LATER`，其余一律留默认 `DELEGATED`，没有任何地方设 `RUN_FIRST`——只有 `serve_manager` 的处理分支和测试引用它。迁移时可以顺手确认是否保留。

### 5. DELEGATED 把两个概念混在了一起 ⚠️ 需改 schema

`subordinate_workers[]` 同时表示两种东西：

- **gpustack 要跑的容器**（INITIALIZE_LATER / RUN_FIRST）
- **gpustack 只做了资源预留**（DELEGATED，容器归别的框架管）

Workload 是前者。所以模型里必须能区分，否则第二种会被当成第一种。

**这不是推测，探针打到生产代码验证过**（`test_delegated_subordinate_is_marked_error_by_the_current_sync`）：DELEGATED 的从属 worker 上，事件路径提前返回所以从不建容器，但对账循环不知道这回事——查不到 workload，直接把从属状态写成 ERROR。

链路是通的：`is_gguf_model()` 按模型来源和 `.gguf` 后缀判断（不看 backend）→ GGUF selector 会填 `subordinate_workers` → 调度器不给它设 INITIALIZE_LATER → 留在 DELEGATED。

**但可达性待产品确认**：v2 的 `BackendEnum` 里没有 llama-box/GGUF 项，`_SERVER_CLASS_MAPPING` 也没有对应 server 类，所以 GGUF 模型在 v2 能否真的走到多 worker 调度，我从代码判断不了。

无论可达与否，结论都一样，因为 **DELEGATED 是 `mode` 的默认值**——将来任何带从属节点的新 backend 都会继承这个行为。

> **给 schema 的结论**：加 `managed: bool`（POC 采用），或者把多节点资源预留并到 leader workload 的 claim 上。前者改动小，后者更接近「Workload 就是容器」的纯粹语义。二选一，但不能不选。

### 6. leader 与 follower 可能落在同一 worker ⚠️ 影响唯一约束

`get_deployment_metadata(worker_id)` 按 worker 查、只返回一个结果，所以今天与 leader 同机的 follower 是**够不到的**：它的容器永远不会被管理。

按 `group_index` 编译（而不是按 worker 反查）就没有这个歧义。`test_leader_and_follower_may_share_a_worker` 同时展示了新旧两种行为。

> **给 schema 的结论**：唯一约束必须是 `(owner_kind, owner_id, worker_id, group_index)`，不能是 `(owner, worker)`。提案里已经是这样，此处是证据。

### 7. `should_update` 不属于聚合 ⚠️ 影响接口

`_get_main_worker_distributed_state` 既折叠状态、又通过比对实例当前状态决定**要不要写**。这两件事混在一个返回值里，每个消费者都得解释那个 flag。

POC 的 `aggregate_group_state` 只折叠，写抑制留给调用方。`test_aggregation_does_not_decide_whether_to_write` 展示了两者在「实例已是 ERROR」时的分歧。

聚合本身与生产行为逐项对齐：两个 follower 状态的 16 种组合全部参数化对比 `_get_main_worker_distributed_state`，结果一致。

## 结论二：`download_progress` 归 Workload

提案 §2.4 留的问题，结论是**放在 Workload 上**（POC 里叫 `progress`）。

理由：从属节点的下载进度天然是**逐 workload** 的状态。留在领域侧就得重新引入一个按 worker 索引的列表——正是这次要消除的那种结构。

代价是一个需要接受的推论：**Workload 行在绑定时创建，不是在容器启动时创建**。模型文件下载发生在容器存在之前，所以行必须先于容器存在。这与 K8s 一致（Pod 先于其容器存在），也与 `PRE_EXECUTION_STATES` 的划分自洽：实例处于 `DOWNLOADING` 时，它的 workload 处于 `pending`。

`test_pre_execution_instance_states_have_no_workload_counterpart` 钉住了这条边界：`PENDING` / `ANALYZING` / `SCHEDULED` / `INITIALIZING` / `DOWNLOADING` 五个实例状态都映射到 workload 的 `pending`，Workload 不复述领域生命周期。

## 对提案的净修改

1. §3 表定义：去掉独立的 `port` 列，只留 `ports` 命名映射（结论 3）
2. §3 表定义：加 `managed: bool`（结论 5）
3. §3 表定义：加 `progress`（结论二）
4. §2.4 的悬置项关闭
5. §2.2 补一行：`start_after` 需要两种门（结论 4）
6. 唯一约束的理由从推测变成证据（结论 6）

**没有发现需要推翻模型的问题。** 六项修改全是加减字段，不动结构。

## 与迁移无关、但已存在的两个问题

POC 顺带证实的，无论提案做不做都成立：

1. DELEGATED 从属节点被对账循环写成 ERROR（结论 5），可达性待确认。
2. 与 leader 同机的 follower 永远不被管理（结论 6）。

## 文件

- `workload.py` —— 提出的资源，纯 dataclass，不接数据库
- `compile_model_instance.py` —— 双向：编译 + 依赖放行 + 状态聚合
- `test_poc.py` —— 31 个用例，全部对着生产代码断言
