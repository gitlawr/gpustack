# 设计：Workload 统一与资源账本收敛

状态：§1/§2 已实施(进度见 §2.5)，§3 为提案
范围：三类已有负载 + 未来的微调任务；之后是模型服务与 GPU instance 的账本统一

> 本文回答三个问题:为什么要统一、统一改了什么、以及在此之上如何让 k8s 集群把调度交还给 k8s。

---

## 1. 出发点

### 1.1 一个控制回路被抄了三遍

worker 上跑着三类由它启动的负载:模型实例、基准测试、共享缓存服务实例,微调任务在计划中。

容器执行层早就统一了——`gpustack_runtime` 的 `WorkloadPlan` 加 Docker/Podman/K8s deployer。**没有统一的是 worker 侧的控制回路**:事件监听、周期对账、卡死自愈、崩溃重启退避、状态写回及其重试、端口分配、日志通道、孤儿容器回收。每一类各写一遍。

代价不是代码量,是**每类负载各自踩一遍同样的坑**。缓存服务上线时修的那批问题(事件丢失、写回竞态、级联删除回收滞后、非优雅终止),模型实例那边有的修过、有的没有。加第四类就再来一轮。

### 1.2 一行两个写者

更深的问题是数据模型。以模型实例为例,一行里同时装着:

- **领域语义**:副本数、模型来源、用户可见的生命周期(分析中、调度中、下载中)
- **执行语义**:绑定到哪台 worker、哪几张卡、进程 PID、端口、容器状态
- **分布式拓扑**:`distributed_servers.subordinate_workers[]`,一个按索引寻址的嵌套列表

于是同一行被多方写:调度器写绑定,worker 写执行状态,供给子进程也写执行状态,控制器写生命周期。而写回是"读整行 / 改一个字段 / 写回整行",两个写者就会互相覆盖。

实施中真实发生过:一个分布式实例的从属节点报告 RUNNING,被主 worker 的整行回写抹掉,而从属只在启动时报一次,于是实例**永久卡在 STARTING**。这类问题在当前模型里只能靠打补丁,因为根因是"一行多个写者"。

### 1.3 为什么"统一"能解决

把执行语义抽成独立资源之后:

- **一行一个容器**。谁跑它、跑在哪、现在什么状态,都在这一行上,由一方写。
- **横向能力只实现一次**。重启策略、优雅终止、日志、指标目标发现、孤儿回收,对四类负载是同一套。
- **领域语义留在上层**。副本数、逐节点扇出、run-to-completion 由各自的 controller "编译"成 Workload,再把执行状态聚合回领域状态。用户面 API 不变,用户也不直接创建 Workload。

`restartPolicy` 是这个抽象成立的关键:`always` 是服务(模型实例、缓存服务),`never` 是任务(基准测试、微调)。两者的差别到这一层就只剩这一个字段。

---

## 2. 重构做了什么(高层)

```
Model ──────────┐
CacheService ───┼── controller 编译 ──> Workload ──> worker 控制回路 ──> 容器
Benchmark ──────┤         ▲                  │
(FineTuneJob) ──┘         └── 状态聚合 ───────┘
```

四件事:

**① 抽出共享控制回路**（`gpustack/worker/controlloop/`）
监听、对账、退避、写回、端口、日志、回收——八块,一处实现。抽取过程中修掉了原本只在某一类里存在的缺陷,例如退避计数只存内存(worker 重启即归零)、活跃集读取失败时孤儿回收会把一切当成孤儿。

**② 引入 Workload 资源**
只承载执行语义:spec(镜像、命令、端口、资源、重启策略)、binding(worker、加速器)、status(状态、PID、进度)。

关键取舍:

- `owner_kind` + `owner_id` 而非外键——目标表随 kind 变化,生命周期归 controller 管,不归数据库
- `group_key` / `group_index` / `role` 取代嵌套的从属列表,一个分布式实例是**一组行**,leader 在 `group_index=0`
- `reserved_claims` 表示"在别的节点上占了资源但 gpustack 不跑容器"(DELEGATED 模式),保住"有行就有容器"这个不变式

**③ 逐类迁移**
按影响面从小到大:缓存服务实例(未发布,直接替换掉原表)→ 基准测试 → 模型实例(已发布,需双写、对账、可回退)。

**④ 状态双向映射,先比对后切换**
worker 报执行状态,服务端把整组折叠回领域状态。折叠先只比对不采信,分歧记入日志;沉默才是可以切换的证据。

这套比对装置在真实环境拦下了**八个会造成事故的缺陷**,单元测试一个都没抓到。它们几乎全部是同一类:**两个写者、不同时刻**。举三个:

- 实例失败后被重新调度,而它的行还留着上一轮的 ERROR。折叠一旦生效就会把 ERROR 写回去,**把重启撤销掉**。
- worker 失联是服务端标记的,worker 自己不可能上报,所以行还停在 RUNNING。折叠会**撤销失联标记**。
- worker 先写实例、再镜像行,两次写之间读到的组合是自相矛盾的。折叠据此把一个**刚死掉的容器写回 RUNNING**。

还有一个不是它抓的,但后果正是**让它失效**:某处对 ORM 读回的字符串取 `.value` 会抛异常,折叠对每个实例崩溃,分歧日志恒空——闸门会读成"通过"。

**这条经验适用于 §3**:凡是把某个值的读取或写入换个来源,都要能同时跑两种读法并比较,而且**比较必须能自证它跑过**——只在不一致时打日志的比对,空日志同时意味着"一致"和"根本没跑"。

### 2.5 当前进度

| | 状态 |
|---|---|
| 共享控制回路 | 已完成,三类负载都在用 |
| 缓存服务实例 | 已完成,原表删除,实例即 Workload 行 |
| 基准测试 | 已完成,Workload 行已写入;领域行仍权威 |
| 模型实例 —— 编译成行 | 已完成 |
| 模型实例 —— 状态折叠 | 已完成并生效,单机与分布式均在真实环境验证 |
| 模型实例 —— 消费方迁移 | 进行中,服务端多数已切,调度器与 worker 侧待定 |
| 模型实例 —— 摘除重复字段 | 未开始(不可逆,需数据迁移) |

最后一步之前,领域资源上仍保留着执行字段的副本,两者由服务端在同一事务内派生保持一致。

---

## 3. 下一步:资源账本与调度的收敛

### 3.1 现状

**有三本账,不是两本。**

| 路径 | 容量从哪里读 | 谁决定放置 |
|---|---|---|
| 模型实例(常规) | gpustack 的 `Allocated`,只累加 ModelInstance 行 | gpustack 调度器 |
| 模型实例(**vGPU**,`model.gpu_type_selector`) | **operator 的 `Devices` CRD**,`status.groups[].accelerators[].remaining` | gpustack 选节点,operator 按需切分 |
| GPU instance | operator(`Instance` CRD,分配结果回写到 status) | operator / k8s |
| 缓存服务实例、基准测试 | **无** | 创建时指定 worker,或 per_node 铺开 |

三件事值得单独说:

**缓存服务对调度器不可见。** 它用 CUDA-IPC 映射 KV 缓冲区,实际占显存,但编译出的 Workload 行连 `computed_resource_claim` 都不填。调度器会把模型实例排到已被它占用的卡上。基准测试同理(多数是 CPU 负载,但"没声明"和"不占用"目前无法区分)。

**vGPU 那条路已经把容量账交给了 operator。** `VGPUResourceFitSelector` 读的是 `Devices` CRD 的 `remaining`,不用 `Allocated`。这不是权宜之计——分区是 operator 的 device-manager 按需切的,只有它知道还剩多少。

**所以"统一账本"在 k8s 上不是要新建一本,而是要把另外两个消费者也接进 operator 那本。** 常规模型实例和缓存服务是缺口,GPU instance 和 vGPU 模型实例已经在里面。

**另一个缺口:常规模型服务在 k8s 上绕开了 k8s 调度器。** worker 以 DaemonSet 运行,gpustack 自己选节点和卡,再由 runtime 的 k8s deployer 建容器。等于在 k8s 里跑第二个调度器,而它看不见非 gpustack 的 Pod——包括 GPU instance 的 Pod。

### 3.2 第一步:让账本覆盖所有 Workload(与集群类型无关)

Workload 表已经有 `gpu_indexes` / `computed_resource_claim` / `reserved_claims`,**缺的是生产者**:缓存服务和基准测试编译时不填。

要做的:

1. 缓存服务的 provider 声明里给出资源需求(显存、是否独占),编译时填入 Workload
2. 基准测试同理(多数是 CPU 负载,但要显式声明"不占卡"而不是留空)
3. 资源核算改为按 Workload 聚合、不按 owner_kind 过滤

第 3 条大部分已经就绪:`compute_worker_allocated_from_workloads` 已在比对模式下运行,只是过滤了 `owner_kind=MODEL_INSTANCE`。去掉过滤即可,前提是 1 和 2 完成——否则会把"没声明"当成"不占用",账反而更不准。

**这一步与 Docker/k8s 无关,两种集群都受益。**

### 3.3 第二步:按集群类型分开"谁做调度"

提案的核心:**Docker 集群自己调度,k8s 集群把调度交给 k8s。**

```
              评估（我们）          调度（谁）        binding 从哪来
Docker 集群    资源需求估算    →    gpustack 调度器  →  调度器写入
k8s 集群       资源需求估算    →    k8s / Volcano   →  从 Pod 观测回写
```

**为什么这个划分是对的**:模型需要多少显存是我们的领域知识(和后端、量化、上下文长度有关),k8s 不懂;而哪个节点有空、和别的工作负载怎么抢,是 k8s 的领域,我们不该在它旁边再算一遍——尤其算不准,因为看不见非 gpustack 的 Pod。

**为什么这次重构让它变便宜**:Workload 已经把 **binding 和 spec、status 分开**了。谁来填 binding 是一个可以按集群类型不同的细节,**读者不用改**——资源核算、放置查询、日志路由都只读 binding,不关心它从哪来。

**而且这不是新架构,是把已有的一条路推广开。** GPU instance 现在就是这么工作的:gpustack 写 `Instance` CRD,集群里的 operator 变成 Pod,gpustack 从 CRD 的 status 读回 `node_name`、设备分配、Pod IP。常规模型服务反而是那个特例。

具体形态:

- **k8s 集群**:controller 把 Workload 编译成 Pod/Job(资源需求作为 requests/limits),不填 `worker_id`/`gpu_indexes`。k8s 调度完成后,从 Pod 的 `nodeName` 和设备分配**观测回写**这些字段。这条路径和现有的状态折叠是同一个形状(观测 → 聚合 → 写回领域资源),已有机制可复用。
- **Docker 集群**:维持现状,调度器写 binding。
- **通过 k8s 机制部署的东西**(helm chart 等):把它的 Pod **同步入库为 Workload**,`owner_kind` 标记来源。这样账本自动覆盖它们,不需要它们知道 gpustack 的存在。

设备粒度上已有先例:`_sync_vgpu_allocation` 已经在从设备插件写的注解里读回真实的设备分配,证明"观测回写"这条路是通的。

### 3.4 GPU instance 并入同一本账

读过 `gpu_instances/` 之后,结论比预想的顺:**它已经是这个形状了。**

`GPUInstance` 上有 `GPUInstanceSpec`(镜像、命令、端口、env、resources、卷)和 `GPUInstanceStatus`(`phase` / `node_name` / `pod_ips` / `allocations`),binding 是从 CRD status **观测回来的**,不是我们写的。换句话说 GPU instance 与 Workload 的差别不在数据模型,**在运行时是谁**:一个由集群里的 operator 跑,一个由我们的 worker 跑。

那些看起来"多出来"的东西并不需要 Workload 承载:

| 资源 | 形态 | 归属 |
|---|---|---|
| 持久卷、卷类型 | 独立 CRD + 独立表 | owner 侧领域概念,编译进 spec 的 `volume` |
| SSH 公钥 | 同上 | 同上 |
| instance type / flavor | 集群级 CRD | 目录,不是实例的一部分 |

这和 Model 与模型实例 Workload 的关系是同一种:目录、配置、凭据留在 owner 侧,**编译**成执行层能懂的东西。

所以并入的路径是:GPUInstance 成为第五种 `owner_kind`,它的 controller 把 spec 编译成 Workload,binding 从 CRD status 观测回写——正是 §3.3 给 k8s 定的那条路。

并入之后:

- **一本账**:`Allocated` 由所有 Workload 聚合,不区分 owner
- **一套回收**:孤儿回收和级联删除的三层机制同样适用
- **一套横向能力**:日志、指标目标发现、优雅终止不用再实现第二遍

**真正的开放问题只剩一个,而且不在数据模型上**:GPU instance 的运行时是 operator,`WorkloadPlan` 走不到它。要么 Workload 的 spec 增加"由谁执行"这一维(我们的 worker / 集群 operator),要么承认这类 Workload 的 deployer 就是"写 CRD 并观测"。后者更接近现状,也更像 §3.3 里 k8s 集群本来就要做的事——**两者其实是同一件事的两个说法**。

### 3.5 复杂集成只在 k8s 做

Volcano、JobSet 这类只在 k8s 生态存在的东西,不必在 Docker 集群上找对应物。Workload 的 spec 里带调度提示(队列、gang 大小、优先级),**k8s deployer 映射,Docker deployer 忽略**。

分布式实例正好落在这里:`group_key` 天然对应 gang,而 `INITIALIZE_LATER`(从属节点等主节点初始化后再启动)这类启动顺序协调,在 k8s 上应该交给 JobSet/LeaderWorkerSet,而不是继续由我们的 worker 互相等待。

### 3.6 风险与顺序

按依赖排序,每一步都能独立发布:

| 步骤 | 依赖 | 可回退 |
|---|---|---|
| 1. 缓存服务/基准测试声明资源 | — | 是 |
| 2. 账本按 Workload 聚合、去掉 kind 过滤 | 1 | 是(比对模式) |
| 3. k8s 集群改为观测回写 binding | 2 | 是(按集群类型) |
| 4. helm 部署的 Pod 同步入库 | 3 | 是 |
| 5. GPU instance 并入 Workload | 3 | 需数据迁移 |

第 5 步排在第 3 步之后不是偶然:第 3 步做完之后,"binding 从 CRD/Pod 观测回写"已经是一条走通的路径,GPU instance 并入就只是再挂一个 `owner_kind`,而不是同时发明机制和迁移数据。

**最大的风险是账本口径**:如果一个负载既被"预留"(我们写 binding)又被"观测"(从 Pod 读回),会重复计数。所以每种集群类型必须只有一个 binding 的来源,这也是 §3.3 那张表要明确到"binding 从哪来"这一列的原因。

**第二个风险是 k8s 上的准入**:交给 k8s 调度之后,我们不再能保证"排得下"。资源评估仍要做,但它从"分配"降级为"准入检查 + requests",排不下时是 Pod Pending 而不是我们拒绝。这是行为变化,需要在 UI 上如实呈现,不能假装还是原来的语义。
