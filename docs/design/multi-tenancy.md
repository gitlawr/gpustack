# 多租户设计

## 概述

GPUStack 当前只区分管理员（admin）和普通用户两类身份，普通用户的能力局限于通过 `ModelRoute` 调用管理员上架的模型推理。随着 **GPU 实例管理**（K8s Pod + SSH 访问）功能的引入，普通用户需要能够独立创建和管理自己的工作负载，且不同用户/部门之间的资源必须互相隔离。

本设计在现有 User/Admin 两层身份之上引入 **Organization → UserGroup → User** 三层租户模型：

- 用户可加入多个 Organization（跨组织成员）
- **租户资源**（Model / ModelRoute / ModelInstance / ApiKey 等）归属单个 Organization，跨 Org 默认硬隔离
- **基础设施**（Cluster / CloudCredential / WorkerPool / Worker）始终归属一个 Org：admin 在 "All" 模式下创建时默认落到 Default Org，`organization_id NOT NULL`。跨 Org 共享通过 `cluster_access` 表达，不靠 NULL 表示 Global。InferenceBackend 仍是 Hybrid（admin 维护的 Platform catalog 行 + 各 Org 的扩展行共享 `backend_name`）
- GPU 实例配额通过 K8s `ResourceQuota` 落地
- 平台公共资源通过 ModelRoute 显式发布到其他 Org

GitHub Issue: TBD

## 需求

### 必须实现

| 编号 | 需求 |
|---|---|
| F1 | 引入 Organization 作为硬隔离与计费单元；用户可加入多个 Org。**租户资源**（Model / ModelRoute / ModelInstance / ApiKey 等）强制归属单个 Org；**基础设施**（Cluster / CloudCredential / WorkerPool / Worker）也强制归属（admin 在 "All" 模式默认落到 Default Org），跨 Org 共享通过 `cluster_access` 表达；InferenceBackend 是 Hybrid（Platform catalog + Org override 共享 `backend_name`），由平台 admin 维护内置目录 |
| F2 | 引入 UserGroup（Org 内子单元）以支持部门/团队级资源共享 |
| F3 | 普通用户仅可见/管理在当前 Org 上下文中可见的资源；v1 中所有用户创建的资源都 owner=org，等价于"全 Org 可见"。schema 上保留 `owner_type/owner_id` 三档（org/group/user），为后续精细化预留 |
| F4 | 管理员可将 Cluster 授权给 Org / Group / User |
| F5 | 用户可在多个被授权的 Cluster 上创建 GPU 实例（K8s Pod） |
| F6 | 每个 (Org × Cluster) 自动 provision 一个 K8s namespace，GPU 实例落入对应 namespace |
| F7 | 管理员可为 (Org × Cluster) 设置 GPU/CPU/内存/Pod 数量配额，落地到 K8s `ResourceQuota` |
| F8 | 同一用户可在不同 Org 之间切换上下文，行为完全独立 |
| F9 | 现有 `ModelRoute` 扩展为支持 principal（org/group/user）粒度的发布 |
| F10 | 现有功能（admin 上架模型、推理调用、Worker、Cluster 等）平滑迁移，admin 体验不变 |

### 不在本设计范围内

| 编号 | 项 | 原因 |
|---|---|---|
| N1 | GPU 实例 CRD 设计 | K8s 团队负责 |
| N2 | Backend ↔ K8s API 反向代理实现 | 单独设计 |
| N3 | UserGroup 级别配额 | 一期不做，预留字段与扩展点 |
| N4 | 模型推理并发数/调用速率配额 | 走计费/限流路径，不在本期 |
| N5 | SSH 网关路由细节 | 单独设计 |
| N6 | 跨 Org "默认共享" 资源（取代旧 `is_system` 概念） | 与硬隔离/计费冲突；统一通过 ModelRoute 显式发布 |

## 解决方案

### 整体思路

**身份层、平台基础设施层与租户资源层解耦**：

- **身份层**：在 `User` 之上新增 `Organization`、`OrganizationMembership`、`UserGroup`、`UserGroupMembership` 描述"用户在组织中的归属与角色"。
- **平台基础设施层**：`Cluster`、`Worker`、`CloudCredential` 不带 `organization_id`，由平台 admin 全局管理；通过新增的 `cluster_access` 表把 cluster 显式授权给 principal（org / group / user），同一 cluster 可同时授权给多个 principal。
- **租户资源层**：所有租户资源（`models` / `model_instances` / `model_routes` / `api_keys` 等）增加 `organization_id`（强制）+ `owner_type/owner_id`（精细化），形成"硬隔离 + 精细化可见性"的双层归属。
- **K8s 资源层**：通过 `tenant_namespaces`、`tenant_quotas` 把租户语义投射到 K8s namespace + ResourceQuota，由 reconciler 维护一致性。

**为什么 Cluster 不归属单个 Org**：Cluster 代表物理/云端的计算池，先天是共享基础设施——admin 把同一台 K8s Cluster 同时授权给"研发"和"算法"两个 Org 是常态。如果硬塞 `organization_id` 进 `clusters`，要么逼 admin 为每个 Org 复制 cluster（资源浪费），要么需要额外的"共享 cluster"豁免逻辑（破坏硬隔离的简单性）。`cluster_access` 这张多对多授权表是更直接的表达。Org 真正"独占"的是该 Org 在该 Cluster 上对应的 K8s namespace 和 ResourceQuota——租户隔离的边界从"Cluster 整体"下沉到了"namespace"。

### Personal Org（每个 user 一个私有 namespace）

参考 GitHub / OpenAI 的做法，每个 user 创建时自动获得一个 **Personal Org**（`is_personal=true`，`name='Personal'`，`slug='user-{id}'`），用户在其中是 admin。这是 user 的个人工作区，资源默认归这里。

- **非 admin 用户**：不再自动加入 Default Org（变更前是的）。要让 ta 进入团队工作区，admin 显式加成员
- **admin 用户**：除了自己的 Personal Org 外，还自动加入 Default Org 并担任 admin
- **删除 user**：cascade 删除其 Personal Org（连带 Org 内的 model_routes / api_keys / org-owned clusters 等）。共享出去的资源（cluster_access / model_route_principals 命中的）也随 Org 删除而失效——前提是 maintainer 已离开
- **Org 切换器**：Personal Org 永远在列表第一位

为什么不让所有人共享 Default Org：
- 默认共享会让每个新用户的资源对 admin / 其他成员立刻可见，违反"个人空间"心智
- 在严格租户场景里，admin 不一定希望自己看到所有 user 的私货
- Personal Org + 显式加入工作区 = "GitHub 体验"，对开发者友好

`organizations.is_personal=true` 的 Org 在 admin 的 Org 管理列表中默认隐藏（避免噪音），仅通过用户管理路径反查。

### 跨 Org 用户的请求语义

每个 API 请求按以下优先级解析 `current_organization_id`：

1. **API Key 调用**：从 `access_key` 反查 `api_keys.organization_id`（**忽略请求头**，避免误用）
2. **UI 会话 + 请求头**：携带 `X-Organization-Id` 时使用该值
3. **普通用户无请求头**：回退到 `users.default_organization_id`
4. **平台 Admin 无请求头**：解析为 `None`（act-across-all 模式，列表过滤跳过）；如携带请求头则以该 Org 上下文 act-as

UI 表现：
- 普通用户右上角切换器只能在加入的 Org 中切，永远有具体 org 上下文
- 平台 Admin 切换器多一档"All"，对应"无 org 上下文"——用于跨 Org 全平台视角

### 三层身份并存

```
User                     — 全局身份；登录、SSH 公钥、API key 主体
OrganizationMembership   — 用户在某 Org 内的角色（admin / user）
TenantContext            — 单次请求解析后的执行上下文
```

**Org 角色简化为两档（v1）**：`admin` 管理本 Org 的成员和基础设施，`user` 仅作消费者。原计划的三档（`owner` / `manager` / `member`）在 v1 收敛——GPUStack 自托管，没有计费或法律意义上的 Owner，`admin` 已涵盖原 `owner` + `manager` 的全部能力，只是**不能踢掉/降级最后一个 admin**（路由层 last-admin guard 校验）。未来若需要拆分"创建者 / 管理者"再加一档不晚。

平台 `is_admin` 与 Org role **完全解耦**：是不是平台超管，与是不是某 Org 的 admin 互不蕴含。代码里始终用 `is_platform_admin` vs `org_role == OrgRole.ADMIN` 区分，不写裸 `is_admin`。

### 统一过滤

通用查询过滤封装在 FastAPI 依赖中：

```python
stmt = stmt.where(Resource.organization_id == ctx.current_org_id)
stmt = stmt.where(Resource.cluster_id.in_(ctx.accessible_cluster_ids))
stmt = stmt.where(
    tuple_(Resource.owner_type, Resource.owner_id).in_(ctx.accessible_principals)
)
```

`organization_id` 一层放在最外，是所有租户隔离的兜底——任何遗漏的过滤最差也只会跨 owner，不会跨 Org。

**两类身份完全旁路过滤**（统一通过一个 `_bypass_tenant_filter(ctx)` 谓词判断）：

- **平台 admin 且无 Org 上下文**：跨 Org 全平台视图
- **系统用户 (`user.is_system=True`)**：worker / cluster service account 等服务端自己派生的内部账号——它们要跨 Org 读取资源（worker 取 instance 对应的 Model 详情、cluster 心跳等都不带 Org 上下文），不能被租户过滤拦下。

平台 admin 带 Org 上下文（act-as 模式）会经过 `organization_id` 过滤但跳过 `owner` 过滤——admin 在 Org 内可见全部资源不论 owner 是 group / user。

**写操作的 act-as 语义**：admin 在 Org 上下文里**不再被允许直接写 Global 行**——比如 admin 切到 Default 时点编辑 Platform vLLM，请求会被服务端的 redirect helper 转成"upsert Default Org 的 vLLM 行"，跟非 admin Org admin 一样的行为。要直接改 Platform 行需要切回"All"模式。这条规则用于 cluster / cloud_credential / worker_pool / inference_backend 等所有带 Hybrid 语义的资源。

### K8s GPU 实例多租户

- 每个 (Org × K8s Cluster) 自动 provision 一个 namespace：`gpustack-{org-slug}`（例如内置 Org 是 `gpustack-default`，自建 Org `acme` 是 `gpustack-acme`）
- 用户创建 GPU 实例 → backend 反向代理 → 落入该 namespace
- `ResourceQuota` apply 到 namespace，K8s 调度时自动拦截超限请求
- backend 不维护 GPU 实例状态表，**CRD 是 source of truth**

## 用户交互

### 平台 Admin

**变更前**：admin 看到所有 cluster、所有 model、所有 worker；为非 admin 用户上架模型并通过 ModelRoute 显式授权。

**变更后**，新增工作流：

1. **组织管理**：在"组织管理"页面创建/编辑 Org，关联计费账户
2. **用户管理**：现有用户列表保留；为每个用户管理其加入的 Org 列表，并为每个 (user, org) 设置 Org 内角色
3. **集群授权**：Cluster 详情页新增"访问授权"标签，把 cluster 授权给 Org / Group / User
4. **配额设置**：Cluster 详情页或 Org 详情页设置 (Org × Cluster) 配额
5. **Group 管理**：可由 admin 代管，也可由 Org admin 在其 Org 内自治

保留现有：模型上架/管理、Worker 管理、推理监控等。Admin 上架的模型默认归属内置 **platform Org**，通过 ModelRoute 发布给其他 Org。

### Org admin

普通用户的子集，由平台 admin 在添加成员时指定角色。新增能力：

- 管理本 Org 内成员（增删改角色，含其他 admin —— 但**不能踢/降级最后一个 admin**）
- 管理本 Org 的 UserGroup（组织内的子分组）
- 管理本 Org 内成员的 group 归属
- 查看本 Org 在各 cluster 的配额使用情况
- 管理本 Org 内的资源（v1 中所有资源都是 owner_type='org'，全 Org 可见）
- 管理本 Org 自家的 cluster / cloud_credential / worker_pool / inference_backend 等基础设施

### 普通用户

**变更前**：登录后看到 `/my-models`，使用自己的 API key 调推理。

**变更后**：

1. 登录后右上角增加 **Org 切换器**，列出加入的所有 Org，默认选中 `default_organization_id`
2. 切换 Org 后整个 UI 重新加载该 Org 的资源（列表、配额、API key 等）
3. **GPU 实例**：在被授权的 cluster 上创建 GPU 实例，上传 SSH 公钥，运行后 SSH 接入；UI 不让用户选择 owner，自动归 Org（同 RunPod team / Lambda team 默认共享行为）
4. **API Key**：归属 Org 隐式由当前切换器值决定（`X-Organization-Id`）；调用时该 key 仅能访问该 Org 的资源。平台 admin 在 "All" 模式下创建 key 时弹一个目标 Org 选择器以指定 key 应绑哪个 Org
5. **推理调用**：行为不变。`/my-models` 返回的 ModelRoute 由 `non_admin_user_models` 视图决定，按 `access_policy` 过滤：
   - `PUBLIC` / `AUTHED` → 所有非 admin 用户都看见（admin 上架的公共服务默认就是这个语义）
   - `ALLOWED_USERS` → 仅 `usermodelroutelink` 中显式列出的用户（旧路径，兼容期保留）
   - `ALLOWED_PRINCIPALS` → 视图内 join `model_route_principals` 按 user / org / group 任一匹配；这是跨 Org 精细化发布的入口

#### 资源可见性 — Org 内默认共享，UI 不暴露 owner 选择器

一期所有租户资源（Model / ModelRoute / ModelInstance / **GPU Instance** / ApiKey 之外）创建时自动设 `owner_type='org', owner_id=current_org_id`，**UI 不提供 visibility / owner 选择器**——同 Org 内成员默认互相可见，对照 RunPod team / Lambda team 的行为。

理由：
- 多数客户场景（小到中等团队）想要的就是"团队工作区式"共享；强制让用户每次创建都选 visibility 是 over-engineering
- Group 级 / User 级私有归属在 schema 上保留（`owner_type` 字段是 `org/group/user` 三档），但**v1 流程不放出 UI**，避免画蛇添足
- 哪天真有"组内私货"诉求再加 advanced 折叠区的选择器，schema 已就位

`owner_type='group'` / `owner_type='user'` 在 v1 仅用于：
- admin 在 Cluster Access 给 group / user 授权（粗粒度准入门）
- admin 在 ModelRoute `ALLOWED_PRINCIPALS` 发布给 group / user
普通用户不会感知到这两个值的存在。

### CLI / API 调用变更

| 场景 | 变更前 | 变更后 |
|---|---|---|
| `Authorization: Bearer <api_key>` | 等价于 user 身份 | 等价于 (user, key.org_id) 上下文 |
| 平台 admin 调用列表 API | 看到全平台 | 默认看到全平台；带 `X-Organization-Id` 时按该 Org 过滤 |
| 普通用户列表 API | 看到自己的 | 必须带 `X-Organization-Id` 或使用绑定 Org 的 API key；返回当前 Org 内可见资源 |

## 已知问题和限制

| 编号 | 限制 |
|---|---|
| L1 | UserGroup 级 ResourceQuota 一期不实现；同 Org 内多个 Group 共享 Org 级配额 |
| L2 | 模型推理调用频率/并发不在 Quota 体系内，依赖现有计费/限流 |
| L3 | 跨 Org 共享必须通过 ModelRoute 显式发布，不存在"全 Org 默认共享"的资源 |
| L4 | `owner_type/owner_id` 是多态字段，无 DB 级 FK 约束，需依赖应用层校验 |
| L5 | 平台 admin 的 `is_admin=true` 是全局旁路，没有更细粒度的"平台运维 vs 平台超管"分级 |
| L6 | API Key 与 Org 是绑定关系，跨 Org 必须使用不同的 API key（不支持运行时切 Org） |
| L7 | 一个 (Org × Cluster) 当前对应一个 namespace；如需进一步隔离 Group（独立 NetworkPolicy 等），需扩展 `tenant_namespaces` 增加 group 维度 |
| L8 | 平台 admin 默认是 platform Org admin，权限来源冗余，需文档说明 |
| L9 | BYO cluster 一期采用统一计量（不区分 cluster 归属对资源用量都计入 Org），不区别"自带硬件"和"平台共享" |
| L10 | BYO cluster 的 `set-default` 仍仅限平台 admin（Org 自管 cluster 不影响"平台默认"概念） |
| L11 | Grafana 监控入口（`/v1/config` 与 `/v1/grafana/*` 反代）在 v1 中保留 admin-only，普通用户看不到任何 Grafana 链接。原因：per-resource dashboard 的"过滤"仅依赖 URL 上的 template variables（`var-model_name` 等），用户进了 Grafana 可改变量看其它 Org 数据，缺乏租户级别的查询时隔离。等 P 阶段在 metrics 上加 `org_id` 标签 + 反代时 PromQL 注入再放开 |

---

## 实现细节

### 架构

#### 模块分层

```
┌──────────────────────────────────────────────┐
│ Routes Layer                                 │
│  - 平台级   (require_platform_admin)         │
│  - Org 级   (require_org_role)               │
│  - 用户级   (TenantContext + 过滤)           │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ TenantContext (FastAPI Dependency)           │
│  解析: current_user, current_org_id,         │
│        accessible_cluster_ids,               │
│        accessible_principals                 │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Services Layer (现有)                         │
│  list/get/update/delete 调用统一过滤函数      │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Storage Layer                                 │
│  + organizations / organization_memberships   │
│  + user_groups / user_group_memberships       │
│  + cluster_access                             │
│  + tenant_namespaces / tenant_quotas          │
│  ~ models / api_keys 加 organization_id       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Reconcilers (后台进程)                         │
│  - NamespaceReconciler                        │
│  - QuotaReconciler                            │
└──────────────────────────────────────────────┘
```

#### 新增组件

**TenantContext 依赖**
统一替代大多数业务路由上的 `CurrentUserDep`。`CurrentUserDep` 仅保留给 `/me/*` 这类纯用户身份的端点。

**NamespaceReconciler**（后台进程）
- 输入：`tenant_namespaces` 表的状态
- 输出：在目标 K8s cluster 创建/删除 namespace、ServiceAccount、RoleBinding
- 触发：行变更事件 + 定期全量校准

**QuotaReconciler**（后台进程）
- 输入：`tenant_quotas` 表 + `tenant_namespaces` 表
- 输出：每个 (org, cluster) 对应 namespace 上的 `ResourceQuota` 对象
- 触发：行变更事件 + 定期全量校准

部署形态：
- 一期与 server 同进程跑，作为 lifespan 启动的 task
- 后期可拆为独立 deployment + leader election

#### 数据模型

**新增表**：

```sql
CREATE TABLE organizations (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    description TEXT,
    billing_account_ref TEXT,
    is_platform BOOLEAN NOT NULL DEFAULT FALSE,
    is_personal BOOLEAN NOT NULL DEFAULT FALSE,  -- 每个 user 私有的 Personal Org
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE organization_memberships (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, organization_id)
);

CREATE TABLE user_groups (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    UNIQUE (organization_id, name)
);

CREATE TABLE user_group_memberships (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id INTEGER NOT NULL REFERENCES user_groups(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, group_id)
);

CREATE TABLE cluster_access (
    cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    principal_type TEXT NOT NULL CHECK (principal_type IN ('org','group','user')),
    principal_id INTEGER NOT NULL,
    granted_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP,
    PRIMARY KEY (cluster_id, principal_type, principal_id)
);

CREATE TABLE tenant_namespaces (
    id INTEGER PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    namespace_name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',  -- pending|provisioning|ready|error|deleting
    state_message TEXT,
    UNIQUE (cluster_id, organization_id),
    UNIQUE (cluster_id, namespace_name)
);

CREATE TABLE tenant_quotas (
    id INTEGER PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    gpu INTEGER,                  -- requests.nvidia.com/gpu
    cpu_milli INTEGER,            -- requests.cpu (millicores)
    memory_bytes BIGINT,          -- requests.memory
    pod_count INTEGER,            -- pods
    UNIQUE (cluster_id, organization_id)
);
```

**已有表的字段扩展**：

```sql
ALTER TABLE users
    ADD COLUMN default_organization_id INTEGER REFERENCES organizations(id);

ALTER TABLE api_keys
    ADD COLUMN organization_id INTEGER NOT NULL REFERENCES organizations(id);
ALTER TABLE api_keys DROP CONSTRAINT uix_user_id_name;
ALTER TABLE api_keys
    ADD CONSTRAINT uix_user_org_name UNIQUE (user_id, organization_id, name);

ALTER TABLE models
    ADD COLUMN organization_id INTEGER NOT NULL REFERENCES organizations(id);
ALTER TABLE models ADD COLUMN owner_type TEXT NOT NULL DEFAULT 'org';
ALTER TABLE models ADD COLUMN owner_id INTEGER NOT NULL;

ALTER TABLE model_instances
    ADD COLUMN organization_id INTEGER NOT NULL REFERENCES organizations(id);
-- model_instances 的 owner 跟随其 model
```

**ModelRoute 扩展**：

```sql
ALTER TABLE model_routes
    ADD COLUMN organization_id INTEGER NOT NULL REFERENCES organizations(id);

CREATE TABLE model_route_principals (
    route_id INTEGER NOT NULL REFERENCES model_routes(id) ON DELETE CASCADE,
    principal_type TEXT NOT NULL CHECK (principal_type IN ('org','group','user')),
    principal_id INTEGER NOT NULL,
    PRIMARY KEY (route_id, principal_type, principal_id)
);

-- access_policy 增加枚举值 ALLOWED_PRINCIPALS（保留 ALLOWED_USERS 兼容）
```

### API 改动

#### 路由分层

```python
# 平台超管 only
platform_admin_router = APIRouter(dependencies=[Depends(require_platform_admin)])
platform_admin_router.include_router(organizations.router, prefix="/organizations")
platform_admin_router.include_router(cluster_access.router, prefix="/clusters/{id}/access")
# Worker / Cluster CRUD / 模型上架 仍在平台级

# Org 级（require_org_role 校验当前 user 在 current_org 中是 admin）
org_router = APIRouter(dependencies=[Depends(get_tenant_context), Depends(require_org_admin)])
org_router.include_router(user_groups.router, prefix="/organizations/{org_id}/groups")
org_router.include_router(memberships.router, prefix="/organizations/{org_id}/members")
org_router.include_router(quotas.router, prefix="/organizations/{org_id}/quotas")

# 用户级（任何已登录、已选定 Org 的用户）
tenant_router = APIRouter(dependencies=[Depends(get_tenant_context)])
tenant_router.include_router(api_keys.router, ...)         # 隐含按 current_org 过滤
tenant_router.include_router(my_models.router, ...)         # ModelRoute 按 principal 解析
tenant_router.include_router(gpu_instances.router, ...)     # 反向代理到 K8s

# 自身身份（不需要 Org 上下文）
me_router = APIRouter(dependencies=[Depends(get_current_user)])
me_router.include_router(me.router, ...)                    # /me/organizations 等
```

#### 新增端点

| 路径 | 方法 | 权限 | 说明 |
|---|---|---|---|
| `/v1/organizations` | GET, POST | 平台 admin | 列表 / 创建 Org |
| `/v1/organizations/{id}` | GET, PUT, DELETE | 平台 admin | Org 详情 / 编辑 / 删除 |
| `/v1/organizations/{id}/members` | GET, POST | 平台 admin / Org admin | 成员管理 |
| `/v1/organizations/{id}/members/{user_id}` | PUT, DELETE | 平台 admin / Org admin | 角色变更 / 移除（last-admin guard） |
| `/v1/organizations/{id}/groups` | GET, POST | 平台 admin / Org admin | Group 管理 |
| `/v1/organizations/{id}/groups/{gid}/members` | GET, POST, DELETE | 平台 admin / Org admin | Group 成员管理 |
| `/v1/clusters/{id}/access` | GET, POST, DELETE | 平台 admin | 集群访问授权 |
| `/v1/organizations/{id}/quotas` | GET, PUT | 平台 admin | 配额管理（org × cluster） |
| `/v1/me/organizations` | GET | 任意登录用户 | 我加入的 Org |
| `/v1/me/organizations/{id}/clusters` | GET | 任意登录用户 | 当前 Org 我能访问的 cluster |

#### 现有端点的行为变更

| 端点 | 变更 |
|---|---|
| `POST /v1/api-keys` | 必须携带 `X-Organization-Id`（或使用已绑定 Org 的 API key 调用）；保存时记录 `organization_id` |
| `GET /v1/api-keys` | 只返回当前 Org 下的 key |
| `POST /v1/models`、`GET /v1/models` | 加 `organization_id` 过滤；admin 创建的默认归属 platform Org |
| `GET /v1/clusters` | 平台 admin 看全部；普通用户看 `accessible_cluster_ids`（来自 `cluster_access` 中授权给当前 Org / 用户所在 group / 用户本人的 cluster）。一期普通用户看不到 Cluster 管理菜单，过滤主要为 GPU 实例创建路径上的 cluster 选择器服务 |
| `GET /my-models` | 走 `non_admin_user_models` 视图，按 `access_policy` 过滤；详见上一节 |

#### 请求头与认证

```
Authorization: Bearer <token>          # 不变
X-Organization-Id: <org_id>            # 新增；UI 会话路径必须；API key 路径忽略
```

API Key 反查流程：`access_key` → `ApiKey` 行 → 取 `user_id` 与 `organization_id` → 组装 TenantContext。

### 其他

#### 数据迁移

迁移在 alembic upgrade 中完成：

1. 创建 `organizations` 表，插入 `id=1, name='Default', slug='default', is_platform=true` 作为内置 Org
2. 创建 `organization_memberships`：仅 admin 用户加入 Default Org（role=ADMIN）；非 admin 用户**不加入** Default Org
3. 为每个非 system 用户创建 Personal Org（`is_personal=true`，`slug='user-{id}'`），并在 `organization_memberships` 中加为 ADMIN
4. 创建 `user_groups`、`user_group_memberships`、`cluster_access`、`tenant_namespaces`、`tenant_quotas`（空表）
5. `users.default_organization_id` = 该 user 的 Personal Org id
6. `api_keys.organization_id` 全部回填 1（admin 旧 key 保留 Default Org 上下文；普通用户旧 key 同理）
7. `models / model_instances / model_routes.organization_id` 全部回填 1（admin 上架资产归 Default Org）
8. `models.owner_type='org', owner_id=1` 回填
9. 把现有所有 cluster 在 `cluster_access` 中授权给 Default Org（保留现状）
10. 现有 ModelRoute 的 `UserModelRouteLink` 复制一份到 `model_route_principals`（principal_type='user'）

迁移须在 sqlite 与 postgres 双 DB 测试通过；`ALTER TABLE` 兼容性由 alembic 的 `batch_alter_table` 处理。

#### 平台 Org 与平台 admin 的关系

- **平台 admin** = `users.is_admin=true`，全局旁路所有 Org 隔离（同 system user 一样进 `_bypass_tenant_filter`）
- **内置 Org** = `organizations.is_platform=true` 的特殊 Org，name="Default" / slug="default"，id 永远是 `PLATFORM_ORGANIZATION_ID = 1`；承载平台公共资产（admin 上架的模型等）
- 平台 admin 默认是内置 Org 的 ADMIN，同时也是自己 Personal Org 的 ADMIN，但三个身份职责不同：
  - **平台 admin**：管理跨 Org 资源（创建 Org、cluster_access、worker 等）
  - **内置 Org admin**：管理 Default Org 内部资产（公共模型、Group 等）
  - **Personal Org admin**：自己的私有工作区
- 实务上同一人持有这些角色，但权限来源不同，便于审计

**为什么内置 Org 起名 "Default" 而不是 "Platform"**：每个普通用户登录时这就是他们看到的 Org（除非 admin 把他们加到别的 Org），slug 也作为 K8s namespace 模板的一部分（`gpustack-default`）。"Default" 更准确反映"兜底租户"的角色，避免和"平台基础设施"概念混淆。

#### ModelRoute 兼容期

新增 `model_route_principals` 表后：
- 旧 `UserModelRouteLink` 表保留 30 天兼容期，读取时 union 两表
- `access_policy` 加 `ALLOWED_PRINCIPALS`，旧的 `ALLOWED_USERS` 视为只看 principal_type='user' 的子集
- 兼容期结束后由独立迁移移除旧表

#### 反向代理对接（仅说明接口契约）

GPU 实例操作走 backend 反向代理 K8s API（具体由另一份设计覆盖）。本设计提供以下契约：

- 给定 (user, current_org_id, cluster_id)，反向代理层应：
  1. 校验 user 在 current_org 中的成员资格
  2. 校验 (current_org / user / user 的 group) 任一在 `cluster_access` 中有该 cluster
  3. 解析出目标 namespace（`tenant_namespaces` 中 `(cluster_id, current_org_id)` 唯一一行）
  4. 解析或按需创建该 user 在该 namespace 的 ServiceAccount Token
  5. 在 K8s API URL 中注入 namespace 路径段后转发

#### Reconciler 同步语义

**NamespaceReconciler**：
- Desired = `tenant_namespaces` 中 state ∈ {pending, provisioning, ready}
- Actual = K8s cluster 中以 `gpustack.ai/managed=true` 标签筛选的 namespace
- 差异处理：缺失 → 创建 ns + 绑定 RBAC；多余（state=deleting）→ 删除 ns

**QuotaReconciler**：
- Desired = `tenant_quotas` 中已对应 ready namespace 的行
- Actual = 各 namespace 上以 `gpustack.ai/quota-managed=true` 标签筛选的 ResourceQuota
- 差异处理：apply / patch / delete

两个 reconciler 都采用 **事件驱动 + 周期全量** 双路径；周期间隔 5 分钟，事件触发延迟 1–3 秒。

### 研发计划

按 PR 维度分阶段，每个阶段对外可单独发布；admin 体验在 P0–P2 期间不变。

#### P0：Schema 基础（1 个 PR）

- alembic 迁移：新增 7 张表 + 现有表字段
- 数据回填到 platform Org（id=1）
- SQLModel 定义新表
- **不改路由、不改业务逻辑**
- 测试：sqlite + postgres 双 DB 迁移通过 + 回滚通过

工作量：约 3 天。

#### P1：Auth 改造（1 个 PR）

- `TenantContext` 依赖实现
- `require_platform_admin` / `require_org_role` 实现
- API Key 反查携带 `organization_id`
- 现有所有 list/get 接口套上 TenantContext 过滤（不改路由结构、不改 URL）
- **现有所有 API 行为对 admin 完全不变**（platform Org 是默认上下文）
- 测试：admin 可见性、平台 Org 普通用户可见性、跨 Org 看不到对方资源

工作量：约 5 天。

#### P2：Org / Group / Membership API（1–2 个 PR）

- `/v1/organizations` 全套 CRUD
- `/v1/organizations/{id}/members`、`/groups`、`/groups/{id}/members`
- `/v1/clusters/{id}/access` 授权管理
- `/v1/me/organizations`、`/v1/me/organizations/{id}/clusters`
- 文档与 admin UI（UI 可单独 PR）

工作量：约 7 天。

#### P3：K8s 多租户基础设施（2 个 PR）

- PR a：`tenant_namespaces` reconciler + `cluster_access` 触发的自动 namespace provision
- PR b：`tenant_quotas` reconciler + admin 配额设置 API

工作量：约 7 天。

#### P4：ModelRoute 扩展（1 个 PR）

- `model_route_principals` 表上线
- `access_policy` 增加 `ALLOWED_PRINCIPALS`
- `/my-models` 接口按 principal 解析当前 Org 上下文
- UI"模型发布"对话框支持选 org/group/user

工作量：约 3 天。

#### P5：UI 改造（独立 track，与 P2–P4 并行）

- Org 切换器
- Org / Group / Member 管理页
- 配额展示与设置页
- API Key 创建时选 Org
- 集群授权页

工作量：前端单独估，约 10 天。

#### 阶段依赖关系

```
P0 ──┬─→ P1 ──┬─→ P2 ──→ P3
     │        │
     │        └─→ P4
     │
     └─→ P5（前端，依赖 P2/P3/P4 的 API 出来）
```

P0 必须先完成；P1 是所有后续工作的前置；P2 / P3 / P4 可并行；P5 跟随 API 节奏。

#### 风险与缓解

| 风险 | 缓解 |
|---|---|
| 迁移过程中现有用户体验中断 | P0–P1 设计为对 admin 行为零变更；普通用户原本仅有 `/my-models`，迁移后等价 |
| `owner_type/owner_id` 多态字段无 FK 校验，可能产生脏数据 | 应用层校验 + 周期一致性扫描；删除 user/group 时联动把资源 owner 改回 org 兜底 |
| Reconciler 与 K8s 状态不一致 | 事件驱动 + 周期全量校准两条路径并存；ResourceQuota 用标签做 ownership 标识 |
| 跨 Org 用户在 UI 切换 Org 时缓存污染 | 切换 Org 触发前端 store 全量重置 + 路由刷新 |
| API Key 与 Org 强绑定带来的迁移问题 | 旧 API key 默认绑定 platform Org，使用语义不变；用户跨 Org 时需新建 key |
| 平台 admin 旁路过宽 | 关键写操作打审计日志；后续可在 `is_admin` 之上叠加更细粒度权限 |

## Bring-Your-Own Cluster（v1 已落地）

`Cluster` / `CloudCredential` / `WorkerPool` 一律 NOT NULL `organization_id`：

```sql
ALTER TABLE clusters
    ADD COLUMN organization_id INTEGER NOT NULL
    REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE cloud_credentials
    ADD COLUMN organization_id INTEGER NOT NULL
    REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE worker_pools
    ADD COLUMN organization_id INTEGER NOT NULL
    REFERENCES organizations(id) ON DELETE CASCADE;

-- 每个 cluster / cloud_credential / worker_pool 都隶属于一个 Org。
-- admin 在 "All" 模式下创建时由路由层默认落到平台 (Default) Org；
-- 跨 Org 共享走 cluster_access，不再用 NULL `organization_id` 表示。

-- 每个 Org 至多一个 default cluster：
CREATE UNIQUE INDEX uix_clusters_default_per_org
    ON clusters (organization_id)
    WHERE is_default = TRUE AND deleted_at IS NULL;
```

WorkerPool 的 `organization_id` 在创建时由其 cluster 同步过来（denormalized），列表过滤时不用 join。

### 权限语义

| 主体 | 自家 Org 的 cluster | 别人家 Org 的 cluster |
|---|---|---|
| 平台 admin | CRUD（act-as 也可以；"All" 模式下能直接管理任意 Org 的 cluster） | CRUD |
| Org admin（cluster owner Org） | CRUD（含 set-default、worker_pool、cluster_access） | 仅当有 `cluster_access` 行时可见，且只能"用"不能"管" |
| Org user / cluster_access 受让人 | 同 Org 隐式可见可用；通过 cluster_access 可跨 Org 借用 | 借来的 cluster 只授 USER 权限（部署模型 / 起 GPU 实例），不能管基础设施 |
| System user (worker / cluster account) | 全部可见可读（bypass） | 同上 |

写权限永远属于 cluster owner Org 的 admin（外加平台 admin）；`cluster_access` 表达的是"借用权"，不会越权升格成管理权。

### 创建路径校验

- 平台 admin：建任意 Org 的 cluster；"All" 模式下省略 `organization_id` 时默认落到平台 (Default) Org
- Org admin：只能建 `organization_id = 自己当前 Org`
- 其他 Org user / 普通用户：不能创建任何 cluster

### Default cluster（per-Org）

`clusters.is_default` 不再是平台单独一档，而是每个 Org 各有一个，由部分唯一索引（`UNIQUE(organization_id) WHERE is_default = TRUE AND deleted_at IS NULL`）保证。语义：

- 每个 Org 在该 Org 上下文部署模型时的兜底 cluster；写权限沿用 `assert_cluster_writable`
- **新建 cluster** 时若该 Org 还没有任何 cluster，路由层把这条自动标成 default（避免一个额外的"set-default"圆球）
- **部署表单兜底链**：`当前 Org 的 default → 平台 (Default) Org 的 default → 任意 Ready cluster → list[0]`。admin "All" 视角没有 Org 上下文，跳过第一档直接看平台 Org 的 default

### `cluster_access` 行为

- 拥有方 Org 的成员**隐式**具备访问权；要把 cluster 借给其他 Org / Group / User，写 `cluster_access` 行
- `cluster_access` 命中只授**USER 权限**（部署 / GPU 实例消费），管理权仍属于 owner Org 的 admin

### Quota / Namespace

`(Org × Cluster) → namespace + ResourceQuota` 模型不变。Org 在自家 cluster 上仍然落入 `gpustack-{org-slug}` namespace（这样 Group 级别再隔离仍有空间）。Org 自管的 cluster 上对 owner Org 默认 quota 通常为 unlimited，admin 看心情干预。

### 计费

v1 采用**统一计量**：所有 namespace 的 GPU/CPU/内存用量都计入 Org，**不区分 cluster 归属**。Org 自带硬件也走计量但通常计费规则可以打折/豁免，由商业侧另行决定。这个简化让计费代码只一条路径，避免分支。

如果将来要做"Org 自管 cluster 零边际成本"，再加 cluster-level "billable" 标签即可。

### 路由分层调整

`/v1/clusters`、`/v1/cloud-credentials`、`/v1/worker-pools` 从 `v1_admin_router` 移到 `v1_base_router`——任何登录用户都能进 endpoint，但 endpoint 内部用 `assert_cluster_writable` / `assert_org_owned_writable` / `validate_org_owned_owner` 做按行的所有权校验。`set-default` 和 worker token 这种平台级动作仍然在路由内部 raise `is_admin` 限制。

### UI 变化

- Cluster Management 菜单的 access flag 从 `canSeeAdmin` 改成 `canManageInfra`（admin 或在任一 Org 持有 admin 角色的用户都看得到）
- Cluster / CloudCredential 创建表单：**只有平台 admin 在 "All" 模式才显示 "Organization" 下拉**（默认 Default Org）；其他场景（Org admin / admin act-as）当前 Org 是隐式 owner，不展示选择器。这条规则两边表单一致
- 列表（cluster / credential / worker_pool）过滤已经在 server 端做了 visibility，前端不用变

## Inference Backend Hybrid (v1 已落地)

InferenceBackend 跟 cluster 类似走"Global + Org override"的 Hybrid 模型，但区别在于一个 backend (e.g. vLLM) 可以**同时存在 Platform 行（admin 维护的内置 / community catalog）和某 Org 自己的扩展行**——Org 行在内部表里跟 Platform 行共享 `backend_name`，区分键是 `(backend_name, organization_id)` 复合 unique。

### 数据模型

```sql
ALTER TABLE inference_backends
    ADD COLUMN organization_id INTEGER NULL
    REFERENCES organizations(id) ON DELETE CASCADE;
-- 复合唯一：同一 backend_name 在 Global (NULL) 和每个 Org 各最多一行
ALTER TABLE inference_backends
    ADD CONSTRAINT uix_inference_backends_name_org
    UNIQUE (backend_name, organization_id);
```

### 视角合并规则

非 admin / Org user 看到的列表是**单卡 collapse 视图**——同一 `backend_name` 的 Platform 行 + 自家 Org 行被合并成一张卡片：
- metadata（`is_built_in` / `backend_source` / `description` / 等）：Org 行优先（Org 行 inherit Platform 的 `is_built_in=True`，所以"Org 扩展的内置 backend 仍然是 built-in"）
- `version_configs`：union，Org 同 key 覆盖 Platform 同 key
- `enabled`：`Platform.enabled OR Org.enabled`——避免 Org 行误存的 `enabled=False` 把 Platform-enabled 的 backend 隐藏（取舍：Org 没法显式 disable Platform 共享的 community backend）

admin 看到的是**uncollapsed 视图**：Platform 行和各 Org 行作为独立卡片显示，便于直接管理。Owner tag 显示 "Global" 或 Org 名以区分。

### 写操作的 redirect

**任何调用方在 Org 上下文里写 Global 行**，服务端 `_redirect_global_edit_to_org_row` 会把请求转向 upsert 该 Org 的对应行：
- 找不到 Org 行 → 用 payload 创建一行（继承 Platform 的 `is_built_in` / `backend_source`）
- 找到了 → update 那一行
- Platform 行不动；admin 想改 Platform 必须切到 "All" 上下文

非 admin 在 Org 上下文也走同样路径——这就是"Org admin 在 vLLM 卡上加版本，背后透明地落到 Org 自家行"的实现。

### 部署候选（`/v2/inference-backends/list`）

部署表单选 backend 时调用 `/list`，按 `backend_name` 分组合并返回；Org 行的 `version_configs` overlay 在 Platform 之上，所以 Org 加的自定义版本和 Platform 的内置版本一起出现在选择列表里。

### 删除规则

- Platform built-in 行：永远不可删（admin 维护的 catalog）
- Platform community 行：UI 上点删走 soft-disable（PUT `enabled=false`），不真删行
- Platform Custom 行：admin 在 "All" 模式可真删
- 任何 Org 行（不论 source）：自家 Org admin 或 平台 admin 可真删——等同于"重置该 backend 在该 Org 的所有自定义"

## Usage 视角（v1 已落地）

Usage 页面为 Org admin / 平台 admin 提供两种视角，由顶部 `Org usage` / `My usage` segmented 切换：

| 视角 | 后端过滤 | 含义 |
|---|---|---|
| **Org usage** | `organization_id == current_org_id` | provider 视角：本 Org 的模型/infra 被消耗了多少 |
| **My usage** | `user_id == self`（**drop org filter**）| 个人视角：我自己累计调用了多少（跨 org，覆盖 AUTHED 共享场景）|

| 角色 / Context | toggle | 默认 scope |
|---|---|---|
| admin "All" | 隐藏 | `org`（无 org_id 即无 filter，等价"全平台"）|
| admin act-as / Org admin | 显示 | `org` |
| Org user / Personal Org 用户 | 隐藏 | `mine` |

`mine` 视角刻意 drop org filter 以解决 AUTHED 模型的跨 Org 共享场景：Personal Org 用户调 Default Org 的 AUTHED 模型，usage 行 `organization_id=Default`、`user_id=user`，他们在 Personal 上下文需要能看到自己的消耗记录。

## ModelRoute 名称的 Org 命名空间（v1 已落地）

ModelRoute 名字在 Org 内唯一（`(name, organization_id)` 维度），跨 Org 可以重名。但客户端调用 `/v1/chat/completions` 时通过请求体 `model` 字段路由，如果两个 Org 都有名为 `qwen3-0.6b` 的 route，Higress AI proxy 会因为 model 名碰撞而无法分发——所以实际暴露给客户端的 **effective model name** 带上 Org slug 前缀：

```text
平台 Org (is_platform=true) :  qwen3-0.6b           （保持向后兼容）
其他 Org (slug=org1)        :  org1/qwen3-0.6b      （OpenAI / HF 风格的 namespace）
```

实现：

- `gpustack/schemas/model_routes.py` 提供 `effective_route_name(name, slug, is_platform)` 工具
- `sync_gateway` 加载 route owner Org，把 effective name 透传给：
  - `sync_model_route_mapper` — AI proxy `modelMapping` 的 key
  - `ensure_model_ingress` — Higress ingress 的 `x-higress-llm-model` 头匹配
  - `ensure_route_generic_transformer_config` — `/model/proxy/<id>/...` 路径反写出来的 header value
- 客户端展示侧（`ApiAccessInfo` modal、ModelRoute 列表）走前端的 `effectiveRouteName(name, ownerOrg)` helper 计算同一个值

`/model/proxy/<route_id>/...` 路径仍按 id 路由，不受名字影响——admin 想直接拼 URL 调试或绕过 model 名解析时仍可用。
