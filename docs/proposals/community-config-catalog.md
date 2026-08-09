# 设计文档：吸收社区部署配置进入 Model Catalog

状态：Draft · 范围：本版本（PD 分离明确不在本设计内，见 §9）

## 1. 背景与目标

vLLM Recipes 与 SGLang cookbook 沉淀了大量「某模型 × 某硬件 × 某目标」的推荐部署配置（启动参数、环境变量、并行拓扑、量化、验证过的硬件）。这些是社区实测出来的知识，但 GPUStack 的 `model-catalog.yaml` 目前靠手工维护，既跟不上上游更新，也没有把这些配置的「选择维度」暴露给用户。

目标：

1. **机制**：把两个社区源确定性地转换为 GPUStack catalog 的 `ModelSpec`，可持续同步、可审计。
2. **UX**：让用户在部署时能像在 vLLM/SGLang 官网那样，按「硬件 / 量化 / 优化目标 / 拓扑」选择推荐配置，并看到配置来源与验证状态。
3. **闭环**：与已落地的 benchmark/auto-tune 打通——社区配置作为「起点」，实测把 `recommended_rate` 写回。

非目标（本版本）：PD 分离（prefill/decode disaggregation）。见 §9。

## 2. 两个社区源的实际形态

两个源都已是**结构化、可确定性解析**的，无需 LLM 抽取（LLM 仅作 vLLM 旧版 markdown 的兜底）。

### 2.1 vLLM Recipes — `models/<hf_org>/<hf_repo>.yaml`

结构化 YAML。最终启动命令由多层**组合**而成：

```
base_args + features.<f>.args + variants.<v>.extra_args + hardware_overrides.<hw>.extra_args
base_env  + variants.<v>.extra_env + hardware_overrides.<hw>.extra_env
```

关键字段：`model.model_id`、`model.docker_image`（按 brand：nvidia/amd）、`model.min_vllm_version`、`variants.<v>`（`precision` / `model_id` / `vram_minimum_gb` / `supported_hardware`）、`meta.hardware`（硬件 → `verified`）、`compatible_strategies`（**并行拓扑**：`single_node_tp` / `single_node_tep` / `multi_node_tp_pp` / `multi_node_dep` / `pd_cluster`…）。

### 2.2 SGLang cookbook — `docs/src/snippets/configs/<hf_org>/<model>.jsx`

`export const config = { ... }` 的**纯静态对象字面量**（无 spread/函数调用），核心是 `cells` 数组。每个 cell：

```js
{
  match: { hw, variant, quant, strategy, nodes },   // 选择键
  verified: true,                                     // 是否实测验证
  env: [...],                                          // 环境变量
  flags: ["--model-path {{MODEL_NAME}}", "--tp 8", ..., "--port {{PORT}}"],
}
```

配套：`modelNames`（`variant|quant` → HF repo）、`dockerImages`（hw → image）、`defaultAccuracy`、`benchmarkCommands`。

### 2.3 关键区别：两个源的 "strategy" 是不同的轴

| 源 | 该源的选择轴 | 语义 |
|---|---|---|
| vLLM | `compatible_strategies` | **并行拓扑**（TP / TP+PP / DEP / …） |
| vLLM | `variants[].precision` | 量化 |
| vLLM | `meta.hardware` | 硬件 |
| SGLang | `strategy`（low-latency/balanced/high-throughput） | **优化目标** |
| SGLang | `nodes`（single/multi-2） | 拓扑（节点数） |
| SGLang | `quant` / `variant` / `hw` | 量化 / 变体 / 硬件 |

**结论（修订）**：用户可选维度收敛为**三轴**——`后端`、`量化`、`Mode(预设)`；**拓扑（单机/多机）不作为用户选择，由 GPUStack 根据集群环境推断**（见 §5.2）。

- 「Mode」是统一的「推荐预设」轴：SGLang `strategy` 贡献目标型（latency/balanced/throughput，已对应 `ModelSpec.mode`）；vLLM `compatible_strategies` **拆分**为 `{节点数}`（→ 归入推断的拓扑）+ `{并行种类}`（tp/tep/dep/tp_pp，→ 归入 Mode）。
- 注意：Mode 语义因此混合了「目标」与「并行机制」，UI 标签建议用「预设 / Profile」。且当前 vLLM 结构化 recipe 的 `strategy_overrides` 多为空（各 strategy flags 不区分），在上游填充前 vLLM 实际只贡献单一 Mode。

## 3. GPUStack 侧现状（已核实）

- `Catalog` → `ModelSet` → `ModelSpec`，`ModelSpec` 复用部署用的 `ModelSpecBase`，因此一个 spec 即一份预填部署配置。（`gpustack/schemas/model_sets.py`, `gpustack/schemas/models.py`）
- 一个 set 天然可挂多个 spec，按 `gpu_filters`（vendor / compute_capability / vendor_variant）+ `mode` 区分。（`filter_specs_by_gpu` in `gpustack/routes/model_sets.py`）
- `backend_parameters: List[str]` 为原始 CLI flag；校验为**黑名单**：拒绝 `--port` / `--api-key` / `--served-model-name`。（`validate_model_in` in `gpustack/routes/models.py`）
- **并行度**：若用户未在 flags 中指定，调度器按 GPU 数**自动注入** `--tensor-parallel-size` / `--tp`；若指定了则作硬约束并与 `gpus_per_replica` 交叉校验，不一致报错（除非 `GPUSTACK_SKIP_GPU_COUNT_CHECK`）。多机 TP/PP/DP 均支持（Ray 与 MP/torchrun 两条路）。EP 仅昇腾 MindIE 拓扑感知，vLLM EP 为透传。（`gpustack/utils/vllm_topology.py`, `gpustack/policies/candidate_selectors/vllm_resource_fit_selector.py`, `gpustack/worker/backends/vllm.py`）
- catalog → 部署的粘合在 `set_default_spec()`（`gpustack/scheduler/evaluator.py`）：把 catalog spec 的 `backend_parameters` / `env` / `categories` 填入用户未显式设置的字段。
- **无**「命名 strategy / parallelism mode」的一等概念；拓扑只能靠 `gpus_per_replica` + `distributed_inference_across_workers` + 原始 flags 表达。

## 4. 数据映射（source → ModelSpec）

一个模型家族对应一个 `ModelSet`；`(backend × 硬件 × 量化 × 优化目标 × 拓扑)` 的每个有意义组合展开成一个 `ModelSpec`。

| 源字段 | ModelSpec 字段 | 说明 |
|---|---|---|
| vLLM `variants[].model_id` / SGLang `modelNames[variant\|quant]` | `huggingface_repo_id`（+ 派生 `model_scope_model_id`） | 不同量化常是不同 checkpoint |
| vLLM `meta.title` / SGLang `modelName` | `ModelSet.name` | 家族名 |
| vLLM `meta.tasks` | `categories` / `capabilities` | text/multimodal/omni/embedding |
| vLLM `meta.hardware` key / SGLang `match.hw` | `gpu_filters`（见 §6 硬件表） | h200→nvidia+cc9.0；mi355x→amd+gfx950 |
| vLLM `variants[].precision` / SGLang `match.quant` | `quantization` | fp8/bf16/nvfp4/mxfp4 |
| SGLang `match.strategy` / vLLM `compatible_strategies` 的并行种类 | `mode`（预设） | SGLang: low-latency→latency 等；vLLM: `single_node_tp`/`multi_node_tp`→ 同一 `tp` 预设（节点数被剥离去推断） |
| SGLang `match.nodes` + tp/pp 数字 / vLLM 节点数前缀 | `gpus_per_replica` + 节点数 + `distributed_inference_across_workers`（**存为变体属性，供 §5.2 推断，不作用户选项**） | 见 §5 |
| 组合后的 args（去掉数量类与托管类 flag） | `backend_parameters` | 见 §5 |
| 组合后的 env | `env` | |
| vLLM `model.docker_image[brand]` / SGLang `dockerImages[hw]` | `image_name` | |
| vLLM `model.min_vllm_version` | `backend_version` | |
| 源类型 | `backend` | vLLM → `vLLM`；SGLang → `SGLang` |
| vLLM `verified` map / SGLang `cell.verified` | provenance `verified_hardware`（见 §7） | |

**backend 共存**：同一模型的 vLLM 与 SGLang cell 都并入同一个 `ModelSet`，成为 `backend` 不同的多个 spec，UX 上可切换后端。

## 5. 并行度翻译规则（核心，也是最需要评审的部分）

社区把并行度写死为整机 GPU 数（`--tp 8`、`--tp 16`、`--dp 8`），而 GPUStack 自己调度 GPU。翻译策略：

1. **确定目标 GPU 拓扑**：由 `(hw, nodes)` 与 tp/pp 数字推出「每副本 GPU 数」与「节点数」。
   - SGLang：`nodes: single` + `--tp 8` → 单机 8 GPU；`nodes: multi-2` + `--tp 16` → 2 节点 × 8 GPU。
   - vLLM：由 `compatible_strategies` + `vram_minimum_gb` + 命令里的 tp 推。
   - 落到 `gpus_per_replica`（+ 多机时 `distributed_inference_across_workers=true`）。
2. **剥离数量类 flag**：`--tensor-parallel-size`/`--tp`/`--pipeline-parallel-size`/`--pp`/`--data-parallel-size`（纯 DP 副本）从 `backend_parameters` 删除，交给调度器自动注入。
3. **剥离托管类 flag / 占位符**：`--model-path`/`--served-model-name`/`--host {{HOST_IP}}`/`--port {{PORT}}`/`--api-key`（GPUStack 管理，且部分被 `validate_model_in` 拒绝）。
4. **保留非数量类 flag（透传）**：`--enable-expert-parallel`、`--speculative-*`、`--mem-fraction-static`、`--chunked-prefill-size`、`--kv-cache-dtype`、`--moe-a2a-backend`、`--quantization` 等。
5. **格式归一**：`--flag value` → `--flag=value`（用 `gpustack/utils/command.py`）。

### 5.1 待评审的边界：MoE 的 DP-Attention / EP

`--tp 8 --dp 8 --enable-dp-attention`（GLM-5.2 balanced）里 `--dp 8` **不是**数据并行副本，而是 attention-DP，真实 GPU 数仍是 8（`= tp`），而非 `tp*dp=64`。但 GPUStack 现有的 `get_world_size_from_backend_parameters` 计算 `world = tp*pp*pcp*dp`，直接透传会得到 64 与 `gpus_per_replica=8` 冲突而报错。

**候选方案（需与调度器 owner 定）：**
- (A) 生成器识别 `--enable-dp-attention`，把 `--dp` 视为「attention-DP」而非副本，`gpus_per_replica = tp`，`--dp`/`--enable-dp-attention` 作透传；并让 `world_size` 计算跳过带 dp-attention 语义的 `--dp`。
- (B) 生成器一律显式写 `gpus_per_replica`（由已知节点 shape 推），并对这类 cell 允许 world/count 不一致（类似 `GPUSTACK_SKIP_GPU_COUNT_CHECK` 的按-spec 白名单）。

倾向 (A)（语义正确、不放宽全局校验）。GLM-5.2 端到端示例（§8）会把这个 case 具体化。

### 5.2 拓扑推断（单机/多机不作为用户选择）

社区 flags 把 GPU 数烤死（单机 bf16 `tp8` vs 多机 bf16 `tp16` 是**不同的 flag 组**），所以「推断拓扑」不是「拿一组 flags 自由放置」，而是：

> catalog 为每个拓扑变体保留各自的 flags 与 GPU 需求（`gpus_per_replica` + `distributed`）；用户只选 `后端/量化/Mode`，**GPUStack 复用调度器 resource-fit（§10 dry-run 接口），自动挑出能在当前集群放下的那个变体**。

规则：
1. 候选 = 当前 `(后端, 量化, Mode)` 下的全部拓扑变体。
2. 可行性：非 distributed 变体需「某单节点 GPU 数 ≥ `gpus_per_replica`」；distributed 变体需「多节点合计 ≥ `gpus_per_replica` 且节点数>1」。
3. 平局策略：优先最少节点（单机 > 多机）→ 优先 `verified` → 最小 GPU 占用。
4. 无可行变体 → 不可部署，给出原因（"BF16 需 16 卡，当前无单节点满足"）。
5. UX：拓扑以**只读的解析结果**呈现（"将以 多机·2×8 部署"），非下拉选项。

这把 §10 的 dry-run 从「置灰不可选项」升级为「自动选型」，因此它成为 load-bearing，必须确定性且可解释。

### 5.3 蒸馏成 GPUStack 原生三档 Mode（最终形态）

不逐硬件保留 spec，而是**按经验把社区配置蒸馏成 GPUStack 已有的 Standard/Latency/Throughput 三档**。这既对齐现有 UX，又天然提高对国内硬件（A100/A800、H20、消费卡、昇腾）的适用性。

**Mode 来源分工：**
- **Standard** ← vLLM Recipes（维护最好、覆盖模型最多的基线；无则回退 SGLang `balanced`）
- **Latency / Throughput** ← SGLang cookbook（`low-latency` / `high-throughput`）

**蒸馏键 = `(backend, quantization, 厂商类, mode)`**：定义模式的**结构性 flag 在厂商类内一致**，故对该组内跨硬件的归一化 flag **求交集**得到一条 spec；随 GPU 变化的数值量级自然落选（由 auto-tune 在真实硬件上补回）。**不能跨厂商蒸馏**（AMD 无 EAGLE、用 tilelang）→ AMD 自成一组。

**`gpu_filters` 由「量化要求 + 每个 flag 的真实要求」算出**（`compute_gpu_filters` = `quant_base_filters` 合并 `FLAG_REQUIREMENT`；绝不用 verified 硬件），而不是按厂商类一刀切。量化基线：`fp8→nvidia cc>=8.9`（含 Ada/H20，修正 recipe 常见的 `>=9.0` 误杀）、`nvfp4→cc>=10.0`、`mxfp4→amd gfx950`、`bf16/…→无门槛`。逐 flag 要求：`deepep→Hopper+`、`EAGLE/MTP→nvidia`（CUDA-only）、`tilelang/aiter→amd`、`trtllm_mha→Blackwell`；未列出的 flag 视为可移植、不收窄门槛。

**效果：门槛「参数允许多广就多广」，GPU-unlimited 配置自动浮现**——例如 BF16 throughput（仅 `--max-running-requests`）→ `{}`（任意 GPU 含 A100/AMD）；BF16 latency（含 EAGLE）→ 仅 `{vendor: nvidia}`（不限 cc，A100 可跑）；FP8 throughput（含 deepep）→ `cc>=9.0`、FP8 latency（无 deepep）→ `cc>=8.9`。诚实边界：带厂商专属加速（EAGLE/deepep/tilelang）的配置无法做到跨厂商无限；要更广就把这些加速项从 mode 预设移入 feature tile（按硬件点亮），mode 预设保留可移植核心。拓扑由调度器推断（§5.2），不 pin GPU 数。

**模型内在 parser**（`--reasoning-parser`/`--tool-call-parser`）烤进每条 mode spec，不做 tile。

**feature tiles（能力开关）**：把 vLLM `features`（opt-in/非 parser）∪ SGLang `playgroundFeatures`（带 flags 的项）统一成可点选 tile：`{id,label,flags,env,default_on,backends,disable_by_hw}`。UI 点亮 → flags 合并进 `backend_parameters`；`disable_by_hw` 按硬件置灰（数据来自 cookbook）。**数值调优**（mem-fraction/chunked-prefill/max-running）不做 tile，蒸馏时丢弃、交 Mode 预设 + auto-tune。

原则：`gpu_filters` 只编码真实硬件要求；`verified` 只作软信号，绝不当硬过滤。诚实边界：蒸馏救中小模型 + 避免误杀；frontier 大模型（GLM-5.2）在小卡上物理跑不了，此时由 §10 dry-run 如实说明要求。

实测：**GLM-5.2**（双源）→ 15 mode-specs（standard/latency/throughput 各 5）+ 5 tiles（从 61 条塌缩）；**Qwen3.6-27B**（vLLM + 重建 SGLang）→ 9 mode-specs + 7 tiles，其中 BF16 Standard 无 `gpu_filters`（A100/消费卡可见）。

## 6. 硬件 → `gpu_filters` 映射表

| 源 hw | vendor | compute_capability | vendor_variant |
|---|---|---|---|
| h200 / h20 | nvidia | `>=9.0,<10.0` | — |
| b200 / b300 / gb300 | nvidia | `>=10.0` | — |
| mi300x | amd | — | gfx942 |
| mi325x | amd | — | gfx942 |
| mi355x | amd | — | gfx950 |
| ascend 910b | ascend | — | 910b |

（此表随上游硬件扩展维护；`compute_capability` 为 pip-style specifier，沿用现有 catalog 惯例。）

## 7. Provenance schema 变更

给 `ModelSpec`（`gpustack/schemas/model_sets.py`）新增**可选、向后兼容**字段，不影响现有 catalog：

```python
class SpecProvenance(BaseModel):
    source: Optional[str] = None          # "vllm-recipes" | "sglang-cookbook" | "manual"
    source_url: Optional[str] = None      # 上游 recipe/cookbook URL
    source_version: Optional[str] = None  # 上游 commit / date_updated
    verified_hardware: Optional[List[str]] = None  # 实测验证过的 hw 列表
    verified: Optional[bool] = None       # 该 spec 是否上游标注 verified

# ModelSpec 增加：
provenance: Optional[SpecProvenance] = None
```

用途：UI 打「来自 vLLM Recipes / SGLang cookbook」+「已验证」徽章；生成器据 `source_version` 做增量更新与冲突检测；审计。

## 8. 生成器与 CI

**位置**：`hack/gen-catalog/`（新增），产出 `gpustack/assets/model-catalog.yaml` 与 `-modelscope.yaml`。

**流程**：

```
拉取上游源 (vllm recipes YAML / sglang jsx)
  → source adapter 解析为统一中间表示 (IR = 展开后的 ModelSpec 候选)
      · vLLM adapter：组合 base/variant/hardware/features → 每个 (variant × hw) 一条
      · SGLang adapter：解析对象字面量 → 每个 cell 一条
  → normalize：并行度翻译(§5) + flag 剥离/格式归一 + hw→gpu_filters(§6) + mode 映射
  → merge：按模型家族聚合成 ModelSet，vLLM/SGLang 作不同 backend 的 spec
  → emit：HF 版 + ModelScope 版 (HF repo → MS id 映射)
  → validate：Catalog pydantic + denylist + spec 去重
```

**SGLang `.jsx` 解析**：因是纯静态对象字面量，用一个小 node 脚本 `require` 后 `JSON.stringify` 导出，或用 JS 对象字面量解析器；不引入 LLM。

**CI**：`hack/ci.sh` 增一步——重新生成并 `git diff` 校验（漂移则失败），或定期 job 开 PR。**产物始终经人工 review 合入**，生成器不自动直推。

**双 YAML 同步**：生成器同时产 HF/ModelScope 两份，消除手工同步。

## 9. UX 设计（Tier 0 / Tier 1）

### Tier 0（零引擎改动，随生成器立即得）
生成器为每种硬件选好推荐 spec；「优化目标」用现有 `mode` 承载。UX 几乎不变，纯内容变好。

### Tier 1（真正对齐社区官网的部署体验）
部署对话框把选择维度显式化为**三个联动选择器**：`后端` · `量化` · `Mode(预设)`，外加**来源/验证徽章**。**拓扑不作为选择器**，改为按 §5.2 推断并以**只读解析结果**呈现（"将以 单机·8 GPU / 多机·2×8 部署"）。

- Mode 选项来自该 `(后端,量化)` 的 specs（去重）。
- 拓扑由 §10 dry-run 自动选型；多个可行时按 §5.2 平局策略选默认，用户可在高级项覆盖。
- **能力 tile 网格**（§5.3 `feature_tiles`）：点方块开关一项能力，其 flags/env 合并进 `backend_parameters`/`env`；按当前硬件置灰不支持项并给出原因。
- 选中后自动回填 `backend_parameters` / `env`；高级用户可编辑底层原始参数。
- 交互从「选 GPU / 选拓扑」转向「选后端+量化+预设 + 点能力方块，平台自动放置」。

### PD 分离（本版本不做）
PD 是多角色部署（prefill 池 + decode 池 + KV 传输 + router），非「换组 flag」，GPUStack 引擎当前无此能力。本设计中：
- 生成器**跳过** `compatible_strategies: pd_cluster` 与 SGLang `pdDisagg` cell，或标 `unsupported/preview` 不产出可部署 spec；
- UX 不暴露 PD。
留作独立 roadmap。

## 10. 可行性门控（后端 dry-run）

Tier 1 的置灰需要「选择阶段」就知道某 spec 能否在当前集群跑，而现有 `filter_specs_by_gpu` 只比 vendor/cc/variant，不看 GPU 数/VRAM/拓扑。方案：

- 新增后端 **dry-run/preview 接口**，复用调度器候选选择器（`vllm_resource_fit_selector` 等）的可行性逻辑，对每个候选 spec 返回 `{feasible: bool, reason?, placement_hint?}`。
- UI 拉该接口渲染可选/置灰 + 资源需求提示，保证「能选 = 能部署」，避免前端粗算与真实调度不一致。

## 11. 与 benchmark/auto-tune 的闭环

社区 spec 是起点；`verified`/上游 benchmark 数只是参考硬件上的结果。部署后可一键触发 auto-tune，在用户真实硬件上复核并写回 `recommended_rate`/`sla_met_rate`。后续可把「实测优于社区默认」的配置沉淀为新的候选 spec（社区知识 → 实测 → 回馈）。

## 12. 分阶段落地

1. **P1**：provenance schema 字段 + 生成器（vLLM/SGLang adapter）+ GLM-5.2 端到端验证 + CI 校验。产出更好的 catalog 内容（Tier 0）。
2. **P2**：后端可行性/选型 dry-run 接口（§10，含 §5.2 拓扑自动选型）。
3. **P3**：前端三轴选择器（后端/量化/Mode）+ 徽章 + 只读拓扑解析结果（Tier 1）。
4. **P4**：auto-tune 回填闭环。

## 13. 待决问题

- §5.1 DP-Attention/EP 的并行度翻译：采用 (A) 还是 (B)？
- §5.2 拓扑自动选型的平局策略是否就是「最少节点 → verified → 最小占用」？是否需要允许用户在高级项手动覆盖成多机？
- Mode 语义混合（目标 latency/throughput + 机制 tp/dep）：UI 标签是否改为「预设 / Profile」？vLLM `strategy_overrides` 为空期间，vLLM 是否就固定单一 Mode（standard）？
- 生成器运行时机：CI 漂移校验 vs 定期自动开 PR。
