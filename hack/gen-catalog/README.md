# Catalog generator (prototype)

Transforms community deployment configs (vLLM Recipes, SGLang cookbook) into
GPUStack `ModelSet` entries. See
[`docs/proposals/community-config-catalog.md`](../../docs/proposals/community-config-catalog.md).

Status: **prototype**. Validated on GLM-5.2 (dual-source) and Qwen3.6-27B (mid-size,
vLLM + reconstructed SGLang). PD disaggregation is intentionally out of scope.

## Usage

```bash
# vLLM recipe: models/<org>/<model>.yaml   SGLang cookbook: configs/<org>/<model>.jsx
node parse_sglang.js glm-5.2.jsx > glm-5.2.sglang.json
python3 gen_catalog.py glm-5.2.vllm.yaml glm-5.2.sglang.json > examples/glm-5.2.catalog.yaml
# a model may have only one source; pass '-' to skip:
python3 gen_catalog.py qwen.vllm.yaml -  > out.yaml
```

## Output shape

Per model: **distilled Mode specs** + a unified **`feature_tiles`** list.

- **Modes** map to GPUStack's native `Standard` / `Latency` / `Throughput`:
  - **Standard** ← vLLM Recipes (canonical baseline; falls back to SGLang `balanced`)
  - **Latency / Throughput** ← SGLang cookbook (`low-latency` / `high-throughput`)
- **Distillation** key = `(backend, quantization, vendor-class, mode)`: the mode-defining
  STRUCTURAL flags are consistent within a vendor class, so cross-hardware **intersection**
  yields one spec per group; per-GPU numeric magnitudes that vary fall out (auto-tune
  recovers them). Can't distill across vendors (AMD has no EAGLE, uses tilelang) → AMD gets
  its own vendor-class specs.
- **Parsers** (`--reasoning-parser` / `--tool-call-parser`) are model-intrinsic → baked into
  every mode spec, never a tile.
- **`gpu_filters`** = quantization requirement merged with each flag's TRUE requirement
  (`compute_gpu_filters` = `quant_base_filters` + `FLAG_REQUIREMENT`), never "verified hw".
  So the gate is as broad as the config allows: BF16 throughput (`--max-running-requests`
  only) → `{}` GPU-unlimited; BF16 latency (EAGLE, CUDA-only) → `{vendor: nvidia}` (any cc);
  FP8 throughput (deepep→Hopper+) → cc≥9.0 vs FP8 latency → cc≥8.9. Flags not in the table
  are portable and never narrow the gate.
- **Topology** (single/multi-node) is inferred by the scheduler — not pinned here.
- **`feature_tiles`**: opt-in / non-parser capabilities from vLLM `features` ∪ SGLang
  `playgroundFeatures`, each `{id,label,flags,env,default_on,backends,disable_by_hw}`.
  The UI toggles a tile to merge its flags into `backend_parameters`; `disable_by_hw`
  greys it out on unsupported GPUs (data straight from the cookbook).

## Examples

- `examples/glm-5.2.catalog.yaml` — 15 mode specs (5 standard/latency/throughput) + 5 tiles.
- `examples/qwen3.6-27b.catalog.yaml` — 9 mode specs + 7 tiles; BF16 Standard has no
  `gpu_filters` (broadly runnable). SGLang input reconstructed from the cookbook page
  (`examples/qwen3.6-27b.sglang.recon.json`; raw jsx not on the repo's main branch).

## Known prototype limitations

- vLLM `strategy_overrides` is empty upstream today → vLLM contributes only Standard; the
  Latency/Throughput richness comes from SGLang.
- Numeric tuning (mem-fraction, chunked-prefill, max-running) is dropped from distilled
  presets on purpose; recovered per-hardware by the benchmark/auto-tune loop.
- ModelScope mirror output not yet emitted (HF repo → MS id map is a TODO).
- CPU (`--device cpu`) cells are skipped.
