#!/usr/bin/env python3
"""Generator: vLLM Recipes YAML + SGLang cookbook -> GPUStack catalog (final shape).

Design (see docs/proposals/community-config-catalog.md):
  * Output per model = distilled Mode specs + a unified `feature_tiles` list.
  * Modes map to GPUStack's native Standard / Latency / Throughput:
      - Standard      <- vLLM Recipes (canonical baseline; falls back to SGLang balanced)
      - Latency/Throughput <- SGLang cookbook (low-latency / high-throughput)
  * Distillation key = (backend, quantization, vendor-class, mode): the mode-defining
    STRUCTURAL flags are consistent within a vendor class; per-GPU numeric magnitudes
    that vary fall out of the cross-hardware intersection (auto-tune recovers them).
  * Model-intrinsic parsers (reasoning / tool-call) are baked into every mode spec.
  * `gpu_filters` = the quantization's TRUE requirement only (never "verified hw").
  * Topology (single/multi-node) is inferred by the scheduler, not pinned here.

Usage:
    node parse_sglang.js glm-5.2.jsx > glm-5.2.sglang.json
    python3 gen_catalog.py <vllm.yaml|-> <sglang.json|-> > out.yaml
    ('-' skips a source; a model may have only one.)
"""
import json
import re
import sys
from collections import OrderedDict

import yaml

# --- quantization + vendor-class -> TRUE hardware requirement (drives gpu_filters).
# fp8 exists on BOTH nvidia (Ada+, cc>=8.9) and AMD (MI300+), so the requirement is
# vendor-dependent. Never derived from "verified hardware".
#   nvidia: fp8->cc>=8.9, nvfp4->cc>=10.0, bf16/fp16/...-> (vendor only, or broad for vLLM)
#   amd:    mxfp4->gfx950, else vendor amd
def quant_base_filters(quant):
    """The quantization's own requirement (checkpoint/HW-feature), vendor-neutral where
    the quant itself is (fp8 exists on nvidia Ada+ AND amd; here nvidia-typed as the
    common case). bf16/fp16/awq/gptq/int* have no floor."""
    q = (quant or "").lower()
    if q == "fp8":
        return {"vendor": "nvidia", "compute_capability": ">=8.9"}
    if q == "nvfp4":
        return {"vendor": "nvidia", "compute_capability": ">=10.0"}
    if q == "mxfp4":
        return {"vendor": "amd", "vendor_variant": "gfx950"}
    return {}


# Per-flag TRUE hardware requirement. gpu_filters of a spec = quant requirement merged
# with the requirement of every flag it actually carries. Flags NOT listed are portable
# (any GPU) -> they never narrow the gate. This makes gpu_filters as broad as the config
# genuinely allows (e.g. a throughput preset of only --max-running-requests stays open).
FLAG_REQUIREMENT = {
    "--moe-a2a-backend": {"vendor": "nvidia", "compute_capability": ">=9.0"},  # deepep: Hopper+
    "--enable-dp-attention": {"vendor": "nvidia"},
    "--dsa-prefill-backend": {"vendor": "amd"},
    "--dsa-decode-backend": {"vendor": "amd"},
    "--linear-backend": {"vendor": "amd"},   # aiter
    "--moe-backend": {"vendor": "amd"},       # aiter
    "--attention-backend": {"vendor": "nvidia", "compute_capability": ">=10.0"},  # trtllm_mha
    "--mamba-radix-cache-strategy": {"vendor": "nvidia"},
    "--speculative-algorithm": {"vendor": "nvidia"},  # EAGLE/MTP not on AMD ROCm
    "--speculative-num-steps": {"vendor": "nvidia"},
    "--speculative-eagle-topk": {"vendor": "nvidia"},
    "--speculative-num-draft-tokens": {"vendor": "nvidia"},
    "--speculative-config": {"vendor": "nvidia"},  # vLLM mtp/dspark
    "--mm-attention-backend": {"vendor": "nvidia"},  # fa3/fa4
}


def _cc_floor(s):
    m = re.search(r">=\s*([0-9.]+)", s or "")
    return float(m.group(1)) if m else 0.0


def merge_req(a, b):
    out = dict(a)
    for k, v in b.items():
        if k == "compute_capability" and out.get(k):
            out[k] = out[k] if _cc_floor(out[k]) >= _cc_floor(v) else v
        elif k == "vendor" and out.get(k) and out[k] != v:
            out[k] = out[k]  # keep first (flags within a vendor group agree)
        else:
            out.setdefault(k, v)
    return out


def compute_gpu_filters(quant, params):
    gf = quant_base_filters(quant)
    for p in params:
        req = FLAG_REQUIREMENT.get(p.split("=", 1)[0])
        if req:
            gf = merge_req(gf, req)
    return gf

SGLANG_STRATEGY_TO_MODE = {
    "low-latency": "latency",
    "balanced": "standard",
    "high-throughput": "throughput",
}

MANAGED_FLAGS = {
    "--model-path", "--model", "--served-model-name",
    "--host", "--port", "--api-key",
    "--nnodes", "--node-rank", "--dist-init-addr",
}
COUNT_FLAGS = {
    "--tensor-parallel-size", "--tp", "--tp-size",
    "--pipeline-parallel-size", "--pp", "--pp-size",
    "--data-parallel-size", "--dp", "--dp-size",
}
# Model-intrinsic parsers: always baked into a mode spec, never a toggle tile.
PARSER_FLAGS = {"--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice"}


def vendor_class(hw):
    hw = (hw or "").lower()
    if hw.startswith("mi") or hw.startswith("amd") or "gfx" in hw:
        return "amd"
    if hw in ("cpu", "xeon"):
        return "cpu"
    return "nvidia"


# ---------------------------------------------------------------------------
# flag parsing / normalization
# ---------------------------------------------------------------------------
def pair_args(tokens):
    pairs, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if " " in tok and tok.startswith("--"):
            k, v = tok.split(" ", 1)
            pairs.append((k.strip(), v.strip()))
            i += 1
        elif tok.startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                pairs.append((tok, tokens[i + 1]))
                i += 2
            else:
                pairs.append((tok, None))
                i += 1
        else:
            i += 1
    return pairs


def _is_placeholder(v):
    return v is not None and re.fullmatch(r"\{\{[A-Z0-9_]+\}\}", v.strip()) is not None


def normalize(tokens):
    """Return a list of '--k=v'/'--k' strings with managed + count + placeholder
    flags stripped. Parsers are kept (they get baked into mode specs)."""
    out = []
    for k, v in pair_args(tokens):
        if k in MANAGED_FLAGS or k in COUNT_FLAGS or _is_placeholder(v):
            continue
        out.append(k if v is None else f"{k}={v}")
    return out


def _param_key(p):
    return p.split("=", 1)[0]


# ---------------------------------------------------------------------------
# vLLM adapter -> Standard mode specs + tiles
# ---------------------------------------------------------------------------
def vllm_semantic_tokens(recipe, variant_key):
    """base_args + model-intrinsic parsers only. All other features (default-on or
    opt-in) surface as toggle tiles, so they aren't double-counted here."""
    tokens = list(recipe["model"].get("base_args", []))
    for name, f in (recipe.get("features") or {}).items():
        if name in ("tool_calling", "reasoning"):
            tokens += f.get("args", [])
    tokens += recipe["variants"][variant_key].get("extra_args", [])
    return tokens


def vllm_standard_specs(recipe, family):
    specs = []
    min_ver = recipe["model"].get("min_vllm_version")
    for variant_key, variant in recipe["variants"].items():
        quant = variant.get("precision", variant_key)
        repo = variant.get("model_id", recipe["model"]["model_id"])
        params = normalize(vllm_semantic_tokens(recipe, variant_key))
        specs.append(_mode_spec(
            family, "vLLM", repo, quant, "standard", params, min_ver,
            _vllm_url(recipe)))
    return specs


def vllm_tiles(recipe):
    """Opt-in + non-parser features become toggleable tiles."""
    tiles = []
    opt_in = set(recipe.get("opt_in_features", []))
    for name, f in (recipe.get("features") or {}).items():
        if name in ("tool_calling", "reasoning"):
            continue  # parsers -> baked, not tiles
        desc = f.get("description", "")
        if "modes" in f:
            for mid, md in f["modes"].items():
                tiles.append(_tile(
                    f"{name}:{mid}", f"{name} ({mid})", desc, "vLLM",
                    normalize(md.get("args", [])), default_on=name not in opt_in))
        else:
            tiles.append(_tile(
                name, name, desc, "vLLM", normalize(f.get("args", [])),
                default_on=name not in opt_in))
    return tiles


def _vllm_url(recipe):
    return ("https://github.com/vllm-project/recipes/blob/main/models/"
            f"{recipe['model']['model_id']}.yaml")


# ---------------------------------------------------------------------------
# SGLang adapter -> Latency / Throughput (and fallback Standard) + tiles
# ---------------------------------------------------------------------------
def sglang_distilled_specs(cfg, family):
    """Distill cells into one spec per (quant, vendor-class, mode) via cross-hardware
    intersection of normalized flags within that group."""
    model_names = cfg.get("modelNames", {})
    parsers = _sglang_parser_flags(cfg)
    groups = OrderedDict()  # (quant, vclass, strategy) -> {repo, flagsets:[...]}
    for cell in cfg.get("cells", []):
        m = cell["match"]
        vclass = vendor_class(m["hw"])
        if vclass == "cpu":
            continue
        repo = model_names.get(f"{m['variant']}|{m['quant']}")
        if not repo:
            continue
        key = (m["quant"], vclass, m["strategy"])
        g = groups.setdefault(key, {"repo": repo, "sets": []})
        g["sets"].append(normalize(cell.get("flags", [])))

    specs = []
    for (quant, vclass, strategy), g in groups.items():
        sets = g["sets"]
        common = set(sets[0])
        for s in sets[1:]:
            common &= set(s)
        ordered = [p for p in sets[0] if p in common]
        # bake parsers (SGLang keeps them out of cell flags)
        for p in parsers:
            if _param_key(p) not in {_param_key(x) for x in ordered}:
                ordered.append(p)
        specs.append(_mode_spec(
            family, "SGLang", g["repo"], quant,
            SGLANG_STRATEGY_TO_MODE.get(strategy, "standard"),
            ordered, None, _sglang_url(cfg)))
    return specs


def _sglang_parser_flags(cfg):
    out = []
    for item in (cfg.get("playgroundFeatures", {}).get("parsers", {}) or {}).get("items", []):
        if item.get("flag"):
            out += normalize([item["flag"]])
    return out


def sglang_tiles(cfg):
    tiles = []
    pf = cfg.get("playgroundFeatures", {})

    def disable_of(o):
        d = o.get("disable")
        if not d:
            return []
        d = d if isinstance(d, list) else [d]
        res = []
        for item in d:
            when = item.get("when", item) if isinstance(item, dict) else {}
            res.append({"hw": when.get("hw"), "reason": item.get("reason", "")})
        return res

    def walk(o):
        if isinstance(o, dict):
            if o.get("flags") or o.get("flag"):
                raw = o.get("flags") or [o.get("flag")]
                if not any("parser" in f for f in raw):  # parsers baked
                    tiles.append(_tile(
                        str(o.get("id") or o.get("label")),
                        o.get("label", o.get("id", "")),
                        o.get("description", ""), "SGLang",
                        normalize(raw), default_on=False, disable_by_hw=disable_of(o)))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(pf)
    return tiles


def _sglang_url(cfg):
    return ("https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/"
            f"configs/{cfg.get('github', {}).get('cookbookModel', '')}.jsx")


# ---------------------------------------------------------------------------
# shared spec / tile builders
# ---------------------------------------------------------------------------
def _mode_spec(family, backend, repo, quant, mode, params, backend_version, url):
    spec = OrderedDict()
    spec["name"] = family
    spec["backend"] = backend
    if backend_version:
        spec["backend_version"] = backend_version
    spec["source"] = "huggingface"
    spec["huggingface_repo_id"] = repo
    spec["quantization"] = quant.upper()
    spec["mode"] = mode
    gf = compute_gpu_filters(quant, params)
    if gf:
        spec["gpu_filters"] = gf
    if params:
        spec["backend_parameters"] = params
    spec["provenance"] = OrderedDict(
        source="vllm-recipes" if backend == "vLLM" else "sglang-cookbook",
        source_url=url, distilled=True)
    return spec


def _tile(tid, label, desc, backend, flags, default_on=False, disable_by_hw=None):
    t = OrderedDict()
    t["id"] = tid
    t["label"] = label
    if desc:
        t["description"] = desc
    t["backends"] = [backend]
    if flags:
        t["flags"] = flags
    t["default_on"] = default_on
    if disable_by_hw:
        t["disable_by_hw"] = disable_by_hw
    return t


# ---------------------------------------------------------------------------
# assembly (mode division of labor) + emit
# ---------------------------------------------------------------------------
def assemble(vllm_recipe, sglang_cfg):
    meta = (vllm_recipe or sglang_cfg).get("meta", {}) if vllm_recipe else {}
    family = (vllm_recipe["meta"]["title"] if vllm_recipe
              else sglang_cfg.get("modelName"))

    vllm_std = vllm_standard_specs(vllm_recipe, family) if vllm_recipe else []
    sg_specs = sglang_distilled_specs(sglang_cfg, family) if sglang_cfg else []

    # Standard <- vLLM; Latency/Throughput <- SGLang; SGLang standard only fills
    # (quant, vendor) gaps vLLM didn't cover. A vLLM standard with no vendor gate
    # (broad, e.g. bf16) covers that quant for every vendor.
    covered = {(s["quantization"], s.get("gpu_filters", {}).get("vendor"))
               for s in vllm_std}
    covered_any_vendor = {q for (q, v) in covered if v is None}
    specs = list(vllm_std)
    for s in sg_specs:
        if s["mode"] == "standard":
            q = s["quantization"]
            v = s.get("gpu_filters", {}).get("vendor")
            if q in covered_any_vendor or (q, v) in covered:
                continue
        specs.append(s)

    tiles = (vllm_tiles(vllm_recipe) if vllm_recipe else []) + \
            (sglang_tiles(sglang_cfg) if sglang_cfg else [])

    ms = OrderedDict()
    ms["name"] = family
    if vllm_recipe:
        ms["description"] = vllm_recipe["meta"].get("description")
    ms["categories"] = ["llm"]
    ms["specs"] = specs
    ms["feature_tiles"] = tiles
    return ms


def _load(path):
    if path == "-":
        return None
    with open(path) as f:
        return yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)


def main():
    vllm_recipe = _load(sys.argv[1])
    sglang_cfg = _load(sys.argv[2])
    ms = assemble(vllm_recipe, sglang_cfg)

    yaml.add_representer(
        OrderedDict,
        lambda d, data: d.represent_mapping("tag:yaml.org,2002:map", data.items()))

    from collections import Counter
    modes = Counter(s["mode"] for s in ms["specs"])
    sys.stderr.write(
        f"{ms['name']}: {len(ms['specs'])} mode-specs {dict(modes)}, "
        f"{len(ms['feature_tiles'])} feature tiles\n")
    print(yaml.dump({"model_sets": [ms]}, sort_keys=False,
                    default_flow_style=False, width=100, allow_unicode=True))


if __name__ == "__main__":
    main()
