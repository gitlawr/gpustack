# GPUStack E2E Test Framework

## Overview

This test framework is designed for GPUStack release regression testing. It supports flexible test case selection via tags, suitable for different GPU environments.

## Prerequisites

- Docker installed and configured
- NVIDIA Container Toolkit configured (for NVIDIA GPU environments)
- Python 3.10+
- Network access (for pulling models and images)

## Directory Structure

```
e2e/
├── README.md                 # This document
├── conftest.py               # pytest fixtures and configuration
├── pytest.ini                # pytest configuration and markers
├── config.yaml.example       # Configuration example
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── client.py             # GPUStack API client
│   ├── config.py             # Configuration management
│   ├── docker.py             # Docker operations
│   ├── k8s.py                # Kubernetes operations
│   ├── models.py             # Model operation helpers
│   └── wait.py               # Wait and polling utilities
├── fixtures/                 # Test data
│   └── __init__.py
└── tests/                    # Test cases
    ├── installation/         # Installation and deployment tests
    ├── model/                # Model deployment tests
    ├── worker/               # Worker management tests
    ├── upgrade/              # Upgrade tests
    ├── provider/             # Provider tests
    ├── route/                # Route tests
    ├── stability/            # Stability tests
    ├── multimodal/           # Multimodal model tests
    ├── distributed/          # Distributed deployment tests
    └── benchmark/            # Benchmark tests
```

## Test Modes

### 1. Connect Mode (Default)

Connect to an existing GPUStack deployment and run tests:

```bash
make e2e
E2E_PASSWORD=123456 make e2e
E2E_SERVER_URL=http://192.168.1.100:80 E2E_PASSWORD=123456 make e2e
```

### 2. Deploy Mode

Deploy GPUStack via Docker, then run tests:

```bash
E2E_DEPLOY=allinone make e2e
E2E_DEPLOY=server make e2e
E2E_DEPLOY=allinone E2E_IMAGE=gpustack/gpustack:v0.5 make e2e
```

## Tag System

### GPU Type Tags
- `@pytest.mark.nvidia` - Requires NVIDIA GPU
- `@pytest.mark.amd` - Requires AMD GPU
- `@pytest.mark.ascend` - Requires Huawei Ascend NPU
- `@pytest.mark.wsl` - Windows WSL environment
- `@pytest.mark.cpu_only` - CPU only

### Deployment Mode Tags
- `@pytest.mark.allinone` - All-in-one deployment
- `@pytest.mark.server_only` - Server-only deployment
- `@pytest.mark.distributed` - Distributed deployment

### Feature Tags
- `@pytest.mark.installation` - Installation tests
- `@pytest.mark.model` - Model deployment tests
- `@pytest.mark.worker` - Worker management tests
- `@pytest.mark.upgrade` - Upgrade tests
- `@pytest.mark.provider` - Provider tests
- `@pytest.mark.route` - Route tests
- `@pytest.mark.stability` - Stability tests
- `@pytest.mark.multimodal` - Multimodal tests (TTS/ASR/Image)
- `@pytest.mark.benchmark` - Benchmark tests

### Backend Tags
- `@pytest.mark.vllm` - vLLM backend
- `@pytest.mark.sglang` - SGLang backend
- `@pytest.mark.mindie` - MindIE backend (Ascend)

### Worker Type Tags
- `@pytest.mark.do_worker` - DigitalOcean Worker
- `@pytest.mark.k8s_worker` - Kubernetes Worker

### Priority Tags
- `@pytest.mark.smoke` - Smoke tests (core functionality)
- `@pytest.mark.regression` - Regression tests (full)
- `@pytest.mark.slow` - Slow running tests

## Quick Start

```bash
# Run all tests against existing server (localhost:80, password Admin@123)
make e2e

# Custom password
E2E_PASSWORD=123456 make e2e

# Smoke tests only
E2E_TAGS=smoke make e2e

# Model tests on NVIDIA GPU
E2E_TAGS=model E2E_GPU=nvidia make e2e

# Deploy all-in-one via Docker, then test
E2E_DEPLOY=allinone make e2e

# Deploy with specific image
E2E_DEPLOY=allinone E2E_IMAGE=gpustack/gpustack:v0.5 make e2e

# Extra pytest args
E2E_ARGS="-k test_version" make e2e
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `E2E_SERVER_URL` | GPUStack server URL | `http://localhost:80` |
| `E2E_PASSWORD` | Admin password | `Admin@123` |
| `E2E_TAGS` | Pytest markers (smoke, model, provider, route, stability) | - |
| `E2E_GPU` | GPU type filter (nvidia, amd, ascend, cpu) | - |
| `E2E_DEPLOY` | Deploy via Docker before testing (allinone, server) | - |
| `E2E_IMAGE` | Docker image, used with `E2E_DEPLOY` | `gpustack/gpustack:dev` |
| `E2E_ARGS` | Additional pytest arguments | - |

## Advanced Usage

### Direct pytest Commands

```bash
# Run specific test file
uv run pytest e2e/tests/model/test_basic_deployment.py -v

# Run with multiple markers
uv run pytest e2e/tests -v -m "nvidia and vllm and smoke"

# Exclude certain tests
uv run pytest e2e/tests -v -m "not upgrade"

# Generate HTML report
uv run pytest e2e/tests -v --html=e2e_report.html --self-contained-html

# Generate JUnit XML report (for CI)
uv run pytest e2e/tests -v --junitxml=e2e_results.xml
```

## Test Case List

| ID | Description | Tags |
|----|-------------|------|
| 1 | Verify RC version UI/backend display | smoke, installation |
| 2 | All-in-one/server-only deployment, catalog model deployment | smoke, installation, allinone, vllm, sglang, nvidia |
| 3 | Add DO Worker, deploy/delete model | worker, do_worker, model |
| 4 | Add K8s Worker, deploy/delete model | worker, k8s_worker, model |
| 5 | Windows WSL installation | installation, wsl, nvidia |
| 6 | AMD GPU model deployment | installation, model, amd |
| 7 | Ascend NPU model deployment | installation, model, ascend |
| 8 | Version upgrade, verify API key and model compatibility | upgrade |
| 9 | vLLM multimodal model deployment (CUDA) | multimodal, vllm, nvidia |
| 10 | vLLM multimodal model deployment (Ascend) | multimodal, vllm, ascend |
| 11 | vLLM/SGLang multi-replica deployment | model, vllm, sglang, distributed |
| 12 | vLLM distributed deployment | distributed, vllm |
| 13 | SGLang distributed deployment | distributed, sglang |
| 14 | MindIE distributed deployment | distributed, mindie, ascend |
| 15 | Add community inference backend | model |
| 16 | Benchmark functionality | benchmark |
| 17 | Grafana dashboard verification | smoke, monitoring |
| 18 | Doubao provider | provider |
| 19 | Qwen provider | provider |
| 20 | OpenAI provider | provider |
| 21 | Route modification and model access | route |
| 22 | Fallback route verification | route |
| 23 | Delete and redeploy model | stability, model |
| 24 | Worker restart model verification | stability |
| 25 | Server restart model verification | stability |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPUSTACK_SERVER_URL` | GPUStack Server URL | `http://localhost:80` |
| `GPUSTACK_ADMIN_PASSWORD` | Admin password | - |
| `GPUSTACK_API_KEY` | API key (optional) | - |
| `E2E_CONFIG_FILE` | Config file path | `e2e/config.yaml` |
| `E2E_TEST_MODE` | Test mode (connect/deploy) | `connect` |
| `E2E_DOCKER_IMAGE` | Docker image for deploy mode | `gpustack/gpustack:dev` |
| `E2E_SKIP_CLEANUP` | Skip cleanup after tests | `false` |
| `E2E_GPU_TYPE` | GPU type | Auto-detect |

## Writing New Tests

### Example: Add a new model deployment test

```python
import pytest
from e2e.utils.client import GPUStackClient
from e2e.utils.models import ModelHelper

@pytest.mark.nvidia
@pytest.mark.vllm
@pytest.mark.model
@pytest.mark.smoke
class TestModelDeployment:
    """Model deployment tests."""
    
    def test_deploy_qwen_from_catalog(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config,
        cleanup_models,
    ):
        """Deploy Qwen model from catalog."""
        model = model_helper.deploy_huggingface_model(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            name="e2e-test-qwen",
            backend="vLLM",
            wait=True,
            timeout=600,
        )
        
        cleanup_models.append(model["id"])
        
        # Verify model inference
        response = model_helper.verify_model_inference(
            model_name=model["name"],
            prompt="Hello",
        )
        assert response["choices"][0]["message"]["content"]
```

## Troubleshooting

### Common Issues

1. **Cannot connect to GPUStack Server**
   - Check if `GPUSTACK_SERVER_URL` is correct
   - Verify server is running and port is accessible

2. **Model deployment timeout**
   - Check GPU resource availability
   - Check network connectivity for model download
   - Increase `timeout` parameter

3. **Permission errors**
   - Verify `GPUSTACK_ADMIN_PASSWORD` or `GPUSTACK_API_KEY` is correct

4. **Docker errors in deploy mode**
   - Ensure Docker is running
   - Check if NVIDIA Container Toolkit is properly configured
   - Try with `use_sudo: true` in config if permission denied
