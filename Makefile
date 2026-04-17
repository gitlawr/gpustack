# Detect operating system
ifeq ($(OS),Windows_NT)
    PLATFORM_SHELL := powershell
    SCRIPT_EXT := .ps1
    SCRIPT_DIR := hack/windows
else
    PLATFORM_SHELL := /bin/bash
    SCRIPT_EXT := .sh
    SCRIPT_DIR := hack
endif

# Borrowed from https://stackoverflow.com/questions/18136918/how-to-get-current-relative-directory-of-your-makefile
curr_dir := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Borrowed from https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
rest_args := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))

$(eval $(rest_args):;@:)

# List targets based on script extension and directory
ifeq ($(OS),Windows_NT)
    targets := $(shell powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Path $(curr_dir)/$(SCRIPT_DIR) | Select-Object -ExpandProperty BaseName")
else
	targets := $(shell ls $(curr_dir)/$(SCRIPT_DIR) | grep $(SCRIPT_EXT) | sed 's/$(SCRIPT_EXT)$$//')
endif

$(targets):
	@$(eval TARGET_NAME=$@)
ifeq ($(PLATFORM_SHELL),/bin/bash)
	$(curr_dir)/$(SCRIPT_DIR)/$(TARGET_NAME)$(SCRIPT_EXT) $(rest_args)
else
	powershell -NoProfile -ExecutionPolicy Bypass "$(curr_dir)/$(SCRIPT_DIR)/$(TARGET_NAME)$(SCRIPT_EXT) $(rest_args)"
endif

help:
	#
	# Usage:
	#
	#   * [dev] `make install`, install development tools, like uv, pre-commit hooks and so on.
	#
	#   * [dev] `make deps`, prepare all dependencies.
	#
	#   * [dev] `make generate`, generate codes.
	#
	#   * [dev] `make lint`, check style.
	#
	#   * [dev] `make test`, execute unit testing.
	#
	#   * [dev] `make build`, execute building.
	#
	#   * [dev] `make build-docs`, build docs, not supported on Windows.
	#
	#   * [dev] `make serve-docs`, serve docs, not supported on Windows.
	#
	#   * [ci]  `make package`, build container images, not supported on Windows.
	#
	#   * [ci]  `make ci`, execute `make install`, `make deps`, `make lint`, `make test`, `make build`.
	#
	#   * [e2e] `make e2e`, run E2E tests.
	#           E2E_TAGS     - filter tests: smoke, model, provider, route, stability, installation.
	#           E2E_GPU      - GPU type: nvidia, amd, ascend, cpu.
	#           E2E_DEPLOY   - set to deploy GPUStack via Docker before testing: allinone (default), server.
	#           E2E_IMAGE    - Docker image (default: gpustack/gpustack:dev), used with E2E_DEPLOY.
	#           Examples:
	#             make e2e                                       # all tests against existing server
	#             E2E_TAGS=smoke make e2e                        # smoke tests
	#             E2E_TAGS=model E2E_GPU=nvidia make e2e         # model tests on NVIDIA
	#             E2E_DEPLOY=allinone make e2e                   # deploy all-in-one then test
	#             E2E_DEPLOY=server E2E_IMAGE=gpustack/gpustack:v0.5 make e2e
	#
	@echo

# E2E Testing
E2E_SERVER_URL ?= http://localhost:80
E2E_PASSWORD ?= Admin@123
E2E_IMAGE ?= gpustack/gpustack:dev
E2E_TAGS ?=
E2E_GPU ?=
E2E_DEPLOY ?=
E2E_ARGS ?=

E2E_MARKER_EXPR := $(if $(E2E_TAGS),-m "$(E2E_TAGS)",)
E2E_GPU_OPT := $(if $(E2E_GPU),--gpu-type $(E2E_GPU),)

ifdef E2E_DEPLOY
e2e:
	E2E_DOCKER_IMAGE=$(E2E_IMAGE) \
	uv run pytest e2e/tests/installation/test_deployment.py -v \
		-m "$(E2E_DEPLOY)" \
		$(if $(E2E_GPU),-k "$(E2E_GPU)",) $(E2E_ARGS)
else
e2e:
	GPUSTACK_SERVER_URL=$(E2E_SERVER_URL) \
	GPUSTACK_ADMIN_PASSWORD=$(E2E_PASSWORD) \
	uv run pytest e2e/tests -v \
		--ignore=e2e/tests/installation/test_deployment.py \
		$(E2E_MARKER_EXPR) $(E2E_GPU_OPT) $(E2E_ARGS)
endif

.DEFAULT_GOAL := build
.PHONY: $(targets)
