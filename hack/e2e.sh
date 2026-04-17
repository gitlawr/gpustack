#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

# E2E test configuration with defaults
# GPUSTACK_SERVER_URL / GPUSTACK_ADMIN_PASSWORD are the canonical env vars.
# E2E_SERVER_URL / E2E_PASSWORD are aliases for convenience.
E2E_SERVER_URL="${E2E_SERVER_URL:-${GPUSTACK_SERVER_URL:-http://localhost:80}}"
E2E_PASSWORD="${E2E_PASSWORD:-${GPUSTACK_ADMIN_PASSWORD:-Admin@123}}"
E2E_IMAGE="${E2E_IMAGE:-gpustack/gpustack:dev}"
E2E_TAGS="${E2E_TAGS:-}"
E2E_GPU="${E2E_GPU:-}"
E2E_DEPLOY="${E2E_DEPLOY:-}"
E2E_ARGS="${E2E_ARGS:-}"

function e2e() {
  local cmd="uv run pytest"

  if [[ -n "${E2E_DEPLOY}" ]]; then
    # Deploy mode: deploy via Docker then test
    gpustack::log::info "Deploy mode: ${E2E_DEPLOY}"
    export E2E_DOCKER_IMAGE="${E2E_IMAGE}"
    cmd="${cmd} e2e/tests/installation/test_deployment.py -v -m ${E2E_DEPLOY}"
    if [[ -n "${E2E_GPU}" ]]; then
      cmd="${cmd} -k ${E2E_GPU}"
    fi
  else
    # Connect mode: test against existing server
    gpustack::log::info "Connect mode: ${E2E_SERVER_URL}"
    export GPUSTACK_SERVER_URL="${E2E_SERVER_URL}"
    export GPUSTACK_ADMIN_PASSWORD="${E2E_PASSWORD}"
    cmd="${cmd} e2e/tests -v --ignore=e2e/tests/installation/test_deployment.py"
    if [[ -n "${E2E_TAGS}" ]]; then
      cmd="${cmd} -m ${E2E_TAGS}"
    fi
    if [[ -n "${E2E_GPU}" ]]; then
      cmd="${cmd} --gpu-type ${E2E_GPU}"
    fi
  fi

  if [[ -n "${E2E_ARGS}" ]]; then
    cmd="${cmd} ${E2E_ARGS}"
  fi

  gpustack::log::info "Running: ${cmd}"
  eval "${cmd}"
}

#
# main
#

gpustack::log::info "+++ E2E +++"
e2e
gpustack::log::info "--- E2E ---"
