#!/bin/bash
#
# GPUStack E2E Test Runner Script
#
# Usage:
#   ./e2e/run_tests.sh [options]
#
# Options:
#   --smoke           Run smoke tests only
#   --gpu nvidia|amd|ascend|cpu  Specify GPU type
#   --tags "tag1 tag2"  Specify test tags
#   --exclude "tag"     Exclude certain tags
#   --parallel N        Parallel execution (requires pytest-xdist)
#   --report            Generate HTML report
#   --help              Show help
#

set -e

# Default values
GPU_TYPE=""
TAGS=""
EXCLUDE=""
PARALLEL=""
REPORT=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            TAGS="smoke"
            shift
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --report)
            REPORT="1"
            shift
            ;;
        --help)
            echo "GPUStack E2E Test Runner Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --smoke             Run smoke tests only"
            echo "  --gpu TYPE          Specify GPU type (nvidia|amd|ascend|cpu)"
            echo "  --tags 'tag1 tag2'  Specify test tags (space separated)"
            echo "  --exclude 'tag'     Exclude certain tags"
            echo "  --parallel N        Parallel execution with N processes"
            echo "  --report            Generate HTML report"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --smoke --gpu nvidia"
            echo "  $0 --tags 'model vllm' --gpu nvidia"
            echo "  $0 --tags 'provider' --exclude 'slow'"
            echo "  $0 --report --parallel 4"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Change to project root directory
cd "$(dirname "$0")/.."

# Build pytest command
CMD="pytest e2e/tests -v"

# Add GPU type
if [ -n "$GPU_TYPE" ]; then
    CMD="$CMD --gpu-type $GPU_TYPE"
fi

# Add tag filter
if [ -n "$TAGS" ]; then
    # Convert space-separated tags to "and" connected
    MARKER_EXPR="${TAGS// / and }"
    CMD="$CMD -m '$MARKER_EXPR'"
fi

# Add exclude tags
if [ -n "$EXCLUDE" ]; then
    if [ -n "$TAGS" ]; then
        CMD="$CMD and not $EXCLUDE"
    else
        CMD="$CMD -m 'not $EXCLUDE'"
    fi
fi

# Add parallel execution
if [ -n "$PARALLEL" ]; then
    CMD="$CMD -n $PARALLEL"
fi

# Add report
if [ -n "$REPORT" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CMD="$CMD --html=e2e_report_${TIMESTAMP}.html --self-contained-html"
fi

# Add extra arguments
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Executing: $CMD"
echo ""

# Run tests
eval "$CMD"
