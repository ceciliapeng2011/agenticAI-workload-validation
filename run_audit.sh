#!/bin/bash
# Master script: Run both audit scripts on one or more repos
# Usage: ./run_audit.sh <repo1_path> [repo2_path] [repo3_path] ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT1="$SCRIPT_DIR/1_bench_harness_audit.py"
SCRIPT2="$SCRIPT_DIR/2_ci_config_audit.py"

# Color codes
BLUE='\033[94m'
GREEN='\033[92m'
RED='\033[91m'
RESET='\033[0m'

if [ $# -eq 0 ]; then
    echo "Usage: $0 <repo1_path> [repo2_path] [repo3_path] ..."
    echo ""
    echo "Example:"
    echo "  $0 ../vllm ../sglang ../TensorRT-LLM"
    echo ""
    echo "This will run both audit scripts (harness + CI) on each repo."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${RESET}"
    exit 1
fi

# Check for required Python modules
python3 -c "import yaml" 2>/dev/null || {
    echo -e "${YELLOW}Installing pyyaml...${RESET}"
    pip install pyyaml
}

echo -e "${BLUE}========================================${RESET}"
echo -e "${BLUE}Agentic AI Runtime Audit Suite${RESET}"
echo -e "${BLUE}========================================${RESET}"
echo ""

for repo_path in "$@"; do
    if [ ! -d "$repo_path" ]; then
        echo -e "${RED}✗ Repo not found: $repo_path${RESET}"
        continue
    fi

    repo_name=$(basename "$repo_path")
    echo -e "${GREEN}■━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${GREEN}REPO: ${repo_name}${RESET}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""

    echo -e "${BLUE}[1/2] Running Benchmark Harness Audit...${RESET}"
    echo ""
    python3 "$SCRIPT1" "$repo_path" 2>/dev/null || true
    echo ""

    echo -e "${BLUE}[2/2] Running CI Configuration Audit...${RESET}"
    echo ""
    python3 "$SCRIPT2" "$repo_path" 2>/dev/null || true
    echo ""
    echo ""
done

echo -e "${BLUE}========================================${RESET}"
echo -e "${GREEN}✓ Audit complete${RESET}"
echo -e "${BLUE}========================================${RESET}"
