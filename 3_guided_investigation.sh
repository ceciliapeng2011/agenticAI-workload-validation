#!/bin/bash
#
# Guided Investigation: How does this runtime evaluate agentic workload perf/accuracy?
#
# This script walks you through specific files/lines found by scripts 1 & 2,
# shows you the relevant code, asks you a question about what you see, and
# accumulates your answers into a structured markdown report.
#
# It does NOT do the reading for you — it just points you at the right place
# and captures your conclusions before you forget them.
#
# Usage: ./3_guided_investigation.sh <path_to_repo> [output.md]

set -uo pipefail

REPO="${1:?Usage: $0 <path_to_repo> [output.md]}"
OUT="${2:-./investigation_$(basename "$REPO")_$(date +%Y%m%d).md}"

if [ ! -d "$REPO" ]; then
    echo "ERROR: repo path not found: $REPO"
    exit 1
fi

REPO="$(cd "$REPO" && pwd)"

BLUE='\033[94m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
BOLD='\033[1m'
RESET='\033[0m'

# ── init output file ─────────────────────────────────────────────────
cat > "$OUT" <<EOF
# Agentic Workload Evaluation — Guided Investigation

**Repo:** $(basename "$REPO")
**Date:** $(date +%Y-%m-%d)

This report was built by walking through specific evidence locations and
recording what was actually found in the code (not guessed from docs/blogs).

---
EOF

section_count=0
checkpoint_count=0

# ── helpers ───────────────────────────────────────────────────────────

section() {
    section_count=$((section_count+1))
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${BLUE}  SECTION $section_count: $1${RESET}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${RESET}"
    echo "" >> "$OUT"
    echo "## Section $section_count: $1" >> "$OUT"
    echo "" >> "$OUT"
}

# checkpoint <title> <shell-command-to-show-evidence> <question-to-ask>
# The shell command is eval'd with REPO in scope (cd into repo first).
checkpoint() {
    checkpoint_count=$((checkpoint_count+1))
    local title="$1"
    local cmd="$2"
    local question="$3"

    echo ""
    echo -e "${YELLOW}── Checkpoint $checkpoint_count: $title ──${RESET}"
    echo ""

    local evidence
    evidence="$(cd "$REPO" && eval "$cmd" 2>&1)"

    if [ -z "$evidence" ]; then
        echo -e "${RED}(no output — file/pattern not found; command was: $cmd)${RESET}"
        evidence="(no output for: $cmd)"
    else
        echo "$evidence" | head -60
        local total_lines
        total_lines=$(echo "$evidence" | wc -l)
        if [ "$total_lines" -gt 60 ]; then
            echo -e "${YELLOW}... ($total_lines total lines, truncated to 60 above)${RESET}"
        fi
    fi

    echo ""
    echo -e "${BOLD}Q: $question${RESET}"
    echo -e "${YELLOW}(type your answer, then press Enter. Empty = skip this checkpoint)${RESET}"
    read -r -p "> " answer

    {
        echo "### Checkpoint $checkpoint_count: $title"
        echo ""
        echo "**Command:** \`$cmd\`"
        echo ""
        echo '```'
        echo "$evidence" | head -60
        echo '```'
        echo ""
        echo "**Question:** $question"
        echo ""
        if [ -n "$answer" ]; then
            echo "**Finding:** $answer"
        else
            echo "**Finding:** _(skipped)_"
        fi
        echo ""
    } >> "$OUT"
}

# free_note <prompt> — for open-ended notes not tied to a specific file
free_note() {
    local question="$1"
    echo ""
    echo -e "${BOLD}$question${RESET}"
    read -r -p "> " answer
    {
        echo "**Note:** $question"
        echo ""
        echo "${answer:-_(skipped)_}"
        echo ""
    } >> "$OUT"
}

echo -e "${BOLD}${GREEN}Guided Investigation: $(basename "$REPO")${RESET}"
echo "Output will be written to: $OUT"
echo ""
echo "For each checkpoint: read the code shown, then answer the question in your own words."
echo "Press Ctrl+C anytime to stop early — everything answered so far is already saved."
echo ""
read -r -p "Press Enter to start..."

# ══════════════════════════════════════════════════════════════════════
section "Agentic / Tool-Call CI — what does it actually test?"
# ══════════════════════════════════════════════════════════════════════

checkpoint \
    "tool-call mentions in CI configs" \
    "grep -rn -i 'tool.call' .buildkite/*.yaml .github/workflows/*.yml 2>/dev/null | head -40" \
    "Which CI files reference tool-calling, and what test files/scripts do they invoke?"

checkpoint \
    "the actual test files those CI jobs run" \
    "grep -rn -i 'tool.call' .buildkite/test-amd.yaml .buildkite/rust_frontend.yaml 2>/dev/null | grep -oE '[a-zA-Z0-9_/.-]+\.py' | sort -u | head -20" \
    "Open a couple of these test files. Are they unit tests of a parser, or end-to-end tool-call accuracy tests?"

checkpoint \
    "tool_parsers test coverage" \
    "find . -path '*/tests/*tool_parser*' -o -path '*/tests/*tool_call*' 2>/dev/null | head -30" \
    "Do these tests check PARSING correctness (does the string get parsed into JSON right) or ACCURACY (does the model choose the right tool/args)? This distinction matters a lot."

# ══════════════════════════════════════════════════════════════════════
section "Multi-turn / Agentic Benchmark Scripts — how is the workload shaped?"
# ══════════════════════════════════════════════════════════════════════

checkpoint \
    "benchmark_prefix_caching.py dataset construction" \
    "sed -n '70,120p' benchmarks/benchmark_prefix_caching.py 2>/dev/null" \
    "How does this script build a 'multi-turn' request? Does it replay real conversation turns, or synthesize a fixed-length prefix + random suffix?"

checkpoint \
    "what metrics benchmark_prefix_caching.py reports" \
    "grep -n -iE 'latency|throughput|ttft|tpot|hit.?rate|p50|p90|p99' benchmarks/benchmark_prefix_caching.py 2>/dev/null | head -30" \
    "What's the metric vocabulary here — request-level (TTFT/TPOT) or cache-level (hit rate)? Is there any TASK-level metric (e.g. per-conversation completion time)?"

checkpoint \
    "disaggregated multiturn proxy" \
    "sed -n '1,60p' examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py 2>/dev/null" \
    "This uses a 'conversation_id' concept. Is this a benchmarking tool, or a production routing example? Does it measure anything, or just demonstrate session affinity?"

checkpoint \
    "rust bench multi-turn support" \
    "grep -n -iE 'multi.turn|prefix.*global.*ratio|session' rust/src/bench/README.md 2>/dev/null | head -30" \
    "The README mentioned '--multi-turn-prefix-global-ratio'. Read the surrounding doc — what workload model does this flag represent, and what does it NOT capture about real agentic sessions (e.g. tool latency, dependency between turns)?"

# ══════════════════════════════════════════════════════════════════════
section "How CI actually invokes these benchmarks"
# ══════════════════════════════════════════════════════════════════════

checkpoint \
    "benchmarks.yaml CI job definition" \
    "grep -n -B3 -A15 -iE 'prefix|multi.?turn' .buildkite/benchmarks.yaml 2>/dev/null | head -80" \
    "How often does this run (every PR / nightly / manual)? What flags/config does it use — is prefix caching or KV quantization turned ON when this perf test runs, or just default config?"

checkpoint \
    "does perf CI compare against a baseline/SLO" \
    "grep -n -iE 'baseline|regression|threshold|slo|fail.*if|assert' .buildkite/benchmarks.yaml benchmarks/benchmark_prefix_caching.py 2>/dev/null | head -30" \
    "Is there an automatic pass/fail threshold, or does this just produce numbers for a human to eyeball on a dashboard?"

# ══════════════════════════════════════════════════════════════════════
section "Accuracy evaluation — is agentic/multi-turn covered, or just single-turn?"
# ══════════════════════════════════════════════════════════════════════

checkpoint \
    "lm_eval.yaml task list" \
    "grep -n -iE 'name:|task' .buildkite/lm_eval.yaml 2>/dev/null | head -40" \
    "List the tasks you see. Are any of them multi-turn or tool-calling (e.g. BFCL, tau-bench, MT-bench)? Or is it entirely single-turn QA (gsm8k, mmlu, etc.)?"

checkpoint \
    "search for agentic-specific accuracy benchmarks anywhere in repo" \
    "grep -rniE 'bfcl|gorilla|tau.?bench|agentbench|toolbench' --include='*.py' --include='*.yaml' --include='*.md' . 2>/dev/null | grep -v '.git/' | head -30" \
    "Did anything turn up? If nothing, that itself is the finding — write 'NONE FOUND' and note what that implies for agentic accuracy assurance."

checkpoint \
    "accuracy tests under non-default config (quantization/eviction/prefix-caching ON)" \
    "grep -rln -iE 'enable_prefix_caching.*true|kv.*quant|cache.*eviction' .buildkite/*.yaml tests/ 2>/dev/null | xargs -I{} grep -l -iE 'accuracy|correctness|gsm|mmlu' {} 2>/dev/null | head -20" \
    "Are there any accuracy tests that run WITH these optimizations enabled (not just default config)? If you find one, open it — what does it actually assert?"

# ══════════════════════════════════════════════════════════════════════
section "Synthesis — write your conclusions while the evidence is fresh"
# ══════════════════════════════════════════════════════════════════════

free_note "In one paragraph: how does this runtime define and measure PERFORMANCE for agentic-like workloads (multi-turn/shared-prefix)? Name the actual metric(s) and where they're computed."

free_note "In one paragraph: how does this runtime define and measure ACCURACY for agentic-like workloads (tool-calling, multi-turn)? Or does it not, and instead only validates single-turn accuracy + assumes optimizations are 'safe' generically?"

free_note "What is the single most surprising GAP you found (something you expected to exist based on the marketing/docs, but couldn't find in code/CI)?"

free_note "What is the single most solid PRACTICE you found (something concretely well-built that's worth citing as a good example)?"

echo ""
echo -e "${GREEN}${BOLD}Investigation complete (or interrupted).${RESET}"
echo -e "Report saved to: ${BOLD}$OUT${RESET}"
echo ""
echo "Total checkpoints: $checkpoint_count"
