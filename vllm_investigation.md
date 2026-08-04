# Agentic Workload Evaluation — Guided Investigation

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · **脚本3人工检查点记录（本文档）**
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

**Repo:** vllm
**Date:** 2026-07-30

This report was built by walking through specific evidence locations and
recording what was actually found in the code (not guessed from docs/blogs).

---

## Section 1: Agentic / Tool-Call CI — what does it actually test?

### Checkpoint 1: tool-call mentions in CI configs

**Command:** `grep -rn -i 'tool.call' .buildkite/*.yaml .github/workflows/*.yml 2>/dev/null | head -40`

```
.buildkite/test-amd.yaml:2362:  - pytest -v -s tool_use --ignore=tool_use/mistral --models llama3.2 -k "not test_response_format_with_tool_choice_required and not test_parallel_tool_calls_false and not test_tool_call_and_choice"
```

**Question:** Which CI files reference tool-calling, and what test files/scripts do they invoke?

**Finding:** _(skipped)_

### Checkpoint 2: the actual test files those CI jobs run

**Command:** `grep -rn -i 'tool.call' .buildkite/test-amd.yaml .buildkite/rust_frontend.yaml 2>/dev/null | grep -oE '[a-zA-Z0-9_/.-]+\.py' | sort -u | head -20`

```
(no output for: grep -rn -i 'tool.call' .buildkite/test-amd.yaml .buildkite/rust_frontend.yaml 2>/dev/null | grep -oE '[a-zA-Z0-9_/.-]+\.py' | sort -u | head -20)
```

**Question:** Open a couple of these test files. Are they unit tests of a parser, or end-to-end tool-call accuracy tests?

**Finding:** _(skipped)_

### Checkpoint 3: tool_parsers test coverage

**Command:** `find . -path '*/tests/*tool_parser*' -o -path '*/tests/*tool_call*' 2>/dev/null | head -30`

```
./tests/entrypoints/openai/test_tool_calls_serialization.py
./tests/entrypoints/tool_parsers
./tests/entrypoints/tool_parsers/test_granite4_tool_parser.py
./tests/entrypoints/tool_parsers/test_openai_tool_parser.py
./tests/entrypoints/tool_parsers/__init__.py
./tests/entrypoints/tool_parsers/test_hermes_tool_parser.py
./tests/tool_parsers
./tests/tool_parsers/test_internlm2_tool_parser.py
./tests/tool_parsers/test_ernie45_moe_tool_parser.py
./tests/tool_parsers/test_mistral_tool_parser.py
./tests/tool_parsers/test_hy_v3_tool_parser.py
./tests/tool_parsers/test_granite4_tool_parser.py
./tests/tool_parsers/test_rust_tool_parser.py
./tests/tool_parsers/test_granite_20b_fc_tool_parser.py
./tests/tool_parsers/utils.py
./tests/tool_parsers/test_step3_tool_parser.py
./tests/tool_parsers/__init__.py
./tests/tool_parsers/conftest.py
./tests/tool_parsers/test_longcat_tool_parser.py
./tests/tool_parsers/test_lfm2_tool_parser.py
./tests/tool_parsers/test_minimax_m2_tool_parser.py
./tests/tool_parsers/test_phi4mini_tool_parser.py
./tests/tool_parsers/test_step3p5_tool_parser.py
./tests/tool_parsers/test_llama4_pythonic_tool_parser.py
./tests/tool_parsers/test_poolside_v1_tool_parser.py
./tests/tool_parsers/test_granite_tool_parser.py
./tests/tool_parsers/test_minimax_m3_tool_parser.py
./tests/tool_parsers/test_minicpm5xml_tool_parser.py
./tests/tool_parsers/test_deepseekv32_tool_parser.py
./tests/tool_parsers/test_functiongemma_tool_parser.py
```

**Question:** Do these tests check PARSING correctness (does the string get parsed into JSON right) or ACCURACY (does the model choose the right tool/args)? This distinction matters a lot.

**Finding:** _(skipped)_


## Section 2: Multi-turn / Agentic Benchmark Scripts — how is the workload shaped?

### Checkpoint 4: benchmark_prefix_caching.py dataset construction

**Command:** `sed -n '70,120p' benchmarks/benchmark_prefix_caching.py 2>/dev/null`

```
    # Remove the special tokens.
    return random.choices(
        [v for v in vocab.values() if v not in all_special_ids],
        k=length,
    )


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
) -> list[Request]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range
    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"

    # Filter out sequences that are too long or too short
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if min_len <= prompt_len <= max_len:
```

**Question:** How does this script build a 'multi-turn' request? Does it replay real conversation turns, or synthesize a fixed-length prefix + random suffix?

**Finding:** _(skipped)_

### Checkpoint 5: what metrics benchmark_prefix_caching.py reports

**Command:** `grep -n -iE 'latency|throughput|ttft|tpot|hit.?rate|p50|p90|p99' benchmarks/benchmark_prefix_caching.py 2>/dev/null | head -30`

```
193:    print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
265:            "detokenization time in the latency measurement)"
```

**Question:** What's the metric vocabulary here — request-level (TTFT/TPOT) or cache-level (hit rate)? Is there any TASK-level metric (e.g. per-conversation completion time)?

**Finding:** _(skipped)_

### Checkpoint 6: disaggregated multiturn proxy

**Command:** `sed -n '1,60p' examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py 2>/dev/null`

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Prefill/Decode Proxy with Bidirectional KV Transfer

This proxy sits between clients and a vLLM Prefill/Decode (P/D) deployment,
routing multi-turn chat requests so that each turn reuses KV cache blocks
from the previous turn's Decode node via bidirectional KV transfer.

Architecture:
    Client  ──►  Proxy  ──►  Prefill (P)  ──►  Decode (D)
                   │              │                 │
                   │   kv_transfer_params flow:     │
                   │   D finish ──► proxy caches    │
                   │   next turn ──► proxy sends    │
                   │   cached D blocks to P ──►     │
                   │   P reads D blocks (bidir)     │
                   │   P sends its blocks to D      │

Per-request flow:
    1. Client sends chat/completions request to proxy.
    2. Proxy looks up cached D block info from the previous turn
       (keyed by conversation_id).
    3. If cache hit, proxy attaches D's block info to the request
       so P can read D's KV blocks instead of recomputing.
    4. Proxy sends request to P (max_tokens=1, non-streaming).
    5. P returns kv_transfer_params with its own block info.
    6. Proxy forwards request + P's block info to D (streaming).
    7. D streams the response. The final chunk includes D's
       kv_transfer_params, which the proxy caches for the next turn.
    8. Proxy returns D's response to the client.

Conversation isolation:
    Each request must include a ``conversation_id`` field (top-level in
    the JSON body) to scope the KV cache across turns. Without it, the
    proxy cannot link turns and falls back to no-cache behavior.

    ``conversation_id`` is a non-standard extension to the OpenAI Chat
    Completions schema, consumed by this proxy and not forwarded to the
    vLLM engine. Strict OpenAI-compatible frontends reject unknown
    fields, so clients must opt in only when targeting this proxy.

Usage:
    python disagg_proxy_multiturn.py \\
        --host 0.0.0.0 --port 8000 \\
        --prefiller-host 10.0.0.1 --prefiller-port 8100 \\
        --decoder-host 10.0.0.2 --decoder-port 8200

Benchmarking:
    Use ``benchmarks/multi_turn/benchmark_serving_multi_turn.py`` with
    the ``--send-conversation-id`` flag to inject a per-conversation
    ``conversation_id`` into every request so this proxy can key
    cross-turn KV cache reuse. The flag is *off by default*: without
    it the benchmark sends OpenAI-schema-compliant payloads and every
    turn lands as a cache MISS in this proxy.

    Example:
        python benchmarks/multi_turn/benchmark_serving_multi_turn.py \\
            --model <MODEL> --served-model-name <NAME> \\
            --url http://<proxy_host>:8000 \\
```

**Question:** This uses a 'conversation_id' concept. Is this a benchmarking tool, or a production routing example? Does it measure anything, or just demonstrate session affinity?

**Finding:** exit

### Checkpoint 7: rust bench multi-turn support

**Command:** `grep -n -iE 'multi.turn|prefix.*global.*ratio|session' rust/src/bench/README.md 2>/dev/null | head -30`

```
18:- **Beyond a single run** — concurrency/rate **sweeps**, **multi-run** stats, **multi-turn** conversations, **LoRA** multi-adapter, and result **comparison**.
309:<summary><b>Multi-turn conversations</b></summary>
312:# Synthetic multi-turn (controllable per-turn token lengths)
315:  --dataset-name random --multi-turn --multi-turn-num-turns 5 \
317:  --num-prompts 50 --multi-turn-concurrency 10 \
323:  --dataset-name random --multi-turn \
324:  --multi-turn-min-turns 2 --multi-turn-max-turns 8 \
326:  --num-prompts 100 --multi-turn-concurrency 20
331:  --dataset-name sharegpt --multi-turn \
332:  --num-prompts 50 --multi-turn-concurrency 10 --save-result
337:  --multi-turn --multi-turn-num-turns 3 --multi-turn-delay-ms 500 \
338:  --num-prompts 100 --multi-turn-concurrency 20
499:| `--num-prompts` | `1000` | Number of prompts to generate (conversations in multi-turn mode) |
636:Runs the benchmark once per value, then prints a summary table comparing throughput and latency across all sweep points and identifies the best-throughput configuration. Works in multi-turn mode too. `--sweep-summary-percentiles` appends extra TTFT/TPOT/E2EL columns to the summary, auto-adding any missing percentiles to the computed set so they also appear in result JSON.
641:<summary><b>Multi-turn conversation benchmark</b></summary>
645:| `--multi-turn` | `false` | Enable multi-turn conversation mode (requires `--backend openai-chat`) |
646:| `--multi-turn-num-turns` | `3` | Turns per conversation (synthetic mode) |
647:| `--multi-turn-min-turns` | `0` | Minimum turns per conversation (0 = use `--multi-turn-num-turns`) |
648:| `--multi-turn-max-turns` | `0` | Maximum turns per conversation (0 = `--multi-turn-num-turns` synthetic / uncapped ShareGPT) |
649:| `--multi-turn-concurrency` | — | Concurrent conversations (defaults to `--max-concurrency` or `--num-prompts`) |
650:| `--multi-turn-delay-ms` | `0` | Delay between turns in ms (simulates user think time) |
652:| `--multi-turn-prefix-global-ratio` | `0.0` | Fraction of per-turn input shared across all conversations (random dataset only) |
653:| `--multi-turn-prefix-conversation-ratio` | `0.0` | Fraction shared within each conversation (random dataset only) |
655:With `--multi-turn`, `--num-prompts` controls the number of **conversations**, not individual requests.
668:**Prefix sharing** (random dataset): when `--multi-turn-prefix-global-ratio` or `--multi-turn-prefix-conversation-ratio` is > 0, each turn sends a fixed-length message (no history accumulation) composed of a global prefix + per-conversation prefix + unique suffix. The two ratios must sum to < 1.0.
670:**Router affinity:** every turn sends `X-Session-ID: {conversation_id}` for KV-cache reuse behind a vLLM router.
686:**Assignment scope:** per request in single-shot mode; **per conversation** (sticky across all turns) in multi-turn mode, to avoid breaking prefix-cache reuse mid-dialog.
737:├── multi_turn.rs            # Multi-turn conversation orchestrator (channel workers)
756:│   ├── multi_turn.rs        # Multi-turn synthetic + ShareGPT conversation generators
763:│   ├── calculator.rs        # Percentile/throughput/goodput/peak/multi-turn computation
```

**Question:** The README mentioned '--multi-turn-prefix-global-ratio'. Read the surrounding doc — what workload model does this flag represent, and what does it NOT capture about real agentic sessions (e.g. tool latency, dependency between turns)?

**Finding:** _(skipped)_


## Section 3: How CI actually invokes these benchmarks

### Checkpoint 8: benchmarks.yaml CI job definition

**Command:** `grep -n -B3 -A15 -iE 'prefix|multi.?turn' .buildkite/benchmarks.yaml 2>/dev/null | head -80`

```
(no output for: grep -n -B3 -A15 -iE 'prefix|multi.?turn' .buildkite/benchmarks.yaml 2>/dev/null | head -80)
```

**Question:** How often does this run (every PR / nightly / manual)? What flags/config does it use — is prefix caching or KV quantization turned ON when this perf test runs, or just default config?

**Finding:** _(skipped)_

### Checkpoint 9: does perf CI compare against a baseline/SLO

**Command:** `grep -n -iE 'baseline|regression|threshold|slo|fail.*if|assert' .buildkite/benchmarks.yaml benchmarks/benchmark_prefix_caching.py 2>/dev/null | head -30`

```
benchmarks/benchmark_prefix_caching.py:102:    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"
benchmarks/benchmark_prefix_caching.py:144:        assert min_len <= prompt_len <= max_len, (
```

**Question:** Is there an automatic pass/fail threshold, or does this just produce numbers for a human to eyeball on a dashboard?

**Finding:** _(skipped)_


## Section 4: Accuracy evaluation — is agentic/multi-turn covered, or just single-turn?

### Checkpoint 10: lm_eval.yaml task list

**Command:** `grep -n -iE 'name:|task' .buildkite/lm_eval.yaml 2>/dev/null | head -40`

```
(no output for: grep -n -iE 'name:|task' .buildkite/lm_eval.yaml 2>/dev/null | head -40)
```

**Question:** List the tasks you see. Are any of them multi-turn or tool-calling (e.g. BFCL, tau-bench, MT-bench)? Or is it entirely single-turn QA (gsm8k, mmlu, etc.)?

**Finding:** _(skipped)_

### Checkpoint 11: search for agentic-specific accuracy benchmarks anywhere in repo

**Command:** `grep -rniE 'bfcl|gorilla|tau.?bench|agentbench|toolbench' --include='*.py' --include='*.yaml' --include='*.md' . 2>/dev/null | grep -v '.git/' | head -30`

```
./vllm/benchmarks/datasets/__init__.py:9:    BFCLDataset,
./vllm/benchmarks/datasets/__init__.py:53:    "BFCLDataset",
./vllm/benchmarks/datasets/datasets.py:1841:    bfcl_group = parser.add_argument_group(
./vllm/benchmarks/datasets/datasets.py:1842:        "BFCL dataset options", description=BFCLDataset.__doc__
./vllm/benchmarks/datasets/datasets.py:1844:    bfcl_group.add_argument(
./vllm/benchmarks/datasets/datasets.py:1845:        "--bfcl-categories",
./vllm/benchmarks/datasets/datasets.py:1848:        help="Comma-separated list of BFCL v3 category names (without the "
./vllm/benchmarks/datasets/datasets.py:1849:        "'BFCL_v3_' prefix or '.json' suffix) to sample from, e.g. "
./vllm/benchmarks/datasets/datasets.py:1851:        f"'{','.join(BFCLDataset.DEFAULT_CATEGORIES)}'.",
./vllm/benchmarks/datasets/datasets.py:2284:            args.dataset_path in BFCLDataset.SUPPORTED_DATASET_PATHS
./vllm/benchmarks/datasets/datasets.py:2285:            or args.hf_name in BFCLDataset.SUPPORTED_DATASET_PATHS
./vllm/benchmarks/datasets/datasets.py:2289:                    "BFCL dataset requires the 'openai-chat' backend because "
./vllm/benchmarks/datasets/datasets.py:2292:            dataset_class = BFCLDataset
./vllm/benchmarks/datasets/datasets.py:2293:            # BFCL does not use HF splits/subsets; stub values for base init.
./vllm/benchmarks/datasets/datasets.py:2296:            hf_kwargs = {"categories": args.bfcl_categories}
./vllm/benchmarks/datasets/datasets.py:4532:# BFCL (Berkeley Function Calling Leaderboard) Dataset Implementation
./vllm/benchmarks/datasets/datasets.py:4536:class BFCLDataset(HuggingFaceDataset):
./vllm/benchmarks/datasets/datasets.py:4539:    https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
./vllm/benchmarks/datasets/datasets.py:4541:    BFCL ships one JSON-lines file per category at the repo root (e.g.
./vllm/benchmarks/datasets/datasets.py:4542:    ``BFCL_v3_simple.json``, ``BFCL_v3_live_simple.json``) rather than a
./vllm/benchmarks/datasets/datasets.py:4549:      - translates BFCL function schemas to OpenAI tool format
./vllm/benchmarks/datasets/datasets.py:4559:        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
./vllm/benchmarks/datasets/datasets.py:4563:    # BFCL primitive type names that are not valid JSON Schema types.
./vllm/benchmarks/datasets/datasets.py:4591:        # never touch BFCL.
./vllm/benchmarks/datasets/datasets.py:4594:        filename = f"BFCL_v3_{category}.json"
./vllm/benchmarks/datasets/datasets.py:4602:                f"BFCL category '{category}' not found: file '{filename}' "
./vllm/benchmarks/datasets/datasets.py:4603:                f"does not exist in {self.dataset_path}. Check --bfcl-categories "
./vllm/benchmarks/datasets/datasets.py:4617:        """Recursively translate BFCL-flavored JSON schema to strict JSON Schema."""
./vllm/benchmarks/datasets/datasets.py:4676:            # BFCL question is list[list[dict]] — outer is turns. Use the
./vllm/benchmarks/datasets/datasets.py:4692:            # misses a significant chunk of the true input for BFCL traffic.
```

**Question:** Did anything turn up? If nothing, that itself is the finding — write 'NONE FOUND' and note what that implies for agentic accuracy assurance.

**Finding:** _(skipped)_

### Checkpoint 12: accuracy tests under non-default config (quantization/eviction/prefix-caching ON)

**Command:** `grep -rln -iE 'enable_prefix_caching.*true|kv.*quant|cache.*eviction' .buildkite/*.yaml tests/ 2>/dev/null | xargs -I{} grep -l -iE 'accuracy|correctness|gsm|mmlu' {} 2>/dev/null | head -20`

```
tests/quantization/test_per_token_kv_cache.py
tests/quantization/test_fp8.py
tests/quantization/test_quark.py
tests/models/quantization/test_per_token_kv_cache.py
tests/models/language/pooling/test_auto_prefix_cache_support.py
tests/models/language/generation/test_hybrid.py
tests/compile/passes/test_mla_attn_quant_fusion.py
tests/v1/attention/test_sparse_mla_backends.py
tests/v1/attention/test_attention_backends.py
tests/v1/attention/test_dspark_noncausal_sparse_mla.py
tests/v1/attention/test_mla_backends.py
tests/v1/e2e/test_replayssm_decode.py
tests/v1/e2e/test_cpu_linear_attn_chunked_prefix.py
tests/v1/simple_kv_offload/test_integration.py
tests/v1/core/test_prefix_caching.py
tests/v1/kv_connector/unit/test_offloading_connector.py
tests/v1/kv_connector/nixl_integration/test_multi_connector_edge_cases.py
tests/evals/gsm8k/configs/Qwen3-4B-TQ-k3v4nc.yaml
tests/evals/gsm8k/configs/Qwen3-4B-TQ-t3nc.yaml
tests/evals/gsm8k/configs/Qwen3-4B-TQ-k8v4.yaml
```

**Question:** Are there any accuracy tests that run WITH these optimizations enabled (not just default config)? If you find one, open it — what does it actually assert?

**Finding:** _(skipped)_


## Section 5: Synthesis — write your conclusions while the evidence is fresh

**Note:** In one paragraph: how does this runtime define and measure PERFORMANCE for agentic-like workloads (multi-turn/shared-prefix)? Name the actual metric(s) and where they're computed.

_(skipped)_

**Note:** In one paragraph: how does this runtime define and measure ACCURACY for agentic-like workloads (tool-calling, multi-turn)? Or does it not, and instead only validates single-turn accuracy + assumes optimizations are 'safe' generically?

_(skipped)_

**Note:** What is the single most surprising GAP you found (something you expected to exist based on the marketing/docs, but couldn't find in code/CI)?

_(skipped)_

**Note:** What is the single most solid PRACTICE you found (something concretely well-built that's worth citing as a good example)?

_(skipped)_

