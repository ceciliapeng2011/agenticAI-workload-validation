# vLLM Agentic Workload 性能 / 精度评估方法 — 调查报告

> 
> 仓库路径：`~/agenticAI/vllm`
> 说明：当前代码检出版本晚于前期 1/2 批次脚本执行版本，部分 CI 文件路径已重组

## 一、Tool-Call 精度测试：`tests/tool_use/`

涉及文件：

- `tests/tool_use/test_tool_calls.py`
- `tests/tool_use/utils.py`

### 测试用例说明

1. **`test_tool_call_and_choice`**
单次请求测试，给定 `WEATHER_TOOL` / `SEARCH_TOOL` Schema，校验模型能够正确生成单次工具调用；校验项包含 `function.name`、可解析的 Arguments JSON、参数精确匹配（如 `"Dallas"/"TX"`）。

> 
> 采用**精确断言**，而非相似度打分。

2. **`test_tool_call_with_results`**
内置固定三消息对话 Fixture：`user → assistant.tool_calls → role:"tool"` 返回结果，将整段历史一次性送入模型，校验最终输出包含关键词 `98`。

### 核心重大发现

> 
> 该 “多轮” 测试**并非真实多轮循环**：不会先推理第 1 轮拿到`tool_call`，再回填工具结果执行第 2 轮推理。
> 本质：把预定义完整对话历史作为静态输入，仅验证**单次续写结果**。
> vLLM 当前 CI**没有验证真实多轮推理链路、轮次间 KV Cache 演化行为**。

### Prefix Caching 关键结论

执行检索命令：

```
cd ~/agenticAI/vllm && grep -c -- "--no-enable-prefix-caching" tests/tool_use/utils.py
```

输出：`11`

源码信息：
`tests/tool_use/utils.py` 共定义 **12 套模型测试配置**（Hermes、Llama、Mistral 等，配套不同工具调用解析器），**其中 11 套显式关闭前缀缓存**。
示例片段：

```
"llama": {
    "arguments": [
        "--enforce-eager",
        "--no-enable-prefix-caching",   // 显式关闭前缀缓存
        "--tool-call-parser", "llama3_json",
        ...
    ]
}
```

✅ 推论：
vLLM 所有工具调用 CI 断言均在**前缀缓存关闭**条件下运行。
**不存在任何自动化 CI 验证：开启 Prefix Caching 后，工具调用准确率是否保持稳定。**

## 二、性能基准：官方文档基准 vs 仓库高级测试工具（两者能力脱节）

基准文档路径：`.buildkite/performance-benchmarks/performance-benchmarks-descriptions.md`

### 官方 CI 标准性能测试范围

```
cd ~/agenticAI/vllm && sed -n '1,35p' .buildkite/performance-benchmarks/performance-benchmarks-descriptions.md
```

```
# Performance benchmarks descriptions

## Latency tests
- Input length: 32 tokens.
- Output length: 128 tokens.
- Batch size: fixed (8).
- GPU/HPU Models: llama-3.1 8B, llama-3 70B, mixtral 8x7B.
- CPU Models: llama-3.1 8B.
- Evaluation metrics: end-to-end latency (mean, median, p99).

{latency_tests_markdown_table}

## Throughput tests
- Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm to achieve maximum throughput.
- GPU/HPU Models: llama-3.1 8B, llama-3 70B, mixtral 8x7B.
- CPU Models: llama-3.1 8B.
- Evaluation metrics: throughput.

{throughput_tests_markdown_table}

## Serving tests
- Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm and the arrival pattern of the requests.
- **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
- GPU/HPU Models: llama-3.1 8B, llama-3 70B, mixtral 8x7B.
- We also added a speculative decoding test for llama-3 70B on GPU, under QPS 2
- CPU Models: llama-3.1 8B.
- Evaluation metrics: throughput, TTFT (time to the first token, with mean, median and p99), ITL (inter-token latency, with mean, median and p99).
- For CPU, we added random dataset tests to benchmark fixed input/output length with 100 prompts.
```

### 官方 CI 基准局限

全部测试为**单轮独立请求**：基于 ShareGPT 采样、固定输出长度、泊松请求到达；
❌ 无会话建模、无多轮上下文持续增长、无前缀复用场景；
指标仅包含标准指标：e2e latency /throughput/ TTFT / ITL；
❌ **缺少任务级指标、Cache 命中率指标**。

### 仓库内置高级多轮压测工具：两套并存，各有侧重

vLLM 有**两套独立的多轮压测器**，能力互补。

#### 第一套：Python 版 `benchmarks/multi_turn/benchmark_serving_multi_turn.py`

配套文件 `bench_dataset.py`、`convert_sharegpt_to_openai.py`、`generate_multi_turn.json`；README 标题即 *"Benchmark KV Cache Offloading with Multi-Turn Conversations"*。关键参数：

```
--num-clients                "Number of clients that will send requests in parallel"
--max-active-conversations   "Max number of active conversations at a time (for all clients)"
--send-conversation-id       会话亲和路由
--verify-output              校验输出
--conversation-sampling / --max-turns / --request-rate / --warmup-percentages
```

报告的指标：

```
ttft_ms / tpot_ms / latency_ms
moving_avg_ttft_ms / moving_avg_tpot_ms    ← 滑动均值，可观测缓存填满/驱逐引起的时延漂移（独有）
approx_cached_percent                      ← = history_num_tokens / input_num_tokens，客户端近似估算
input_num_tokens / output_num_tokens / output_num_chunks / conversation_id / client_id
```

**这是五个系统里对"N 个并发会话、各自维护独立 KV 状态"建模最完整的工具**——`--num-clients` × `--max-active-conversations` 的组合恰好是多 Agent 并发场景的负载形状（多个 Agent 各跑自己的会话、共同争抢 KV cache）。原始数据按 `conversation_id + start_time_ms` 排序，保留了按会话/轮次做二次分析的入口。

需要注意 `approx_cached_percent` 的语义：它是**客户端侧的近似估算**（假设全部历史都命中缓存），不是服务端上报的真实命中率——实际历史可能早已被驱逐。方向可用，但不宜作为判据。

#### 第二套：Rust 版 `rust/src/bench/`

`vllm-bench`，`vllm bench serve` 的 Rust 重构版本，支持真实多轮会话建模：

```
Turn 1: send [user_1], get assistant_1
Turn 2: send [user_1, assistant_1, user_2], get assistant_2
Turn N: send full history + user_N — measures growing-context performance
```

能力清单：

- `--multi-turn-delay-ms`：模拟用户思考间隔
- `--multi-turn-prefix-global-ratio` / `--prefix-conversation-ratio`：前缀共享流量建模
- `X-Session-ID` Header：会话亲和路由
- 逐轮指标拆解：per-turn TTFT / TPOT / ITL / E2EL

> 
> 能力高度贴近真实 Agentic 会话负载特征

### 关键结论

查看 CI 执行脚本：`.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`
官方 Nightly/CI 流水线**并未启用上述任何一套多轮测试能力**，持续使用传统单轮 ShareGPT + 泊松压测。

```
cd ~/agenticAI/vllm && grep -rn "multi_turn\|multi-turn" .buildkite/
```

两套多轮压测器在 `.buildkite/` 下**均零引用**。

现状总结：
✅ vLLM 代码库**具备 Agent 感知性能测试能力，且有两套互补实现**（Python 版强在多并发会话建模，Rust 版强在前缀共享比例控制与 per-turn 指标拆解）
❌ 两套能力**都未接入日常 CI 回归流水线**
属于典型：**代码存在能力，工程回归实践未跟进**

补充：仓库集成 NVIDIA GenAI-Perf（`.buildkite/performance-benchmarks/tests/genai-perf-tests.json`），但参数 `genai_perf_input_parameters = {}`，**未开启会话 / 多轮模式**。

## 三、BFCL 的两条独立链路：压测负载源（不打分）与精度评测脚本（不接 CI）

vLLM 对 BFCL 的使用分成两条互不相干的链路，需要分开看：

- **链路 A**：`vllm/benchmarks/datasets/datasets.py::BFCLDataset` —— 把 BFCL 当**性能压测的负载生成器**，不校验工具调用正确性（本节 3.1）
- **链路 B**：`.buildkite/scripts/tool_call/run-bfcl-eval.sh` —— 真实的 **BFCL 工具调用正确性评测**，但未接入任何 CI pipeline（本节 3.2）

### 3.1 链路 A：BFCLDataset 作为压测负载源

文件路径：`vllm/benchmarks/datasets/datasets.py`
关键代码片段

```
cd ~/agenticAI/vllm && sed -n '4650,4680p' vllm/benchmarks/datasets/datasets.py
```

```
            self._load_category(c) for c in categories
        ]
        # Round-robin interleave so that when --disable-shuffle is set,
        # taking the first num_requests rows still yields balanced category
        # coverage. When shuffle is on (the default) this ordering is
        # randomized away, which is fine — the subsequent random sample is
        # already balanced in expectation.
        interleaved: list[dict] = []
        max_len = max((len(rows) for rows in per_category_rows), default=0)
        for i in range(max_len):
            for rows in per_category_rows:
                if i < len(rows):
                    interleaved.append(rows[i])

        if not self.disable_shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(interleaved)

        sampled_requests: list[SampleRequest] = []
        for row in interleaved:
            if len(sampled_requests) >= num_requests:
                break
            question = row.get("question")
            functions = row.get("function")
            if not question or not functions:
                continue
            # BFCL question is list[list[dict]] — outer is turns. Use the
            # first turn only; skip multi-turn categories in this loader.
            if not isinstance(question, list) or not question:
                continue
            first_turn = question[0]
```

### 调研解读

1. 内置 `BFCLDataset`，对接伯克利工具调用榜单数据集；代码注释目标：`producing production-alike tool calling traffic`（生成贴近生产的工具调用流量）
2. BFCL 数据集定位：**性能压测负载生成器**
   - 将 BFCL Function Schema 转为 OpenAI 工具调用格式
   - 仅用于吞吐、延迟压力测试
   - **不存在校验模型输出 tool\_call 与标准答案对比的逻辑**
3. 核心限制注释：
> 
> BFCL question is list \[list \[dict\]\] — outer is turns. Use the first turn only; skip multi-turn categories in this loader.
> BFCL 原生支持多轮工具调用，但是 vLLM 数据集加载器**主动丢弃多轮内容，只使用第一轮对话**。

### 链路 A 检索结论

- `vllm/benchmarks/*.py`：**无任何解析、校验响应 tool\_calls 字段逻辑**；压测链路只关注性能指标，完全不校验工具调用正确性。
- `tests/benchmarks/test_bfcl_dataset.py`：仅校验数据管道、Schema 转换正确性，不评估模型精度。

### 3.2 链路 B：`run-bfcl-eval.sh` —— 真实的 BFCL 精度评测，但不在 CI 里

```
cd ~/agenticAI/vllm && cat .buildkite/scripts/tool_call/run-bfcl-eval.sh
```

```bash
# Run BFCL (Berkeley Function Call Leaderboard) tool-calling correctness
# evaluation against a local vLLM server.
#   BFCL_MODEL          - HF model name (default: openai/gpt-oss-20b)
#   BFCL_TEST_CATEGORY  - BFCL test categories (default: multi_turn)
#   BFCL_API_TYPE       - API type: "chat_completions" or "responses"
#   BFCL_TOOL_CALL_PARSER / BFCL_REASONING_PARSER / BFCL_TP_SIZE ...
TEST_CATEGORY="${BFCL_TEST_CATEGORY:-multi_turn}"
```

这是一套**完整的、对着真实 vLLM server 跑的 BFCL 工具调用正确性评测**：默认测试类别就是 `multi_turn`（BFCL 最难的多轮类别），支持 `chat_completions` 与 `responses` 两种 API 形态，可配置 tool-call parser / reasoning parser / TP size / 温度等。

但两个限制使它不构成 CI 防线：

```
cd ~/agenticAI/vllm && grep -rn "run-bfcl-eval\|tool_call/" .buildkite/*.yaml .buildkite/**/*.yaml
```

1. **`.buildkite/*.yaml` 里零引用**——没有任何 pipeline 调用这个脚本，它不属于任何 CI job
2. **脚本内没有精度阈值或 pass/fail 判定**——全文只有一处 `exit 1`，条件是"server 600 秒内没起来"；跑完的 BFCL 分数不与任何基线比较、不构成门禁

### 两条链路合起来看

vLLM 不只是"手里有零件"，而是连"BFCL 多轮工具调用精度评测"这个成品都已经做出来了，却依然没有接入自动化回归——**缺的不是能力或认知，而是把已有能力固化成日常门禁的工程决策**。这与第四节那句开发者原话（"建议手动跑 bfcl multi_turn"）完全吻合：那句"建议手动跑"的对象，就是仓库里这个现成的脚本。

## 四、源码一手证据：官方确认缺少 Agentic 精度自动化 CI

文件路径：`tests/entrypoints/openai/chat_completion/test_serving_chat.py`

```
cd ~/agenticAI/vllm && sed -n '1412,1424p' tests/entrypoints/openai/chat_completion/test_serving_chat.py
```

```
class TestServingChatWithHarmony:
    """
    These tests ensure Chat Completion requests are being properly converted into
    Harmony messages and Harmony response messages back into Chat Completion responses.
    These tests are not exhaustive, but each one was created to cover a specific case
    that we got wrong but is now fixed.

    Any changes to the tests and their expectations may result in changes to the
    accuracy of model prompting and responses generated. It is suggested to run
    an evaluation or benchmarking suite (such as bfcl multi_turn) to understand
    any impact of changes in how we prompt Harmony models.
    """
```

### 工程翻译

现有消息渲染测试覆盖不全；
修改对话 Prompt 组装逻辑，可能直接影响多轮工具调用准确率；
**官方建议：开发人员手动运行外部 BFCL multi-turn 评测来验证影响。**

> 
> 直接证据：**多轮 Agent / 工具调用精度变动没有自动化 CI 守护，依赖人工事后外部评测**。

## 五、精度 CI（lm-eval-harness）与 KV 量化测试覆盖现状

执行检索命令：

```
cd ~/agenticAI/vllm && ls .buildkite/lm-eval-harness/configs/ | wc -l; echo "---"; grep -l "kv_cache_dtype" .buildkite/lm-eval-harness/configs/*.yaml; echo "---task types used---"; grep -h "^task:\|task_name\|model_name" .buildkite/lm-eval-harness/configs/*.yaml 2>/dev/null | sort -u | head -10; grep -rh "name:" .buildkite/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct.yaml 2>/dev/null
```

输出摘要：

- 评测配置总数：**38 个 yaml**
- 配置中启用 `kv_cache_dtype` FP8：**仅 2 个**
  - `NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`
  - `Qwen3-235B-A22B-Instruct-2507-FP8.yaml`
- 注释说明：开启 FP8 KV Cache 目的是**加速评测执行速度**，不是验证量化对精度的影响。
- 全部评测任务：`gsm8k` 等单轮数学 / 知识问答；**不存在工具调用、多轮对话任务**。

## 六、综合结论

### 性能侧现状

1. 官方 CI 标准性能基准（`.buildkite/performance-benchmarks/`）仅支持**纯单轮、泊松到达 ShareGPT 压测**，指标：latency/throughput/TTFT/ITL；
2. 仓库内置**两套互补的多轮压测器**——Python 版 `benchmarks/multi_turn/`（强在多并发会话建模：`--num-clients` × `--max-active-conversations`、会话亲和、滑动均值时延）与 Rust 版 `rust/src/bench`（强在前缀共享比例控制、per-turn 指标拆解）；BFCL 数据集可生成真实工具调用流量；**以上能力均未接入 CI 日常回归**；
3. 链路 A 的 BFCLDataset 使用时强制截断为单轮，放弃原生多轮场景；
4. 缓存命中指标只有客户端近似值（`approx_cached_percent` = 历史 token / 输入 token，假设全部命中），无服务端真实命中率上报。

### 精度侧现状

1. `tests/tool_use/` 工具调用测试：单轮 / 伪多轮固定历史输入；11/12 模型配置关闭 Prefix Caching；**无法代表开启前缀缓存的生产环境精度表现**；
2. lm-eval-harness 精度回归全部为单轮 QA、数学任务，完全不覆盖 tool-calling、多轮 Agent 场景；
3. KV Cache 量化精度覆盖极低（38 套配置仅 2 套），启用目标为加速评测，不是验证优化副作用；
4. **已存在真实的 BFCL multi-turn 精度评测脚本**（`.buildkite/scripts/tool_call/run-bfcl-eval.sh`），但零 pipeline 引用、零精度阈值判定 → 属于手动工具，不构成 CI 防护；
5. 源码注释官方确认：多轮工具调用准确率变动**无自动化 CI 防护，依赖开发人员手动执行 BFCL 评测**——所指对象正是上述第 4 条那个脚本。

### 一句话总览

vLLM 代码库**已经具备搭建完整 Agentic 负载评测的全部组件，且完成度高于"零件"级别**：两套多轮压测器（含多并发会话建模）、BFCL 负载数据集、**现成可跑的 BFCL multi-turn 精度评测脚本**、量化精度评测框架。
但这些组件相互割裂且均未固化为门禁，**不存在一条端到端 CI 流水线，同时覆盖【多轮会话 + 工具调用 + 主流优化开启 (Prefix Cache / KV 量化)】的自动化性能 + 精度验证**——缺的不是能力，而是把已有能力接入日常回归的工程决策。
当前 Agentic 场景下的精度质量把关高度依赖人工、事后、外部离线评测。