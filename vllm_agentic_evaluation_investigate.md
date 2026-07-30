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

### 仓库内置高级多轮压测工具

路径：`rust/src/bench/`（`vllm-bench`，`vllm bench serve` Rust 重构版本）
支持真实多轮会话建模：

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
官方 Nightly/CI 流水线**并未启用该多轮测试能力**，持续使用传统单轮 ShareGPT + 泊松压测。

现状总结：
✅ vLLM 代码库**具备 Agent 感知性能测试能力**
❌ 该能力仅存在工具集内，**未接入日常 CI 回归流水线**
属于典型：**代码存在能力，工程回归实践未跟进**

补充：仓库集成 NVIDIA GenAI-Perf（`.buildkite/performance-benchmarks/tests/genai-perf-tests.json`），但参数 `genai_perf_input_parameters = {}`，**未开启会话 / 多轮模式**。

## 三、Tool-Calling 负载数据源 BFCL Dataset：仅用于压测，不做精度打分

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

### 全局仓库检索结论

- `vllm/benchmarks/*.py`：**无任何解析、校验响应 tool\_calls 字段逻辑**；压测链路只关注性能指标，完全不校验工具调用正确性。
- `tests/benchmarks/test_bfcl_dataset.py`：仅校验数据管道、Schema 转换正确性，不评估模型精度。

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
2. 仓库内置高级工具 `rust/src/bench`，原生支持多轮会话、上下文持续增长、前缀共享建模；BFCL 数据集可以生成真实工具调用流量；**两项能力均未接入 CI 日常回归**；
3. BFCL 数据集使用时强制截断为单轮，放弃原生多轮场景。

### 精度侧现状

1. `tests/tool_use/` 工具调用测试：单轮 / 伪多轮固定历史输入；11/12 模型配置关闭 Prefix Caching；**无法代表开启前缀缓存的生产环境精度表现**；
2. lm-eval-harness 精度回归全部为单轮 QA、数学任务，完全不覆盖 tool-calling、多轮 Agent 场景；
3. KV Cache 量化精度覆盖极低（38 套配置仅 2 套），启用目标为加速评测，不是验证优化副作用；
4. 源码注释官方确认：多轮工具调用准确率变动**无自动化 CI 防护，依赖开发人员手动执行外部 BFCL 评测**。

### 一句话总览

vLLM 代码库**已经具备搭建完整 Agentic 负载评测的全部组件**：多轮压测工具、BFCL 负载数据集、量化精度评测框架。
但是组件相互割裂，**不存在一条端到端 CI 流水线，同时覆盖【多轮会话 + 工具调用 + 主流优化开启 (Prefix Cache / KV 量化)】的自动化性能 + 精度验证。**
当前 Agentic 场景下的精度质量把关高度依赖人工、事后、外部离线评测。