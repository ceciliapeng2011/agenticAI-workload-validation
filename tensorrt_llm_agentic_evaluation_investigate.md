# TensorRT-LLM Agentic Workload 性能 / 精度评估方法 — 调查报告

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · **TensorRT-LLM** · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 
> 仓库路径：`~/agenticAI/TensorRT-LLM`
> 方法：与 vLLM / SGLang 调查一致——不依赖关键词命中数量,逐个打开源码读实现逻辑,验证"机制是否真的做了该做的事"
> 说明：TRT-LLM 的 CI 体系与前两者不同——主 CI 跑在 Jenkins（`jenkins/L0_Test.groovy`）上,通过 `tests/integration/test_lists/test-db/*.yml` 精选测试清单驱动,而不是 GitHub Actions 全量跑;`.github/workflows/` 里基本只有 bot/label 类工作流。

## 一、Tool-Call 精度测试:两层结构,一层单测、一层端到端,后者比 vLLM/SGLang 更接近真实但覆盖窄

涉及文件:

- `tests/unittest/llmapi/apps/test_tool_parsers.py`(纯解析器单测)
- `tests/unittest/llmapi/apps/_test_openai_tool_call.py`(端到端,单轮)
- `tests/unittest/llmapi/apps/_test_openai_chat_harmony.py`(端到端,真两轮)

### 一个需要先说明的命名约定,以免误判为"死代码"

TRT-LLM 的 `apps/` 目录里大量文件以下划线开头(`_test_openai_*.py`),乍看很像 SGLang 那种"写了但从未被调用"的情况。核实后发现这是**故意设计**,不是遗留:

```
cd ~/agenticAI/TensorRT-LLM && grep -n "def test_openai_tool_call" -A 5 tests/integration/defs/test_e2e.py
```

```python
def test_openai_tool_call(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_tool_call.py")])
```

`test_e2e.py` 里的包装函数会在**独立子进程/venv**里跑这个下划线文件(每个 `_test_*.py` 自己起一个 server,直接被 pytest 默认收集会冲突/太重,所以主动排除自动发现,改由 wrapper 显式调度)。再确认它确实进了 CI 清单:

```
cd ~/agenticAI/TensorRT-LLM && grep -n "openai_tool_call" tests/integration/test_lists/test-db/l0_h100.yml tests/integration/test_lists/test-db/l0_a10.yml
```

```
tests/integration/test_lists/test-db/l0_a10.yml:125:  - test_e2e.py::test_openai_tool_call
```

**确认:这是活跃的 L0(预合并门禁)测试,在 A10 上跑。** ——这一点和 SGLang 的情况(下面会看到)形成直接对照。

### 端到端测试内容:`_test_openai_tool_call.py` 只有单轮

```
cd ~/agenticAI/TensorRT-LLM && grep -n "^async def test_" tests/unittest/llmapi/apps/_test_openai_tool_call.py
```

```
async def test_tool_parser(...)
async def test_tool_parser_streaming(...)
```

两个测试都只验证"给一个问题,模型能不能触发一次正确的工具调用",**没有第二轮**——比 vLLM 的伪多轮 fixture、SGLang 的(死代码)真两轮测试都更"基础":连尝试做多轮验证都没有。

### 但 `_test_openai_chat_harmony.py` 里藏着三者中最扎实的一个真实两轮测试

```
cd ~/agenticAI/TensorRT-LLM && sed -n '94,152p' tests/unittest/llmapi/apps/_test_openai_chat_harmony.py
```

```python
async def test_tool_calls(client: openai.AsyncOpenAI, model: str):
    messages = [{"role": "user", "content": "What is the weather like in SF?"}]
    response = await client.chat.completions.create(
        model=model, messages=messages, tools=[tool_get_current_weather], ...)
    tool_call = response.choices[0].message.tool_calls[0]           # 第 1 轮:真实推理
    answer = get_current_weather(**json.loads(tool_call.function.arguments))
    messages.extend([
        {"role": "assistant", "tool_calls": [tool_call], "reasoning": message.reasoning},
        {"role": "tool", "content": json.dumps(answer), "tool_call_id": tool_call.id},
    ])
    response = await client.chat.completions.create(
        model=model, messages=messages, ...)                        # 第 2 轮:真实推理,依赖第 1 轮真实输出
    assert response.choices[0].message.content
```

核实调用链是活的:

```
cd ~/agenticAI/TensorRT-LLM && grep -rln "test_openai_chat_harmony\b" tests/integration/test_lists/
→ tests/integration/test_lists/test-db/l0_h100.yml
```

### 核心发现

> 
> 这是我在三个 runtime 里读到的**唯一一处"真实两轮推理 + 依赖上一轮真实输出 + 活跃在 L0 CI 里"**的工具调用测试。vLLM 的两轮测试是写死历史的假多轮;SGLang 写对了真两轮实现,但唯一调用点被整个注释掉,从未跑过;TRT-LLM 这一处是**真做了,而且真在跑**。
> 
> 但要注意覆盖面:这个真两轮测试**只存在于 Harmony(gpt-oss)专属的测试文件里**,不是像 vLLM `tests/tool_use/utils.py` 那样横跨 12 个模型配置的通用测试,也不像 SGLang 有覆盖十几个 detector 的单测矩阵。**广度不如另外两者,但这一个个例的深度是三者最高的。**

### Prefix Caching / KV Cache Reuse 状态

```
cd ~/agenticAI/TensorRT-LLM && grep -n "enable_block_reuse" tensorrt_llm/llmapi/llm_args.py | head -3
```

```
enable_block_reuse: bool = Field(default=True, ...)
```

`KvCacheConfig.enable_block_reuse` 默认 `True`。翻查 `_test_openai_tool_call.py`、`_test_openai_chat_harmony.py` 的 server 启动参数,**都没有显式关闭它**。

✅ 推论(与 SGLang 结论一致,与 vLLM 相反):
**TRT-LLM 的工具调用端到端测试是在 KV cache 块复用默认开启的状态下跑的**——包括那个真两轮 Harmony 测试。这意味着"工具调用 + KV cache 复用"这个组合,在 TRT-LLM 这一个具体案例里,是有 CI 覆盖的(虽然只覆盖 Harmony 这一种格式)。

## 二、性能基准:trtllm-bench 原生支持多轮回放,但官方追踪的性能回归数字依然是纯单轮合成负载

### `trtllm-bench` 数据集侧:真支持多轮,来源可以是真实数据集(如 MT-Bench)

```
cd ~/agenticAI/TensorRT-LLM && sed -n '214,230p' tensorrt_llm/bench/dataset/prepare_real_data.py
```

```python
"""
Supports three input modes based on the shape of the data at --dataset-input-key:
1. Single-turn text (default)
2. Multi-turn conversation: value is a list of strings (e.g. MT-Bench ``turns`` field)
3. Multimodal
"""
```

### 执行侧:多轮请求真的按顺序逐轮跑,不是拼接成一个大 prompt

```
cd ~/agenticAI/TensorRT-LLM && sed -n '482,486p' tensorrt_llm/bench/benchmark/throughput.py
```

```python
has_multi_turn = any(r.is_multi_turn for r in requests)
if has_multi_turn:
    logger.info("Multi-turn requests detected. Turns will be processed sequentially within each request.")
```

这个能力是**内置在官方 `trtllm-bench` CLI 主干里**的(不是像 vLLM 那样另起一个没人用的独立 Rust 项目),数据源也可以是真实的 MT-Bench 对话,而不是随机 token。三者里数据源真实感排序:TRT-LLM(可用真实 MT-Bench 对话)> SGLang GSP(结构真实、内容随机)> vLLM(独立工具存在但和主线脱节)。

### 关键结论:能力和官方追踪的性能数字依然是两条线

官方性能回归基线:

```
cd ~/agenticAI/TensorRT-LLM && head -1 tests/integration/defs/perf/base_perf_pytorch.csv
grep -c "multi_turn\|sharegpt\|mooncake" tests/integration/defs/perf/base_perf_pytorch.csv
```

```
network_name,perf_case_name,test_name,threshold,absolute_threshold,metric_type,perf_metric,device_subtype
0
```

命名格式统一是 `input_output_len:128,128`(固定 ISL/OSL 的合成负载)、`bench-pytorch-float16-...-reqs:8192` 这类——**清一色单轮固定长度合成请求,没有一条追踪的性能回归指标用到多轮/MT-Bench/真实数据集**。指标类型倒是比 vLLM/SGLang 更细:`INFERENCE_TIME`、`SEQ_THROUGHPUT`、`TOTAL_OUTPUT_THROUGHPUT`、`KV_CACHE_SIZE`(专门追踪 KV cache 占用大小,这一点 vLLM/SGLang 的官方性能描述文档里都没有)。

现状总结:和 vLLM(rust-bench 有多轮能力,未接入 CI 性能报告)、SGLang(GSP 多轮压测能力接入了功能测试而非性能报告)一样的模式再现一次——**trtllm-bench 的多轮能力存在且比另外两家更原生,但官方追踪、用于判定性能回归的指标体系完全没有使用它**。

## 三、Agentic 负载/Agent 框架:TRT-LLM 是三者中唯一自带"真实 Agent 实现"的,但这些实现活在 `examples/`,精度产出未被日常 CI 追踪

这是与 vLLM、SGLang 最大的结构性差异——vLLM/SGLang 的"agentic benchmark"都只是"回放固定 trace 测吞吐"(SGLang 的 `benchmark/react/README.md` 甚至主动声明"这不是真实 agent")。TRT-LLM 则有一整套**真实可运行的 Agent 编排框架**:

```
cd ~/agenticAI/TensorRT-LLM && find examples/scaffolding -maxdepth 2 -type d
```

```
examples/scaffolding/mcp/{tavily_search,coder,wordllama,google_search,e2b,weather,fetch_webpage,google_scholar}
examples/scaffolding/trace_replay
examples/scaffolding/contrib/{iter_research,DeepConf,open_deep_research,TreeInference,Coder,AsyncGeneration,tree_of_thought_research,Dynasor}
```

`contrib/Coder` 是一个**真实的 SWE-bench 编码 agent 实现**:

```
cd ~/agenticAI/TensorRT-LLM && grep -n "run_swebench\|preds.json\|traj.json" examples/scaffolding/contrib/Coder/README.md | head -6
```

```
python examples/scaffolding/contrib/Coder/run_swebench.py ...
swebench_output/<dataset>-<split>-<model>-<time>/preds.json          # 真实 SWE-bench 评分用预测文件
swebench_output/<dataset>-<split>-<model>-<time>/<instance_id>.traj.json  # 每个任务的真实轨迹
```

更进一步,`trace_replay/analysis/compute_cache_hit_trace.py` 会**用真实 agent 跑出来的 trace,反推理论上限的前缀缓存命中率**:

```python
r"""CLI: compute the ideal prefix KV-cache hit upper bound for scaffolding traces.
Typical usage:
    python .../compute_cache_hit_trace.py TRACES/swebench-verified/coder/astropy__astropy-7166
"""
```

这正是我在这次系列调查最开始建议的方法论——"自己造一份真实 agent trace,再去分析前缀复用结构"——**TRT-LLM 官方已经把这套工具做出来了,并且是针对 SWE-bench-verified 这样的真实编码 agent 负载**。

### 但要澄清:这些都在 `examples/` 下,和"日常 CI 精度回归"是两回事

```
cd ~/agenticAI/TensorRT-LLM && grep -rl "scaffolding" tests/integration/test_lists/
```

```
tests/integration/test_lists/test-db/l0_l40s.yml
tests/integration/test_lists/test-db/l0_h100.yml
tests/integration/test_lists/qa/llm_function_core.txt
```

命中的是 `tensorrt_llm/scaffolding/`(主代码库里的编排框架模块本身),**不是** `examples/scaffolding/contrib/Coder` 的 SWE-bench 跑分。也就是说:**"Scaffolding 框架能不能正常工作"有 CI 覆盖,但"用 Scaffolding 跑 SWE-bench 能拿多少分、这个分数会不会因为某次 PR 而回退"——没有任何自动化流水线在追踪**。`run_swebench.py` 是留给用户手动跑的工具,不是回归门禁。

## 四、一手证据:三者里"最像真实 agent 循环"的那个测试,亲手打开后发现 LLM 是假的

```
cd ~/agenticAI/TensorRT-LLM && grep -n "class DummyWorker" -A 25 tests/unittest/scaffolding/test_mcp_worker.py
```

`tests/unittest/scaffolding/test_mcp_worker.py::test_scaffolding_with_chat_mcp_controller` 起了一个**真实的 MCP server**(用官方 `mcp` SDK 的 `FastMCP`),注册了真实工具(`add_numbers`、`echo_message`),用真实的 `ChatWithMCPController`(`max_iterations=5`)编排一个"提问 → 决定调用工具 → 拿到工具结果 → 给出最终答案"的完整 agent 循环——看起来是三个 runtime 里唯一一个测试"完整 agent 循环"的用例。

但生成侧用的是:

```python
class DummyWorker(Worker):
    async def dummy_handler(self, task: ChatTask):
        if len(task.messages) == 2:
            task.add_message(AssistantMessage(
                content="call add_numbers(5, 3)",
                tool_calls=[ToolCall("add_numbers", '{"a": 5, "b": 3}')]))
            task.finish_reason = "tool_calls"
        elif len(task.messages) == 4:
            ...
        else:
            task.add_message(AssistantMessage(content="Hello MCP!"))
            task.finish_reason = "stop"
```

**`DummyWorker` 完全按对话轮数(`len(task.messages)`)硬编码返回值,从头到尾没有调用任何真实模型。** 这个测试验证的是`ScaffoldingLlm`/`ChatWithMCPController` 的**编排逻辑**(是不是正确地在第 2 轮触发工具调用、第 4 轮触发第二次工具调用、第 6 轮收尾)——和 SGLang 的 tool_parser 单测、vLLM 的 structured_output 单测本质上是同一类"测试机制,不测真实模型"的模式,只是伪装得更像一个端到端场景。

> 
> **推论:三个 runtime 里,凡是"看起来像完整 agent 循环"的测试,打开生成侧的实现之后,几乎必然会发现是脚本/fixture/dummy 在扮演 LLM,而不是真实推理。** 这是这次系列调查里复现次数最多的一条规律,值得作为审计任何 runtime 时的默认怀疑起点。

## 五、精度 CI 与 KV 量化/缓存复用组合覆盖:三者中最系统化的一处

### 规模

```
cd ~/agenticAI/TensorRT-LLM && grep -c "def test_" tests/integration/defs/accuracy/test_llm_api_pytorch.py
grep -c "enable_block_reuse=True" tests/integration/defs/accuracy/test_llm_api_pytorch.py
grep -c "enable_block_reuse=False" tests/integration/defs/accuracy/test_llm_api_pytorch.py
```

```
224   # 测试函数总数(远超 vLLM 38 个 lm-eval 配置、SGLang 4 个配置)
21    # 显式开启 block reuse 且随后跑精度评测的测试数
59    # 显式关闭 block reuse 的测试数
```

### 关键发现:vLLM/SGLang 都缺失的"量化 × 前缀复用 × 精度"三者同框,这里真的做了

```
cd ~/agenticAI/TensorRT-LLM && sed -n '1285,1300p' tests/integration/defs/accuracy/test_llm_api_pytorch.py
```

```python
def test_fp8_vswa_reuse(self):
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, max_attention_window=[...])
    prequantized_model_path = f"{llm_models_root()}/gemma/gemma-3-1b-it-fp8/"
    with LLM(prequantized_model_path, kv_cache_config=kv_cache_config) as llm:
        task = GSM8K(self.MODEL_NAME); task.evaluate(llm)
        task = MMLU(self.MODEL_NAME); task.evaluate(llm)

@pytest.mark.parametrize("backend", ["xgrammar"])
def test_fp8_guided_decoding_vswa_reuse(self, backend, mocker):
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, max_attention_window=[...])
    llm = LLM(prequantized_model_path, guided_decoding_backend=backend,
              kv_cache_config=kv_cache_config, ...)
    task = JsonModeEval(self.MODEL_NAME); task.evaluate(llm)
```

`test_fp8_guided_decoding_vswa_reuse` 一次性把三件事放在一起测:**FP8 权重量化 + KV block reuse 开启 + xgrammar 约束解码(工具调用依赖的结构化输出机制)+ JSON 格式正确性精度评测**。另外还发现了刻意做的 A/B 对照:

```
test_auto_dtype_vswa_without_reuse()   # enable_block_reuse=False, 精度基线
test_auto_dtype_vswa_reuse()           # enable_block_reuse=True,  同一模型同一任务
```

这正是 vLLM(`--no-enable-prefix-caching` 写死关闭)、SGLang(`--disable-radix-cache` 写死关闭)两边都没做到的事——**TRT-LLM 至少在一部分测试矩阵里,把"缓存复用开/关"当成一个显式的精度测试变量,而不是关掉了事**。

### 另一处真实的"优化不改变输出"验证:KV Pool 重平衡

```
cd ~/agenticAI/TensorRT-LLM && head -22 tests/integration/defs/accuracy/test_kv_pool_rebalance_accuracy.py
```

```python
r"""Accuracy test for the KVCacheManagerV2 rebalance hook.
Verifies that forcing the V2 auto-tuner to fire mid-generation does not
change greedy-decode outputs.
"""
```

这个测试比 vLLM 的 `gsm8k_offloading.py`(靠准确率阈值间接判断有没有腐化)更严格——它直接断言**贪婪解码输出逐字节不变**,不是"准确率没掉太多"这种有容差的代理指标。是三次调查里见过最硬核的一处"优化不该改变输出"验证。

### 但这一切仍然没有覆盖到"多轮 + 工具调用"

上面所有精度测试用的任务集是 `GSM8K` / `MMLU` / `CnnDailymail` / `JsonModeEval` / `GPQADiamond`——**全部单轮**。`JsonModeEval` 虽然和工具调用共享同一套约束解码机制,但它评的是"JSON 格式合不合规",不是"工具调用参数对不对、该不该调这个工具"。

```
cd ~/agenticAI/TensorRT-LLM && grep -rniE "bfcl|gorilla|tau.?bench|agentbench|toolbench" --include="*.py" --include="*.md" . | grep -v "\.git/"
```

除了 examples 里的 SWE-bench(第三节已说明,和日常精度 CI 无关)外,**零命中**——和 vLLM、SGLang 结论一致:没有一家把标准 agentic 精度基准接入自身精度回归体系。

## 六、综合结论:三者对照

### 性能侧

1. `trtllm-bench` 原生支持真实多轮会话回放(可用 MT-Bench 等真实数据集,按轮顺序真实调用),内置于官方 CLI 主干,不像 vLLM 那样是脱节的独立工具;
2. 但官方追踪、用于判定 PR 是否引入性能回归的指标体系(`base_perf_pytorch.csv` 及其背后的 `test_perf.py`)完全是固定 ISL/OSL 的单轮合成负载,和 vLLM/SGLang 一样,**多轮能力与官方性能数字是两条不相交的线**;
3. 独有 `KV_CACHE_SIZE` 作为官方追踪指标之一,这一点比另外两家更细致。

### 精度侧

1. **工具调用端到端测试**:`_test_openai_tool_call.py` 只有单轮,但 `_test_openai_chat_harmony.py::test_tool_calls` 是三个 runtime 里**唯一一处真两轮(依赖上一轮真实推理输出)且确认活跃在 L0 CI 里**的工具调用测试——覆盖面窄(仅 Harmony 格式),但深度最高;
2. **KV cache 复用状态**:默认开启,工具调用测试未显式关闭——与 SGLang 一致,和 vLLM(11/12 配置显式关闭)相反;
3. **量化/复用/精度三者同框**:`test_llm_api_pytorch.py` 224 个测试里有 21 处显式开启 block reuse 并跑精度评测,包含 FP8 量化+复用+guided decoding+精度 的组合测试,以及 reuse/without_reuse 的显式 A/B 对照——这是三次调查里**唯一**做到"缓存复用作为精度测试的显式变量"而非"直接关掉"的案例;
4. 但精度任务集依然清一色单轮(GSM8K/MMLU/JsonModeEval/CnnDailymail),**没有任何标准工具调用/多轮 agentic 精度基准(BFCL/τ-bench 等)被接入**;
5. **独有能力**:`examples/scaffolding` 下有真实可跑的 SWE-bench 编码 agent(含真实轨迹、真实预测文件、真实前缀缓存命中率分析工具)——这是三个 runtime 里唯一的"官方自带真实 Agent 实现"案例;但这套东西的产出(SWE-bench 分数、缓存命中率)**没有被任何日常 CI 流水线追踪**,纯粹是留给用户手动跑的工具;
6. **复现的规律**:三个 runtime 里,任何"看起来像完整 agent 循环"的测试,打开生成侧实现后几乎必然发现是 fixture/脚本/DummyWorker 在扮演 LLM——TRT-LLM 的 `test_scaffolding_with_chat_mcp_controller`(看似最像真实 agent 循环的测试)也不例外。

### 一句话总览

TensorRT-LLM 在"广度"上不如 vLLM(工具调用模型覆盖少)和 SGLang(压根没有独立 agent 框架),但在"深度"上三次调查里最强:唯一有真正活跃、真两轮依赖的工具调用 CI 测试;唯一把"缓存复用"当成精度测试的显式实验变量而不是一关了之;唯一自带可运行的真实 Agent 框架(Scaffolding + SWE-bench Coder)和配套的前缀缓存命中率分析工具。

但最终结论和另外两家一样:**没有一条端到端 CI 流水线把"多轮会话 + 工具调用 + 主流优化组合开启"作为一个整体去做自动化精度回归**——TRT-LLM 的护城河在于"分别把这几件事的一部分做得比另外两家扎实",而不是"把它们真正拼在一起测过"。真实 Agent 能力(Scaffolding/SWE-bench)存在,但产出的精度分数目前活在 `examples/` 里,靠人工触发,没有被日常工程实践当成一等公民去回归追踪——这一点和 vLLM 那句开发者原话("建议手动跑 BFCL multi_turn")所描述的现状,本质上是同一件事。
