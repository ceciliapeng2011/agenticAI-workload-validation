# llama.cpp Agentic Workload 性能 / 精度评估方法 — 调查报告

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · **llama.cpp** · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 
> 仓库路径：`~/agenticAI/llama.cpp`
> 方法：与 vLLM / SGLang / TensorRT-LLM 调查一致——不依赖关键词命中数量,逐个打开源码读实现逻辑
> 说明：llama.cpp 定位与前三者不同——它是单机/边缘推理引擎,不是多租户 serving 系统,这一点会直接决定它"该不该有"agentic serving 评测这件事本身,调查时要考虑这个前提。

## 一、Tool-Call 测试:真实模型 + 原生 MCP 支持,但最有价值的测试被排除在日常 CI 之外

涉及文件:

- `tools/server/tests/unit/test_tool_call.py`
- `tools/server/tests/unit/test_mcp_servers.py`
- `tools/server/tests/unit/test_tools_builtin.py`

### 一处亮点:llama-server 原生支持 MCP 协议(三个云端 runtime 里没有一个的推理服务器自带这个)

```
cd ~/agenticAI/llama.cpp && head -12 tools/server/tests/unit/test_mcp_servers.py
```

```python
"""
Tests for MCP server integration via the /tools endpoint.
Invariants verified:
1. MCP tools appear in /tools listing when configured
2. MCP tools use <server>_<tool> naming
3. MCP tools can be invoked and return correct results
...
6. Warmup populates the tool list at startup
"""
```

`llama-server` 通过 `--mcp-servers-json` 直接对接真实 MCP server(测试用了一个真实的 `mcp_echo_server.py` fixture,不是 mock),自动把 MCP 工具注册进 `/tools` 列表。**这是 vLLM/SGLang/TRT-LLM 的推理服务器本体都没有的能力**——TRT-LLM 的 MCP 支持在独立的 `tensorrt_llm.scaffolding` 编排层,不在 `trtllm-serve` 本体里;llama.cpp 是唯一把 MCP 直接烧进推理服务器的。

### `test_calc_result`:三个云端 runtime 之外,第四种"伪多轮"实现方式,但用的是真实下载的模型

```
cd ~/agenticAI/llama.cpp && sed -n '463,500p' tools/server/tests/unit/test_tool_call.py
```

```python
def do_test_calc_result(server, result_override, n_predict, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "system", "content": "You are a tools-calling assistant..."},
            {"role": "user", "content": "What's the y coordinate of a point on the unit sphere at angle 30 degrees?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_6789",
                "function": {"name": "calculate", "arguments": "{\"expression\":\"sin(30 * pi / 180)\"}"}}]},
            {"role": "tool", "name": "calculate", "content": "0.55644242476", "tool_call_id": "call_6789"},
        ],
        "tools": [...],
    })
    content = body["choices"][0]["message"]["content"]
    assert re.match('^[\\s\\S]*?...\\b0\\.(5\\b|56\\b|556)', content)
```

和 vLLM 的模式一样:**历史是写死的 fixture,不是真实推理出来的第一轮**。区别在于 llama.cpp 这个测试是跨多个真实 HuggingFace 模型(`hf_repo` 参数化,如不同的 GGUF 量化版本、不同 chat template)跑的真实推理,不是 mock,验证的是"给定正确的工具结果,模型能不能算出并说出正确答案(0.56)"——这是三次调查以来第一次看到工具调用测试直接断言一个**具体数值结果**而不只是"格式对不对"。

### 核心发现:这个测试被标记为 `slow`,PR 门禁默认跳过

```
cd ~/agenticAI/llama.cpp && grep -n "slow_tests\|pytest -v -x" .github/workflows/server.yml
```

```yaml
pytest -v -x -m "not slow"          # 常规 PR 触发,排除 slow
...
if: ${{ github.event.schedule || github.event.inputs.slow_tests == 'true' }}
SLOW_TESTS=1 pytest -v -x           # 只有 定时任务 或 手动触发 slow_tests=true 才跑全部
```

`test_calc_result` 标了 `@pytest.mark.slow`(因为要真下载真实模型跑推理,比纯 mock 测试慢得多)。**推论:这个仓库里"最像真实工具调用场景"的测试,不在每次 PR 都会跑的门禁里,只在 schedule 触发(大概率是 nightly)或手动开关时跑。** 和 vLLM(fixture 伪多轮但至少在 PR 门禁里)、SGLang(真两轮但整个类被注释掉从不跑)、TRT-LLM(真两轮且确实在 L0 里跑)对比,llama.cpp 是"测试写得最真实(真模型），但触发频率最低"的一档。

## 二、性能基准:官方旗舰 Benchmark CI 已停用近两年,`llama-bench` 是三个云端 runtime 里最原始的单机微基准

### 关键发现:唯一一个专门的性能 CI 工作流,处于停用状态

```
cd ~/agenticAI/llama.cpp && head -8 .github/workflows/bench.yml.disabled
```

```yaml
# TODO: there have been some issues with the workflow, so disabling for now
#       https://github.com/ggml-org/llama.cpp/issues/7893
name: Benchmark
on:
  push:
    branches: [master]
    paths: ['llama.cpp', 'ggml.c', 'ggml-backend.cpp', 'ggml-quants.c', '**/*.cu', 'tools/server/*.h*', 'tools/server/*.cpp']
```

文件名后缀 `.yml.disabled`——GitHub Actions 不会识别、不会执行这个文件。它原本设计成:**在 `server.cpp`/`ggml` 核心文件变更时,自动在 Azure GPU 上跑一轮性能基准**。这正是本该覆盖"性能有没有回归"的旗舰工作流,但对应的 issue #7893 至今没有解决方案合并回来重新启用它。**这意味着 llama.cpp 目前没有任何自动化的、持续追踪的 serving 性能回归 CI**——这一点比 vLLM/SGLang/TRT-LLM(至少有"单轮合成负载"的性能回归)更彻底缺失。

### `llama-bench`:不是 serving 基准,是纯粹的算子级微基准

```
cd ~/agenticAI/llama.cpp && grep -n '"\-\-[a-z-]*"' tools/llama-bench/llama-bench.cpp | grep -oE '"--[a-z-]*"' | sort -u
```

```
--batch-size --cache-type-k --cache-type-v --cpu-mask --delay --device
--flash-attn --n-cpu-moe --n-depth --n-gen --n-gpu-layers --n-prompt
--no-kv-offload --numa --repetitions ...
```

**没有 `--concurrency`、没有 `--dataset`、没有 session/多轮相关参数。** `llama-bench` 测的是"给定 prompt 长度(`--n-prompt`)和生成长度(`--n-gen`),在给定硬件配置下跑多快",是单请求、单会话、离线批处理式的微基准,连 vLLM/SGLang 那种"至少支持并发+ShareGPT"的服务器级压测都够不上,更不用说多轮/agentic 建模。这是四个 runtime(含 Ollama)里最原始的性能测试工具。

## 三、Prefix Cache / Session 正确性:有真实的行为级验证,是四个系统里少见的"检查观测到的效果"而非"检查内部数据结构"

```
cd ~/agenticAI/llama.cpp && grep -n "assert" tools/server/tests/unit/test_kv_keep_only_active.py | tail -10
```

```python
assert "updating prompt cache" in log.drain()
assert res.body["timings"]["cache_n"] > 0
assert res.body["timings"]["prompt_n"] < original_prompt_n     # ← 真实验证了 prefill token 数因为缓存命中而变少
```

`test_clear_and_restore`(在 `test_kv_keep_only_active.py` 里)不是像 vLLM `test_prefix_caching.py`、SGLang `test_radix_cache_slru_accuracy.py` 那样只测内部 block/hash 数据结构对不对,而是**直接断言一个可观测的效果**:命中缓存后,服务端真实汇报的 `prompt_n`(需要重新计算的 token 数)确实变少了。这是四个系统里我看到的、**唯一一处从"用户可见的性能指标"角度验证前缀缓存有没有真的生效**的测试,而不是停留在"内部逻辑跑通了"这一层。

但同样:这仍然是纯机制正确性验证,**没有和任何工具调用/精度断言结合**——没有一处测试是"前缀命中之后,模型的工具调用输出还对不对"。

## 四、精度评估方法论:整个仓库唯一的"精度"概念是困惑度(Perplexity)+ KL 散度,纯统计、非任务型、且完全手动

```
cd ~/agenticAI/llama.cpp && sed -n '1,15p' tools/perplexity/README.md
```

```
Perplexity measures how well the model can predict the next token with lower values being better.
Within llama.cpp the perplexity of base models is used primarily to judge the quality loss
from e.g. quantized models vs. FP16.
The convention among contributors is to use the Wikitext-2 test set...
```

配合 `--kl-divergence`:

```
In addition to the KL divergence the following statistics are calculated:
* Ratio of mean FP16 PPL and quantized PPL.
* Mean change in "correct" token probability.
* Pearson correlation coefficient of the "correct" token probabilities between models.
```

这是 llama.cpp 社区评估"某个新量化方案有没有明显伤害模型质量"的**唯一、事实标准的方法论**——FP16 模型跑一遍 Wikitext-2,记录逐 token 的 logit 分布,量化后的模型再跑一遍,比较 KL 散度/困惑度比值。

### 核心结论:这套方法论从未接入 CI,而且从设计上就与"任务正确性"无关

```
cd ~/agenticAI/llama.cpp && grep -rl "perplexity" .github/workflows/*.yml
```

**零命中。** perplexity/KL 散度评测**完全是人工触发的**——贡献者提交一个新量化格式的 PR 时,约定俗成地在 PR 描述里贴一份自己手跑的困惑度对比表,没有任何自动化门禁强制要求或验证这件事。

更关键的是这套方法论**在概念上就不衡量任务正确性**:困惑度衡量"预测下一个 token 的能力",不衡量"这次工具调用的参数填对了没有""这个 JSON 输出合不合法""这个多轮任务最后完成了没有"。README 里那句话说得很直白:"finetunes 通常会导致更高的困惑度,即使人类评价的输出质量其实更好"——这说明社区自己也承认困惑度和实际任务表现可能是脱节的,但目前没有替代方案。**换句话说:llama.cpp 生态里,"KV cache 量化/权重量化对 agent 工具调用能力的影响"这个问题,不存在任何测量手段——连非自动化的、人工的手段都没有,因为唯一的精度工具(perplexity)从设计上就回答不了这个问题。**

## 五、综合结论

### 性能侧

1. 官方旗舰性能 CI(`bench.yml`)自 2024 年某个 issue 后被停用至今,**没有任何自动化、持续追踪的 serving 性能回归**——比另外三家(至少有单轮合成负载回归)更彻底;
2. `llama-bench` 是单请求、单会话的算子级微基准,不支持并发、数据集、多轮——四个系统(含 Ollama)里最原始;
3. Prefix cache 的功能测试(`test_kv_keep_only_active.py`)做到了"验证可观测效果"(`prompt_n` 真的变少),比另外三家停留在内部数据结构层面的对应测试更扎实,但同样不涉及并发/多会话/agentic 场景。

### 精度侧

1. `llama-server` 原生支持 MCP 协议,是四个系统里唯一把 MCP 直接内建在推理服务本体(而非上层编排框架)的;
2. 工具调用测试(`test_calc_result`)是三次调查里第一个断言具体数值结果、且用真实下载模型跑的测试,但和 vLLM 一样是"写死历史"的伪多轮,而且被标记 `slow`,**默认 PR 门禁不跑,只在 schedule/手动触发时执行**——是"写得最真实,却跑得最少"的一个案例;
3. 全仓库唯一的精度评估方法论是困惑度 + KL 散度,**从未接入 CI(零命中)、纯人工触发、且在设计上完全无法回答"量化/缓存优化是否损伤工具调用/agentic 任务能力"这个问题**——这是四个系统里精度评估方法论与 agentic 需求最脱节的一个,不是"没做全",而是"用的工具从根上就答不了这个问题"。

### 与前三家、及后续 Ollama 报告的关系

llama.cpp 是 Ollama 的底层引擎(见 Ollama 报告),它在这里呈现出的所有特征——最原始的性能压测工具、无自动化精度回归、精度概念仅限于困惑度——会直接构成 Ollama 报告里"性能/精度评估能力"的**下限**,因为 Ollama 自己在这两方面并没有在 llama.cpp 的基础上做额外补强(详见 Ollama 报告第二、五节)。
