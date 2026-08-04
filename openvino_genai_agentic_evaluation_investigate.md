# OpenVINO GenAI Agentic Workload 性能 / 精度评估方法 — 调查报告

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · **OpenVINO GenAI**
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 
> 仓库路径：`~/openvino.genai`
> 方法：与 vLLM / SGLang / TensorRT-LLM / llama.cpp / Ollama 调查一致——不依赖关键词命中数量,逐个打开源码读实现逻辑
> 说明：OpenVINO GenAI 定位与前五者又不同——它是 Intel 硬件（CPU/GPU/NPU）上的推理引擎 + Python/C++/JS 多语言 SDK,主 CI 跑在 GitHub Actions（`linux.yml`/`windows.yml`/`mac.yml`）而不是前面几家常见的 Jenkins/Buildkite。它自带的精度工具 WWB（who_what_benchmark）和另外五家的对应物都不太一样,调查时要单独理解它的方法论。

## 一、Tool-Call 测试:三层结构——底层解析器单测、跨语言一致性测试、真实推理但不测任务正确性

涉及文件:

- `tests/python_tests/test_parsers.py` / `tests/python_tests/test_vllm_parsers_wrapper.py`
- `tests/python_tests/samples/test_react_sample.py`
- `src/cpp/include/openvino/genai/parsers.hpp`(已知：`Llama3JsonToolParser`、`Llama3PythonicToolParser`、各家 `ReasoningParser`）

### 一个值得记录的工程决策:没有重新发明轮子,直接复用了 vLLM 的 tool-call parser

```
cd ~/openvino.genai && head -12 tests/python_tests/test_vllm_parsers_wrapper.py
```

```python
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openvino_genai import (
    Tokenizer, VLLMParserWrapper, TextParserStreamer,
)
```

OpenVINO GenAI 提供了 `VLLMParserWrapper`,直接包装/复用 vLLM 生态里已经维护的各家模型 tool-call parser,而不是像 vLLM/SGLang/TRT-LLM 各自独立维护一套 per-model parser。这是五个 runtime 之外的第六个系统里,唯一一处"不重复造轮子,直接对接社区已有解析器生态"的设计决策——测试文件本身的版权头都保留了"Copyright contributors to the vLLM project"。

### `test_parsers.py`:真实代码,但喂的是手工构造的增量 delta,不是真实模型输出

```
cd ~/openvino.genai && sed -n '141,175p' tests/python_tests/test_parsers.py
```

```python
def test_stop_invoked_by_tool_call(hf_ov_genai_models):
    class IncrementalToolParser(IncrementalParser):
        def parse(self, delta_msg, delta_text, delta_tokens=None) -> str:
            if delta_text == "{" and not self.started_tool_call:
                self.started_tool_call = True
                ...
```

测试直接手工喂 `delta_text == "{"`、`"}"` 这类构造好的增量片段给解析器,验证流式解析状态机对不对——这是本系列六次调查里第 N 次确认的"解析器单测测的是解析逻辑,不是模型行为"模式,和 vLLM 的 `tests/tool_parsers/`、SGLang 的 `test/registered/unit/function_call/` 是同一类东西。

### `test_react_sample_refs`:三个 marker 里最特殊的 `@pytest.mark.agent`,但验证的是跨语言一致性,不是任务完成度

```
cd ~/openvino.genai && cat tests/python_tests/samples/test_react_sample.py
```

```python
class TestReactSample:
    @pytest.mark.llm
    @pytest.mark.agent
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
    def test_react_sample_refs(self, request, convert_model):
        py_result = run_sample([sys.executable, "text_generation/react_sample.py", convert_model])
        js_result = run_sample(['node', "text_generation/react_sample.js", convert_model])
        assert py_result.stdout == js_result.stdout, f"Results should match"
```

**这是六个系统里唯一带专属 `agent` pytest marker 的测试**,而且是**真实推理**（真的转换 TinyLlama 并跑 ReAct 样例脚本，不是 mock)。但断言目标是"Python 版和 JS 版的输出逐字节一致"——即**跨语言实现一致性回归**,不是"这个 ReAct 循环有没有把 `hotpotqa` 那类问题真的答对"。TinyLlama-1.1B 本身能力有限,大概率也答不对复杂的多跳问题,这个测试设计上就没打算验证任务正确性,只打算验证"两种语言绑定的行为一致"。

## 二、性能基准:`llm_bench` 是单/批量 prompt 微基准,和 llama-bench / ollama bench 同一档,没有找到官方性能回归 CI

```
cd ~/openvino.genai && grep -n "add_argument" tools/llm_bench/benchmark.py | grep -oE '"-[a-zA-Z-]*"' | sort -u
```

```
"-bs" "-d" "-od" "-p" "-r" "-rj"
```

`llm_bench` 的核心参数是 `-p/--prompt`(单个/固定 prompt 集合)、`-bs/--batch_size`、`-d/--device`——**没有 `--concurrency`,没有数据集后端选项,没有多轮/session 相关参数**。这是本系列第三次遇到这个模式（前两次是 llama.cpp 的 `llama-bench`、Ollama 的 `cmd/bench`）——单机推理引擎普遍不会去建模"多租户并发 serving"场景,`llm_bench` 定位是"这台硬件跑这个模型快不快",不是"这个 server 能扛住多少并发 agentic 会话"。

```
cd ~/openvino.genai && grep -rln "perf\|benchmark\|throughput\|latency" .github/workflows/shared/*.yml
```

**零命中**——没有找到类似 TRT-LLM `base_perf_pytorch.csv` 那样的官方性能回归基线追踪机制。性能验证目前看起来依赖 `llm_bench` 手动跑分对比,不是 CI 自动化回归。

## 三、KV Cache Eviction × 精度 × 性能 联合测试——六个系统里这一项做得最扎实的一处

涉及文件：`tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py`、`test_kv_cache_eviction_2.py`

### 测试设计:同一个测试里,性能提升和精度保持都是显式、可调的断言阈值

```
cd ~/openvino.genai && sed -n '55,90p' tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py
```

```python
@dataclass
class CacheOptTestStruct:
    prompt_file: str
    similarity_threshold: float
    avg_cache_usage_optimization_ratio: float   # expecting no less than these optimization ratios
    max_cache_usage_optimization_ratio: float

CacheOptTestStruct(
    test_id="prompts_longer_than_eviction_arena",
    similarity_threshold=0.8,
    max_cache_usage_optimization_ratio=2.0,
    avg_cache_usage_optimization_ratio=1.7,
),
CacheOptTestStruct(
    test_id="prompts_shorter_than_eviction_arena",   # 没触发驱逐的场景
    similarity_threshold=0.98,
    max_cache_usage_optimization_ratio=0.95,  # no improvement expected
    avg_cache_usage_optimization_ratio=0.95,
),
```

对应的断言(同一个测试函数里,两类断言都要过):

```
cd ~/openvino.genai && sed -n '143,163p' tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py
```

```python
max_optimization_ratio = pipeline_noopt_metrics.max_cache_usage / pipeline_opt_metrics.max_cache_usage
avg_optimization_ratio = pipeline_noopt_metrics.avg_cache_usage / pipeline_opt_metrics.avg_cache_usage
...
assert similarity_metric > test_struct.similarity_threshold                          # 精度：相似度必须过线
assert max_optimization_ratio >= test_struct.max_cache_usage_optimization_ratio      # 性能：峰值缓存占用压缩比必须达标
assert avg_optimization_ratio >= test_struct.avg_cache_usage_optimization_ratio      # 性能：平均缓存占用压缩比必须达标
```

**这是六个系统调查以来,唯一一处把"性能收益量化阈值"和"精度保持阈值"写在同一个测试断言里、同时要求都满足的实现。** 对比之前的发现:
- TensorRT-LLM 的 `enable_block_reuse=True` 精度测试,只断言精度过线,不断言"缓存复用带来的收益必须达到多少倍"；
- vLLM/SGLang 的对应测试普遍直接关掉优化项,不测二者的权衡。

而且测试矩阵里专门设计了**边界对照场景**（`prompts_shorter_than_eviction_arena`：prompt 短到不会触发驱逐时，精度阈值提到 0.98、且明确写"no improvement expected"）——说明测试设计者是有意识地在验证"该省的时候真省了、不该动的时候没瞎动"这两个方向,不是只测单一乐观场景。

### 覆盖面:真实长文本数据集(LongBench),覆盖 SnapKV / KVCrush / AdaptiveRKV 多种驱逐算法

```
cd ~/openvino.genai && sed -n '17,33p' tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_2.py
```

```python
from openvino_genai import CacheEvictionConfig, AggregationMode, KVCrushAnchorPointMode, KVCrushConfig
...
OPTIMAL_KVCRUSH_CONFIGS = {
    "samsum": (768, 8, KVCrushAnchorPointMode.ALTERNATING),
    "trec": (960, 2, KVCrushAnchorPointMode.ALTERNATING),
    "qasper": (960, 2, KVCrushAnchorPointMode.ALTERNATING),
}

def test_kvcrush_vs_snapkv_baseline_longbench(subset):
    """Test that KVCrush performs equal or better than SnapKV baseline on LongBench datasets."""
```

用真实 LongBench 子任务(`samsum` 对话摘要、`trec` 问题分类、`qasper` 论文问答)的真实数据 + 真实评分函数(`utils.longbench.evaluate`,按数据集类型选用 ROUGE/分类准确率等标准指标),而不是简单的文本相似度代理——比 WWB 默认用的"和基线模型输出的相似度"更贴近"这个任务到底做对了没有"。

### 确认活跃在 CI 里(Linux/Windows),Mac 上被注释掉

```
cd ~/openvino.genai && grep -n "kv_cache_eviction" .github/workflows/linux.yml .github/workflows/windows.yml .github/workflows/mac.yml
```

```
linux.yml:597:    cmd: 'python -m pytest -v ./tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py'
linux.yml:601:    cmd: 'python -m pytest -v ./tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_2.py'
windows.yml:684:    cmd: 'python -m pytest -s -v tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py'
windows.yml:688:    cmd: 'python -m pytest -s -v tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_2.py'
mac.yml:452:  #   cmd: 'python -m pytest -v ...test_kv_cache_eviction_1.py'   ← 注释掉
mac.yml:457:  #   cmd: 'python -m pytest -v ...test_kv_cache_eviction_2.py'   ← 注释掉
```

**需要说明的局限**:LongBench 的任务全部是单轮长文本 QA/摘要/分类,**不涉及工具调用或多轮 agent 循环**。这项测试对"agentic 特点里的长上下文累积"覆盖得很扎实,但完全没有触及"工具调用输出正不正确""多轮任务最终完不完成"这两个维度——KV cache eviction 对 agent 工具决策能力的影响,在这套测试矩阵里同样是空白。

## 四、边界案例:前缀缓存 + 投机解码的组合测试,只测"不崩",不测精度

```
cd ~/openvino.genai && sed -n '820,847p' tests/python_tests/test_continuous_batching.py
```

```python
def test_eagle3_prefix_caching_no_crash(target_prompt_tokens, eagle3_model_paths):
    scheduler_config = dict_to_scheduler_config({"enable_prefix_caching": True, ...})
    ...
    try:
        second_results = ov_pipe.generate(input_ids, pipeline_generation_config)
    except RuntimeError as exc:
        pytest.fail("Second Eagle3 generate with prefix cache reuse must not raise RuntimeError. ...")

    assert len(first_results.tokens) == 1
    assert len(second_results.tokens[0]) > 0
```

这一处确实把"前缀缓存 + 投机解码（Eagle3）"两个优化项组合在了一起测,但测试名字和断言已经说明一切——`no_crash`,只验证两次 `generate()`（利用了前缀缓存复用）不会在特定 token 边界（127/128/129,明显是在测 block 对齐的边界 bug）抛异常,**完全没有比较两次输出内容是否一致或精度是否保持**。和第三节的 cache eviction 测试对比,这里是"组合覆盖到了,但只测崩不崩,没测对不对"——是本次调查里"优化组合测试"里深度较浅的一个案例,值得和第三节的扎实做法对照着看。

## 五、精度评估方法论(WWB):覆盖面广但类型单一,零 agentic/工具调用评测类型

```
cd ~/openvino.genai && grep -rn "@register_evaluator" tools/who_what_benchmark/whowhatbench/*.py
```

```
image-to-image, image-inpainting, text-to-video, text-chat, reranking,
text-embedding/image-embedding/video-embedding, text-to-image,
visual-text-chat, speech-generation, visual-text/visual-video-text
```

WWB 注册的评测类型横跨文本/图像/视频/语音/多模态——覆盖面在六个系统的自带精度工具里是最广的（另外五家的对应工具基本只覆盖纯文本 QA）。但**没有任何一个评测类型是为工具调用/agent 任务设计的**——`text-chat` 用的核心方法论是"和基线（一般是 FP32/未压缩）模型的输出算相似度分数",这个方法论本身可以套在任何文本任务上,包括理论上可以喂 ReAct/tool-calling 的 prompt 进去比较,但registry 里没有专门的 evaluator 类型或数据集针对这个场景做校准,需要用户自己去接。

```
cd ~/openvino.genai && grep -rniE "bfcl|gorilla|tau.?bench|agentbench|toolbench|swe.?bench" --include="*.py" --include="*.md" .
```

除了一处误报(`bfclm` 是 `BloomForCausalLM` 的缩写)和一处纯粹的 chat template 测试 fixture(`gorilla-llm/gorilla-openfunctions-v2`,只是用来测模板渲染,不是真的接入 Gorilla/BFCL 评测集),**没有任何标准 agentic 精度基准被接入**——和另外五家结论一致。

WWB 已确认通过 `linux.yml` 的多个矩阵化 job(image/VLM/video、nanollava、transformers 版本兼容性等)接入日常 CI——这一点比 llama.cpp(困惑度零 CI 化)、Ollama(零精度评估)更扎实,和 vLLM/SGLang/TRT-LLM 的 lm-eval 类工具处于同一档,但覆盖的任务类型依然不含 agentic/工具调用。

## 六、综合结论

### 性能侧

1. `llm_bench` 是单机批处理式微基准（无并发/数据集/多轮建模）,和 llama.cpp `llama-bench`、Ollama `cmd/bench` 同属最原始一档,符合"单机推理引擎不需要 serving 级压测"的定位,但也意味着**目前完全没有面向 agentic 场景（多轮、并发会话、前缀共享）的性能评测手段**；
2. 没有找到官方性能回归 CI/基线追踪机制。

### 精度侧

1. **本次六系统调查里,"优化 × 精度"联合测试做得最扎实的一处出现在这里**：`test_kv_cache_eviction_1/2.py` 把"精度相似度阈值"和"缓存占用压缩比阈值"写进同一个断言、覆盖 SnapKV/KVCrush/AdaptiveRKV 多种驱逐算法、用真实 LongBench 数据和标准评分函数、且确认活跃在 Linux/Windows CI（Mac 暂时禁用）；
2. 但这项优势的覆盖范围局限在"长上下文 QA/摘要/分类"，完全没有延伸到"工具调用输出对不对""多轮 agent 任务完不完成"——`test_eagle3_prefix_caching_no_crash` 这个前缀缓存+投机解码组合测试就是反例,只测"崩不崩",不测"对不对";
3. 工具调用测试分两层：底层 parser 单测（含巧妙复用 vLLM 解析器生态的 `VLLMParserWrapper`）+ 唯一带 `@pytest.mark.agent` 标记、真实推理的 `test_react_sample_refs`——但后者验证的是"两种语言绑定的输出是否逐字节一致",不是"ReAct 循环有没有把任务做对",本质上仍然没有触及真实的 agentic 任务正确性；
4. WWB 覆盖面（文本/图像/视频/语音/多模态）是六个系统自带精度工具里最广的,且真正接入了 CI,但评测类型注册表里没有工具调用/agentic 这一类,和另外五家一样,没有接入任何标准 agentic 精度基准（BFCL/τ-bench/AgentBench）。

### 一句话总览,及在六系统里的定位

OpenVINO GenAI 在"某个具体优化项的性能-精度权衡该如何联合测试"这个问题上,给出了六个系统里**方法论最严谨的范本**（`test_kv_cache_eviction`）——这一点值得作为其他系统改进自己测试设计的参考;但和另外五家一样,这套严谨的方法论目前只覆盖到长上下文 QA 这一类"agentic 特点",没有延伸到工具调用/多轮任务完成度这个更核心的 agentic 精度维度。加上 `llm_bench` 在性能侧的原始程度、以及 WWB 精度类型注册表里没有 agentic 选项,最终结论和另外五家汇合到同一句话：**没有一个系统把"多轮会话 + 工具调用 + 主流性能优化组合开启"作为一个整体,纳入过日常自动化的 CI 精度回归**——OpenVINO GenAI 的独特价值在于,它已经证明了"联合测试性能和精度阈值"这件事在方法论上是可行的、值得推广的,只是还没有把这套方法论用到工具调用场景上。
