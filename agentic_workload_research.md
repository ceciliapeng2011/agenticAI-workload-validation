# Agentic Workload 方法论笔记

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：**方法论笔记（本文档）** · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

一、Agentic AI Workload 的特点（从 Runtime / 系统角度，不是从 "agent 聪不聪明" 角度）
普通 chat workload 是 "一问一答"，agentic workload 的本质区别在于一个任务 = 一串相互依赖的 LLM 调用，由此产生几个系统层面的独特形状：

表格

| 特点 | 具体表现 | 为什么 Runtime 要关心 |
| --- | --- | --- |
| 多轮、上下文累积增长 | Turn N 的输入 = 全部历史 + 新增内容，不是独立请求 | prompt 长度随轮次线性增长，KV cache 占用持续上升 |
| 高前缀重复率 | 每一轮都重新发送 system prompt + 工具 schema + 之前所有轮次 | 这是 prefix caching /prefix sharing 存在的意义 —— 不复用就是纯浪费算力 |
| 长输入、短输出 | 每轮通常是 "一大段历史 + 工具结果" 进去，只吐出一小段 thought/action | 和传统 chat 的 "输入输出都适中" 完全不同的 prefill/decode 比例 |
| 调用间存在非计算的 "空档" | 等工具执行、等外部 API 返回 | 这段时间 KV cache 要不要保留、保留多久、会不会被驱逐，是 agentic 特有的调度问题 |
| 依赖驱动的到达，不是泊松 | 下一次 LLM 调用必须等上一次工具执行完才发生 | 大多数压测工具默认泊松到达，这对 agentic 是错误的建模 |
| 工具调用是输出的一部分，且要被解析 / 执行 | 输出不是自然语言，而是结构化的 function call | 需要约束解码 (grammar)、tool-call parser，精度不能只看 "像不像人话"，要看 "能不能被正确解析和执行" |
| 多 Agent / 并发子任务 | 一个任务可能 fan-out 成多个子 agent 并行跑 | 跨请求的前缀共享 (不只是同一会话内) 变得重要 |
| 正确性是 "整条轨迹" 的正确性，不是单次输出的正确性 | 单次回复语法正确 ≠ 任务最终完成 | 精度评估要看任务终态，不能只看单轮 exact match |

这 8 条，就是我在读 vLLM 代码时一直在验证 "有没有被建模" 的清单 —— 你会发现上一份报告里几乎每个发现都对应其中一条（比如 "BFCL 被丢弃了多轮部分" 对应第 1 条，"tool\_use 测试关闭了 prefix caching" 对应第 2 条，"CI 用泊松到达" 对应第 5 条）。

## 二、对应的基准测试（按上面这些特点分类，而不是按 "名气" 排序）

表格

| 基准 | 主要测什么特点 | 关键设计 |
| --- | --- | --- |
| BFCL(Berkeley Function Calling Leaderboard) | 第 6 条：工具调用输出的可解析性 / 正确性 | 分 single-turn /multi-turn/parallel 等 category；判定标准是 AST 匹配或真实执行结果匹配，不是文本相似度 |
| τ-bench / τ²-bench | 第 7 条 + 第 8 条：多轮工具调用的任务终态正确性，以及 "重复跑 k 次是否稳定"(passᵏ) | 用数据库终态是否和标准答案一致来判分，而不是判断某一句话对不对 |
| AgentBench / GAIA / SWE-bench | 第 8 条：长链路任务的最终完成度 | 任务通常需要几十次工具调用才能收尾，天然会暴露 "多轮误差累积" |
| WebArena / OSWorld | 第 1+4 条：长上下文、真实环境反馈延迟 | 网页 / 系统状态截图或 DOM 树进入上下文，输入体积随步数暴涨 |
| MLPerf Inference agentic/interactive（需核实版本） | 第 1、5、6 条的系统性能面 | 尝试把 "多轮工具调用" 标准化成一个可复现的性能测试场景，侧重性能而非精度测试 |

这里有一个我在报告里反复强调的关键区分，再次说清楚：
BFCL/τ-bench 这类是**精度基准**—— 判定 "回答对不对"；
而 "这个 runtime 能不能把 BFCL 的负载扛住、扛得快不快" 是完全不同的**性能问题**。

上一份 vLLM 调查报告最核心发现：vLLM 把 BFCL 拿来只做了后者（当作负载生成器测吞吐），没有做前者（没有利用 BFCL 判定工具调用正确性）。两件事极易混淆，后续调研任何推理 Runtime 都需要重点区分。

## 三、为什么我这样检索代码 —— 关键词是怎么来的

不是拍脑袋搜的，是先建立「特点 → 代码里应该长什么样 → 该搜什么词」映射链，再执行 grep。几个典型示例：

1. **第 2 条：高前缀重复率 & prefix caching**
假设：如果该特性被纳入精度测试，代码测试配置会存在前缀缓存开关参数。
→ 不宽泛搜索 `prefix`，精准搜索 `--no-enable-prefix-caching`
目标不是确认功能存在，而是验证：**精度测试是否在开启前缀缓存的环境下执行**。
2. **第 6 条：工具调用输出校验**
假设：评测工具调用正确性，代码必然存在模型输出 `tool_calls` 和标准答案对比逻辑。
→ 全局检索 `vllm/benchmarks/*.py` 内 tool\_call 校验代码
检索结果为空本身就是证据：性能压测链路完全不校验工具调用结果正确性。
3. **第 1 条：多轮上下文处理（BFCL Dataset）**
假设：支持多轮 Agent 负载，则 BFCL 加载器会完整使用数据集多轮对话结构。
→ 不只搜索关键词 `multi-turn`，直接阅读 `BFCLDataset.sample()` 实现逻辑
最终定位关键注释：`Use the first turn only; skip multi-turn categories`。

> 
> 关键词搜索只能确认 "文本是否被提及"，阅读源码才能确认**实际运行行为**。

4. **自动化 CI 覆盖缺口证据挖掘**
假设：开发团队自知测试缺口，通常会写在测试类文档注释 (docstring) 或 PR 描述。
→ 定向读取 `TestServingChatWithHarmony` 的文档字符串
拿到维护者一手书面说明，属于强证据，强于单纯 "检索不到代码就下结论"。

一句话总结方法论：
先把 workload 特点转化成「该特性如果完整支持，代码实现应当具备什么形态」的可验证预期；
再基于预期定向检索 / 阅读源码：搜到则验证假设、检索为空本身是负面发现、阅读源码实现判断功能是否完整落地。

泛泛搜索 `agentic`、`multi-turn` 这类大词只能找到相关入口；想要判断**支持程度、CI 覆盖是否完备**，必须下沉到具体机制、参数、执行逻辑层面验证。


---

### 用户

## 一、更完整的测试类型谱系 (不只是 "精度基准", 按测试性质分类)

上次只讲了 "精度基准测试"(BFCL、τ-bench 这类), 这次按测试的性质把范围铺开，因为这是判断一个 runtime 测得全不全的正确维度:

表格

| 测试类型 | 测的是什么 | 是否需要真跑模型推理 | 典型例子 |
| --- | --- | --- | --- |
| 单元测试 (Unit test) | 纯逻辑正确性，如 "给一段字符串，tool-call parser 能不能正确切出 JSON" | 不需要 —— 喂固定字符串，断言解析结果 | 字符串解析器测试、schema 转换测试、调度器数据结构测试 |
| 功能 / 集成测试 (Functional/Integration) | 打开某个优化 (前缀缓存 / KV 量化 /offload) 后，端到端输出是否还 "正确" | 需要 —— 起一个真实 server/engine | "开 cache 和不开 cache, 输出是否一致" 这类测试 |
| 回归测试 (Regression test) | 专门为修复过的某个具体 bug 写的，防止它复发 | 视 bug 而定 | 很多测试文件开头就写 "regression guard for issue #xxx" |
| 精度 / 能力基准 (Accuracy/Capability benchmark) | 任务能不能做对，通常是外部标准数据集 | 需要 | BFCL、τ-bench、AgentBench、GAIA、SWE-bench、WebArena、OSWorld |
| 性能基准 (Performance benchmark) | 快不快，吞吐 / 延迟 | 需要 | throughput/latency/serving tests |
| 容错 / 混沌测试 (Fault-tolerance /chaos) | 进程崩溃、节点掉线之后，系统能不能恢复、恢复要多久 | 需要，且要主动注入故障 | kill 掉一个 rank, 看多久探测到并恢复 |
| 压力 / 持续负载测试 (Stress/soak test) | 长时间高并发下会不会内存泄漏、性能衰减、死锁 | 需要，且要跑得久 | 业界常见工具:locust、k6、长跑 nightly |

这张表本身就是下一节答案的检索地图 —— 我在 vLLM 仓库里就是按这 7 类逐个去找有没有对应物。

## 二、vLLM 具体是怎么测的 —— 逐项列举 + 对应目标 / 特点

```
# Check if there's an explicit with/without prefix caching output equivalence test
cd ~/agenticAI/vllm && grep -n "def test_prefix_caching_model_output_matches\|same output\|deterministic\|assert.*==.*without" tests/v1/core/test_prefix_caching.py tests/basic_correctness/*.py 2>/dev/null | grep -i "prefix\|output" | head -10
```

> 
> (Bash completed with no output)

```
# Look for prefix-caching output-equivalence functional test
cd ~/agenticAI/vllm && ls tests/basic_correctness/; echo "---"; grep -rln "prefix" tests/basic_correctness/*.py
```

```
__init__.py
test_basic_correctness.py
test_cpu_offload.py
test_mem.py
test_prefetch_offload.py
---
```

```
# Check if basic_correctness parametrizes over prefix caching
cd ~/agenticAI/vllm && grep -n "prefix_caching\|enable_prefix" tests/basic_correctness/test_basic_correctness.py | head -10
```

> 
> (Bash completed with no output)

```
# Peek structured output + reasoning test intent
cd ~/agenticAI/vllm && sed -n '1,30p' tests/v1/structured_output/test_reasoning_structured_output.py 2>/dev/null | head -30
```

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

class MockReasoner:
    def __init__(self, tokenizer):
        self.is_reasoning_end = Mock(return_value=False)
        self.is_reasoning_end_streaming = Mock(return_value=False)

class TestReasoningStructuredOutput:
    """Test reasoning-aware structured output functionality."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock ModelConfig."""
        config = Mock(spec=ModelConfig)
        config.skip_tokenizer_init = True  # Skip tokenizer init to avoid network calls
        config.get_vocab_size = Mock(return_value=50000)
```

现在给出完整清单 —— 每一项都对应到具体文件 / 证据，不是泛泛列名字。

# vLLM 的测试机制汇总表

表格

| # | 测试机制 | 位置 | 测试类型 | 具体测什么 | 对应哪条 agentic 特点 (前文 8 条) |
| --- | --- | --- | --- | --- | --- |
| 1 | Tool-call parser 单测 | `tests/tool_parsers/*.py`(每家模型一个文件，如 hermes、glm4、deepseek 系列) | 单元测试 | 喂固定字符串 (模型可能吐出的原始文本), 断言解析出的 JSON 工具调用是否正确、能否处理 parallel/malformed/escaped 情况。不跑真实模型 | 特点⑥"输出要能被正确解析" —— 但只测解析器代码，不测模型本身输出对不对 |
| 2 | Tool-call 端到端正确性 | `tests/tool_use/test_tool_calls.py` 等 | 功能 / 集成测试 | 起真实 server, 验证模型真的能触发工具调用、参数值对、能处理工具结果。但如前所述：11/12 配置关闭 prefix caching,"多轮" 是写死历史一次性验证 | 特点⑥ + 特点②(但排除了②, 因为 cache 关着) |
| 3 | Prefix cache 内部正确性 | `tests/v1/core/test_prefix_caching.py`, `test_reset_prefix_cache_e2e.py`, `prefix_cache/test_partial_prefix_cache_hits.py` | 单元测试 | 测的是 block hash、block 复用、部分命中这些调度器内部数据结构逻辑对不对 (比如两个请求共享前缀时 block pool 分配对不对)。不测 "开 cache 前后模型输出是否一致"—— 我专门搜了 `tests/basic_correctness/test_basic_correctness.py`, 确认它没有对 `enable_prefix_caching` 做 on/off 参数化的输出等价性验证 | 特点②(高前缀重复率) 的内部机制正确性，但缺 "对最终输出无影响" 这一环 |
| 4 | Scheduler 多请求 / 多 "轮" 测试 | `tests/v1/core/test_scheduler.py::test_ec_connector_with_partial_cache_hit_multi_round` | 单元测试 | 名字带 "multi\_round", 但实际测的是同一个多模态 hash 在两个独立请求间被复用时 block 分配是否正确 —— 不是对话轮次，是 "多模态素材缓存复用" 的场景，容易望文生义搞混 | 特点②的一个子情形 (多模态素材前缀复用), 不是特点①(会话历史累积) |
| 5 | KV Offload + 精度联合测试 | `tests/evals/gsm8k/test_gsm8k_offloading.py` | 功能测试 + 精度联合验证 (全仓库里少见的好实践) | 明确写着 "是为了防止 CPU offload 时 KV 数据静默损坏"—— 两次跑 GSM8K, 一次用 GPU cache、一次强制从 CPU reload, 如果 offload 逻辑有 bug (比如 stride 计算错), 数值会漂移到损坏 KV,GSM8K 准确率会掉到阈值以下从而报警。覆盖 5 种架构 (含 Mamba 混合、MLA 等) | 这是目前仓库唯一一处把 "优化 (offload)" 和 "精度 (准确率阈值)" 直接挂钩验证的地方 —— 但任务仍是单轮 GSM8K, 不是多轮 / 工具调用 |
| 6 | KV Offload/Connector 功能测试 | `tests/v1/kv_offload/`, `tests/v1/simple_kv_offload/`, `tests/quantization/test_cpu_offload.py`, `tests/basic_correctness/test_cpu_offload.py` | 功能测试 | 大多测 "连接器逻辑对不对"(数据搬没搬对、tiering 逻辑对不对), 不直接挂精度指标 (#5 是例外) | 特点②③的支撑机制 |
| 7 | 结构化输出 (约束解码) 测试 | `tests/v1/structured_output/`, `test_reasoning_structured_output.py`, `test_mtp_structured_output.py` | 单元测试 | 验证 grammar 约束解码在 "推理模式 (reasoning)" 和 "投机解码 (MTP)" 共存时，是否仍能产出合法 JSON。这是 "工具调用输出必须结构化" 这条能力的底层支撑 | 特点⑥, 但测的是通用 JSON schema 合规性，不挂钩具体工具调用任务的正确率 |
| 8 | lm-eval-harness 精度回归 | `.buildkite/lm-eval-harness/`(38 个模型配置) | 精度基准 (外部) + CI 回归 | gsm8k/mmlu 这类标准单轮任务，量化 (INT8/FP8/AWQ) 前后精度不能掉太多 | 不覆盖任何 agentic 特点 —— 纯单轮任务 |
| 9 | BFCL 数据集接入 | `vllm/benchmarks/datasets/datasets.py::BFCLDataset` + `tests/benchmarks/test_bfcl_dataset.py` | 性能负载生成器 + 单测 (仅测数据管道) | 真实工具调用 schema 形状的 prompt, 用于压测吞吐；单测只验证 schema 翻译 / 参数解析这些 "数据管道" 逻辑，不验证模型工具调用对不对；且主动丢弃了 BFCL 的多轮类别 | 特点⑥(仅负载真实感)+ 特点①(但被主动阉割了) |
| 10 | 官方性能基准 (latency/throughput/serving) | `.buildkite/performance-benchmarks/` | 性能基准 | ShareGPT 单轮、泊松到达，测 e2e latency /throughput/ TTFT / ITL | 不覆盖任何 agentic 特点 —— 单轮 iid |
| 11 | 多轮感知的性能压测工具 | `rust/src/bench/`(vllm-bench) | 性能基准 (工具存在，未接入官方 CI) | 真实多轮会话建模：历史累积、think-time 延迟、前缀共享比例、session 亲和、per-turn 指标分解 | 特点①②④⑤全覆盖，但仅存在于工具箱，没有被 #10 的官方 CI 采用 |
| 12 | 容错 / 混沌测试 | `tests/v1/fault_tolerance/test_fault_tolerance_e2e.py` | 容错测试 | 主动 kill 掉一个 DP rank, 验证系统在 45 秒内探测到故障并恢复，不影响其余请求 | 不直接对应 agentic 特点，但和 "长时间运行的 agent 会话不能被单点故障打断" 这个诉求相关 |
| 13 | 压力 / 持续负载测试 | 未找到 | — | 检索 `locust`、`k6`、`stress`、`soak`、`chaos` 关键词，`.buildkite/` 和 `tests/` 下都没有专门的长时间高并发压力测试 | 特点⑤(依赖驱动到达、长会话累积并发) 目前没有对应的压力测试，只有 #10 这种短时点性能测试 |

## 三、这张表最值得带走的三条结论

1. "多轮" 这个词在 vLLM 代码里至少有三种完全不同的含义，检索代码务必避免字面混淆:
   - `test_tool_calls.py` 里的 "多轮" = 一段写死的历史，单次静态验证
   - `test_scheduler.py` 里的 "multi\_round" = 同一多模态素材在不同请求间复用
   - `rust/src/bench` 里的 "multi-turn" = **真正的上下文累积会话**
   只有第三种是 Agentic 场景需要的多轮，但是恰恰这一套能力没有接入日常 CI。
2. **优化 × 精度** 的联合验证极度稀缺：
仓库目前仅有 #5 (`gsm8k_offloading.py`) 实现了「功能优化 + 准确率阈值断言」绑定；
前缀缓存、结构化输出、KV 量化均缺少同类验证。`test_prefix_caching.py` 只校验内部数据结构，**不验证开启缓存是否改变模型输出内容**。
3. 压力 / 持续负载测试整体缺失：
Agentic 场景典型特征是长会话、持续累积上下文、依赖驱动式并发；
当前全部性能测试属于**短时快照测试**，缺少长时间运行观测内存泄漏、性能缓慢衰减、死锁等稳定性问题的流水线。
