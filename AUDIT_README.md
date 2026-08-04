# Agentic AI Runtime Audit Suite

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：**详细说明（本文档）** · [5分钟上手](QUICKSTART.md)

一套用于调查 **Runtime Framework 对 Agentic Workload 的性能和精度评估能力** 的自动化脚本。

## 问题定义

当一个 Runtime（如 vLLM、SGLang、TensorRT-LLM）声称"支持 agentic workload"时，需要核实：

1. **Benchmark Harness**：它的压测工具能不能真的生成"多轮 + 工具调用 + 共享前缀 + 思考间隔"的负载？
2. **CI 配置**：它的 CI pipeline 真的在日常守护什么指标？对 agentic 相关优化做了精度回归吗？

## 工具清单

### 脚本 1：`1_bench_harness_audit.py` — 审查 Benchmark 源码

**目标**：判断 harness 能生成什么形状的负载。

**运行**：
```bash
python3 1_bench_harness_audit.py /path/to/repo
```

**检查内容**：
- ✓ Benchmark 目录位置（`benchmarks/`, `bench/` 等）
- ✓ 负载形状：多轮、session、前缀共享、agent/tool-call 相关代码
- ✓ 数据集后端（random / ShareGPT / 多轮专属 / agentic）
- ✓ 请求到达模型（泊松 iid vs 依赖驱动）

**产出**：
```
=== VERDICT ===

Multi-turn capable:     YES
Session-aware:          NO
Prefix-sharing aware:   YES
Agent/tool-call aware:  NO

★★★☆☆ Strong agentic support
→ This is a 'prefix-cache microbench' harness
```

**解读**：
- **★★★★★**：Full agentic support → 真的在测多轮工具循环
- **★★★★☆**：Strong agentic support
- **★★★☆☆**：Partial（通常是前缀缓存微基准，不是真 agentic）
- **★☆☆☆☆**：Pure single-turn → 不适合评测 agentic

---

### 脚本 2：`2_ci_config_audit.py` — 审查 CI 配置

**目标**：发现 CI 真正在守护什么指标。关键原则：**CI 不会说谎**。

**运行**：
```bash
python3 2_ci_config_audit.py /path/to/repo
```

**检查内容**：
- ✓ Performance CI job（有没有、测什么负载）
- ✓ Accuracy CI job（评测什么任务）
- ✓ **关键**：有没有 BFCL / tau-bench / tool-calling 精度任务
- ✓ **关键**：打开 prefix cache / KV 量化 / cache eviction 时，精度回归还在测吗？

**产出**：
```
Performance CI:                 YES
Accuracy/Eval CI:               YES
Agentic/Tool-call CI:           NO ← IMPORTANT
Optimization-specific CI:       PARTIAL/NONE ← CRITICAL

⚠ MAJOR FINDING:
  This runtime has NO CI-guarded agentic evaluation.
  Implications:
    • Tool-call accuracy under optimizations is UNTESTED
    • Agent behavior in multi-turn loops may degrade silently
    • Prefix-cache/KV-quant/eviction safety for agentic unproven
```

**解读**：
- 如果没有 **Agentic/Tool-call CI**：说明runtime没有在daily build时保证工具调用准确率
- 如果没有 **Optimization-specific CI**：说明打开 KV 量化、cache eviction 等优化时，精度合规性**完全无人守护**

这是最尖锐的发现。大多数项目都是这样。

---

### 脚本 3：`run_audit.sh` — 批量运行脚本

**目标**：一次性审查多个 repo。

**运行**：
```bash
chmod +x run_audit.sh
./run_audit.sh /path/to/vllm /path/to/sglang /path/to/TensorRT-LLM
```

**或者逐个运行**：
```bash
python3 1_bench_harness_audit.py /path/to/vllm
python3 2_ci_config_audit.py /path/to/vllm
```

---

## 使用场景

### 场景 1：快速对比 vLLM vs SGLang vs TRT-LLM

```bash
./run_audit.sh ../vllm ../sglang ../TensorRT-LLM > audit_report.txt
```

然后对比 three runtimes 的 **verdict** 部分。答案直接就是：谁对 agentic 的测试最完整。

### 场景 2：深入调查某个 Runtime

```bash
python3 1_bench_harness_audit.py ../vllm 2>&1 | tee vllm_harness.log
python3 2_ci_config_audit.py ../vllm 2>&1 | tee vllm_ci.log
```

然后手工查看 log 里指出的具体文件和行号，进一步深挖。

### 场景 3：构建你自己的 agentic benchmark

脚本会告诉你：
- vLLM harness 能生成什么形状的负载 → 你用什么方式改造它来适配你的 workload
- vLLM CI 在守什么 → 你应该参考什么指标来设计自己的评测框架

---

## 重要提示

### 1. 脚本扫的是模式匹配，不是真正的代码执行

```
✓ "prefix_cache" 出现在代码里 ≠ 功能真的工作
✓ 脚本只能说"这个词出现了 N 次"，不能说"这个功能对吗"
```

**所以每个发现都需要**：进去看指定的文件/行号，确认 context。

### 2. CI 的缺失比有什么更有价值

```
❌ "没有 agentic 评测" 这个发现本身就是结论

因为：
  • 说明 tool-calling 精度在多轮 + 优化下是未知的
  • 说明"可能好、可能坏，没人测过"
```

### 3. 脚本会给出明确的"可进一步调查"的 grep 命令

看到 `xxx.py:123` 这样的位置，直接去 repo 里 grep 那行，看 context。

---

## 输出格式说明

### 脚本 1 的 VERDICT

根据 4 个维度给分：

| 指标 | 意义 |
|---|---|
| **Multi-turn capable** | 代码里能找到"多轮"概念吗 |
| **Session-aware** | 有没有 session/conversation 状态管理 |
| **Prefix-sharing aware** | 有没有前缀缓存、block reuse 等 |
| **Agent/tool-call aware** | 有没有 ReAct、tool-call 相关代码 |

分数意义：
- ⭐️⭐️⭐️⭐️⭐️ (5/4)：Full agentic support → 真在测 agent loop
- ⭐️⭐️⭐️⭐️ (4/4)：Strong agentic
- ⭐️⭐️⭐️ (3/4)：Partial（通常是 prefix-cache focused，缺 agent 特性）
- ⭐️⭐️ (2/4)：Minimal
- ⭐️ (1/4)：No agentic features → Pure single-turn

### 脚本 2 的 VERDICT

关键看这两条：
```
Agentic/Tool-call CI:       YES/NO ← 决定了 tool-call 精度是否有人守护
Optimization-specific CI:   YES/PARTIAL/NONE ← 决定了优化后精度是否有人守护
```

如果都是 NO/NONE，说明这个 runtime 对 agentic 的承诺只是营销，工程支持为 0。

---

## 依赖

```bash
pip install pyyaml
```

或在 `run_audit.sh` 会自动装。

---

## 实际使用例子

假设你要调查 vLLM 对 agentic 的支持：

```bash
# 1. Clone vLLM
git clone https://github.com/vllm-project/vllm ../vllm

# 2. 运行审计
python3 1_bench_harness_audit.py ../vllm

# 输出会说：
# Multi-turn capable: YES
# Agent/tool-call aware: YES (或 NO)
# Verdict: ★★★☆☆
#
# 然后会指出：
# benchmarks/xxx.py:123 → "multi-turn"
# tests/ci_xxx.yml:45 → "gsm8k"

# 3. 手动验证
cd ../vllm
grep -n "multi.turn" benchmarks/serving/serving.py | head -5
cat benchmarks/serving/serving.py | sed -n '120,130p'

# 4. 看 CI
cat .github/workflows/perf_nightly.yml | grep -A5 "accuracy"
```

每一步都指向源码，你可以一层层深挖。

---

## 常见问题

**Q: 脚本说"Multi-turn capable: YES"，但我看代码里只有一个 `multi_turn` 变量？**

A: 正常。脚本只是"找到了这个词"。你需要进去看代码逻辑，判断它是真的多轮处理，还是只是变量名字借用。这就是为什么脚本给出行号——让你能跳到代码去。

**Q: 脚本说 CI 没有 agentic 评测，但 blog 里说支持了？**

A: **CI 是权威**。Blog 可以吹，但 daily nightly build 是客观的。如果 CI 没守，说明那个特性没被 engineering 真正验证过。这个结论应该直接上你的报告。

**Q: 怎么看脚本的输出？**

A: 看三个地方：
1. **VERDICT** 部分 — 快速的分数和评价
2. **具体的文件:行号** — grep 出的证据
3. **缺失部分** — 用红色 ✗ 标出来的，说明某个能力完全没有

---

## 下一步

脚本只是第 1、2 层的自动化取证。实际调查的完整路径是（第 1-5 节对应文章）：

```
脚本1: 审查 harness 源码 ———┐
                           ├→ 判断这个 runtime 的能力真实水位
脚本2: 审查 CI 配置 ————————┤
                           │
查阅 PR/RFC (GitHub) ──────┴→ 回答"怎么评估"的完整链路

    ↓
自己跑一遍 vllm bench serve / SGLang bench_serving
    ↓
得出"这个 runtime 现在对 agentic 负载的评估能力是 XXX"的结论
```

脚本处理的是前两步。后面的工作（PR 挖掘、手工跑 bench）仍需人工。

---

## 许可

这些脚本是为了学习和研究目的。自由使用和修改。
