# Quick Start — 5分钟上手

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
> **审计脚本**：[详细说明](AUDIT_README.md) · **5分钟上手（本文档）**

## 最短路径：审计你下载的 repo

```bash
cd ~/agenticAI

# 对单个 repo 审计
python3 1_bench_harness_audit.py /path/to/vllm
python3 2_ci_config_audit.py /path/to/vllm

# 或一键批量（3个 repo）
./run_audit.sh /path/to/vllm /path/to/sglang /path/to/tensorrt-llm
```

## 看懂输出（重点看这 5 行）

### 脚本 1 输出（Benchmark Harness）

```
=== VERDICT ===
Multi-turn capable:     YES/NO     ← 能测多轮吗
Session-aware:          YES/NO     ← 有 session 管理吗
Prefix-sharing aware:   YES/NO     ← 能测前缀复用吗
Agent/tool-call aware:  YES/NO     ← 能测工具调用吗

★★★☆☆ Partial support (prefix-cache focused)
→ This is a 'prefix-cache microbench' harness
```

**简单判断**：
- 4 个都是 YES → 真的有 agentic benchmark ✓
- 只有前 3 个 YES → 只有前缀缓存，缺 agent 真实特性 ⚠️
- 都是 NO → 纯单轮，不适合测 agentic ✗

### 脚本 2 输出（CI 配置）

```
Agentic/Tool-call CI:       YES/NO ← 最关键：有没有 CI 守护工具调用准确率
Optimization-specific CI:   YES/PARTIAL/NONE ← 有没有在优化条件下测精度

⚠ MAJOR FINDING:
  This runtime has NO CI-guarded agentic evaluation.
```

**简单判断**：
- 第一行是 NO → **即使声称支持 agentic，工具调用精度在 CI 里根本没人测** 🚨
- 第二行是 NONE → **打开 KV 量化、cache eviction 这些优化后的精度，没人守** 🚨

---

## 用一个例子

假设你要对比 vLLM vs SGLang：

```bash
# 准备两个 repo
ls /path/to/vllm
ls /path/to/sglang

# 一键审计
./run_audit.sh /path/to/vllm /path/to/sglang > report.txt

# 看输出（只看 VERDICT 部分）
cat report.txt | grep -A10 "VERDICT"
```

输出会是这样：

```
=== vLLM ===
★★★☆☆ Partial support
  Agentic/Tool-call CI:     NO

=== SGLang ===
★★★★☆ Strong agentic support
  Agentic/Tool-call CI:     YES
```

**结论**：SGLang 对 agentic 的评估比 vLLM 更完整。

---

## 如果想深挖某个发现

脚本会给出 `文件:行号`，直接进去看：

```bash
# 脚本输出：
#   benchmarks/serving.py:123 → "multi-turn"
# 你就直接：

cd /path/to/repo
sed -n '120,130p' benchmarks/serving.py
# 或用你的编辑器 Ctrl+G 跳到 123 行
```

---

## 三种常见场景

### 场景 1：快速对比 3 个 runtime

```bash
./run_audit.sh /path/a /path/b /path/c | tee comparison.txt
# 看 VERDICT，5 分钟完成
```

### 场景 2：深入调查某个 runtime

```bash
python3 1_bench_harness_audit.py /path/vllm > vllm_harness.txt 2>&1
python3 2_ci_config_audit.py /path/vllm > vllm_ci.txt 2>&1

# 看输出里的所有"文件:行号"，逐个验证
vim /path/vllm/benchmarks/serving.py
# ... 手工检查
```

### 场景 3：建你自己的评测框架

```bash
# 先看 vLLM 怎么评测：
python3 1_bench_harness_audit.py /path/vllm
# 看输出里的数据集列表、负载参数等，然后改造或模仿

# 再看 vLLM CI 怎么守护精度：
python3 2_ci_config_audit.py /path/vllm
# 看输出里的评测任务、config 覆盖等，参考建立你自己的 CI
```

---

## 常见输出解读

| 输出 | 意义 | 怎么办 |
|---|---|---|
| ★☆☆☆☆ | 纯单轮，没有 agentic | 找个有星的 runtime 参考 |
| ★★★☆☆ + "prefix-cache microbench" | 只测前缀复用，不测 agent | 看它的 benchmark 源码，自己改 |
| ★★★★☆ + "Strong agentic" | 比较完整的 agentic 评测 | ✓ 参考这个 |
| "Agentic/Tool-call CI: NO" | 没人测工具调用准确率 | 🚨 这个很重要，写进报告 |
| "Optimization-specific CI: NONE" | 优化条件下精度无人守 | 🚨 更重要，也要写进去 |

---

## 脚本没法回答的问题（需要你手工）

脚本 1、2 只能说"有没有"。不能说"好不好"：

```
脚本能说：
  ✓ "有 multi-turn 代码" 或 "没有"
  ✓ "CI 里有 BFCL" 或 "没有"

脚本不能说：
  ✗ "那个 multi-turn 实现是否正确"
  ✗ "那个 BFCL 评测是否覆盖了所有优化配置"
  ✗ "精度数字好不好"
```

所以脚本的输出最多是"去看 X.py 的 Y 行"。最终的判断还是要你读代码。

---

## 如果遇到问题

```bash
# 脚本需要 Python 3.6+
python3 --version

# 需要 pyyaml（run_audit.sh 会自动装，或手工装）
pip install pyyaml

# 脚本卡住或出错？
# 加 -v 看详细日志（目前没有，但脚本可以改）
# 或直接看脚本代码改逻辑
```

---

## 下一步

用脚本初步扫描 → 发现关键缺失 → 手工验证 → 写成报告。

脚本是"自动化的第一遍侦查"，不是"最终结论"。
