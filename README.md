# agenticAI

调查主题：**目前支持 Agentic AI Workload 的底层 Runtime Framework（vLLM、SGLang、TensorRT-LLM、llama.cpp、Ollama、OpenVINO GenAI），它们各自是怎么测试/评估自己对这类负载的性能和精度的？**

调查方法：不依赖关键词命中数量或官方文档/博客的自述，逐个打开源码（CI 配置、benchmark harness、测试用例）读实现逻辑，验证"某个能力是不是真的存在、真的被测了、真的接入了日常 CI"，而不是"提到了这个词"。

## 目录结构

```
agenticAI/
├── README.md                                          本文件
├── QUICKSTART.md                                       审计脚本 5 分钟上手
├── AUDIT_README.md                                     审计脚本详细说明
│
├── 1_bench_harness_audit.py                            脚本1：审查 benchmark 源码，判断负载形状
├── 2_ci_config_audit.py                                脚本2：审查 CI 配置，判断日常真正守护什么
├── 3_guided_investigation.sh                            脚本3：引导式人工深挖 + 结构化记录
├── run_audit.sh                                         批量运行脚本1、2
│
├── vllm_agentic_evaluation_investigate.md               vLLM 调查报告
├── sglang_agentic_evaluation_investigate.md             SGLang 调查报告
├── tensorrt_llm_agentic_evaluation_investigate.md       TensorRT-LLM 调查报告
├── llama_cpp_agentic_evaluation_investigate.md          llama.cpp 调查报告
├── ollama_agentic_evaluation_investigate.md             Ollama 调查报告
├── openvino_genai_agentic_evaluation_investigate.md     OpenVINO GenAI 调查报告（仓库在本目录之外，见下）
├── cross_comparison_agentic_evaluation.md               六份报告的横向对比
│
├── agentic_workload_research.md                         方法论笔记：agentic workload 有什么特点、对应什么基准测试、为什么这样检索代码
├── vllm_investigation.md                                脚本3在 vLLM 上的一次人工检查点记录（早期产物）
│
├── logs/                                                 脚本1、2 的原始运行日志（gitignore，不进版本库）
└── vllm/ sglang/ TensorRT-LLM/ llama.cpp/ ollama/         被调查的五个仓库（gitignore，需自行 clone，见下）
```

> `openvino_genai_agentic_evaluation_investigate.md` 调查的是 `~/openvino.genai`——它不在本目录下、也不是 clone 进来的，是本机已有的一个独立工作目录，报告里的检索命令要在那边执行，不是在 `~/agenticAI` 下。

## 六份调查报告（核心产出）

每份报告结构一致：逐节给出「检索命令 → 实际输出/源码片段 → 解读」，不接受"看起来应该有"这种结论，只接受"打开代码验证过"的结论。

| 报告 | 一句话结论 |
|---|---|
| [vllm_agentic_evaluation_investigate.md](vllm_agentic_evaluation_investigate.md) | 具备搭建 agentic 评测的全部零件（多轮 bench 工具、BFCL 负载数据集、量化精度框架），但零件互相割裂，无端到端 CI 覆盖"多轮+工具调用+优化组合" |
| [sglang_agentic_evaluation_investigate.md](sglang_agentic_evaluation_investigate.md) | 结构上比 vLLM 更领先（多轮压测原生集成、有真实校验多轮前缀增长的 nightly 功能测试），但精度侧殊途同归——真两轮工具调用测试写对了却是死代码 |
| [tensorrt_llm_agentic_evaluation_investigate.md](tensorrt_llm_agentic_evaluation_investigate.md) | 深度最高：唯一活跃的真两轮工具调用 CI、唯一把"缓存复用"当精度实验变量而非直接关掉；但看似最完整的 agent 循环测试打开后发现生成侧是 `DummyWorker` |
| [llama_cpp_agentic_evaluation_investigate.md](llama_cpp_agentic_evaluation_investigate.md) | 旗舰性能 CI 已停用近两年；精度评估只有困惑度/KL 散度，零自动化、且设计上答不了"优化是否损伤 agentic 能力"这个问题；但推理服务器原生支持 MCP，是六者里独一份 |
| [ollama_agentic_evaluation_investigate.md](ollama_agentic_evaluation_investigate.md) | 唯一在产品层解决"agentic 上下文持续增长"的系统（自动上下文压缩）；真实模型工具调用矩阵最广（21 个模型），但未接入公开 CI；自身零精度回归 |
| [openvino_genai_agentic_evaluation_investigate.md](openvino_genai_agentic_evaluation_investigate.md) | 六者里"优化×精度联合断言"方法论最严谨的一个（KV cache eviction 测试同一断言里既要求精度过线又要求压缩比达标）；但这套方法论只覆盖长文本 QA，没延伸到工具调用；`llm_bench` 同样是单请求微基准，无官方性能回归 CI |
| [cross_comparison_agentic_evaluation.md](cross_comparison_agentic_evaluation.md) | 六份报告的横向矩阵 + 四条跨系统重复验证的规律，含最终结论 |

跨系统反复验证的核心规律（详见横向对比报告第二节）：

1. 任何"看起来像完整 agent 循环"的测试，打开生成侧几乎必然发现是 mock/fixture/dummy 在扮演 LLM，或者验证目标另有所指（如 OpenVINO GenAI 的 `react_sample` 测试只验证跨语言输出一致性）
2. 量化/缓存类精度回归，几乎总是在关掉相关优化的前提下做的（TensorRT-LLM、OpenVINO GenAI 是例外，且 OpenVINO GenAI 更进一步把性能收益也写进了同一条断言）
3. 多轮/agentic 压测能力，云端 serving 系统普遍"仓库里有，官方性能指标里没有"；单机推理引擎/产品（llama.cpp、Ollama、OpenVINO GenAI）则是压根没有这个能力
4. 真实、有价值的测试普遍因为太慢/太贵，被排除在"日常必跑"的 CI 门禁之外（OpenVINO GenAI 的 KV cache eviction 测试是反例，真实模型+真实数据集依然稳定接在日常 CI 里）

**最终结论**：六个系统里，没有一个把"多轮会话 + 工具调用 + 主流性能优化组合开启"作为一个整体，纳入过日常自动化的 CI 精度回归。

## 审计脚本（辅助工具）

三个脚本对应调查的前两层取证——快速定位、留给人工深挖的起点，不能替代打开代码验证：

- **脚本1**（`1_bench_harness_audit.py <repo>`）：扫描 benchmark 源码，判断这个 harness 能不能生成"多轮 + session + 前缀共享 + agent/工具调用"形状的负载，给出 ★ 评级
- **脚本2**（`2_ci_config_audit.py <repo>`）：扫描 CI 配置，判断日常 pipeline 真正在守什么——尤其是"有没有 agentic/tool-call 精度 CI""优化开启状态下的精度是否被覆盖"
- **脚本3**（`3_guided_investigation.sh <repo>`）：交互式脚本，对着代码里的关键位置逐个提问，边读代码边记录结论，产出结构化 markdown

详细用法见 [QUICKSTART.md](QUICKSTART.md) 和 [AUDIT_README.md](AUDIT_README.md)。

**重要**：脚本 1、2 只能做到"有没有"（关键词/模式匹配），做不到"对不对"。六份最终调查报告里的核心结论，全部来自在脚本定位的基础上手动打开源码逐行验证——这是脚本无法替代的部分。

## 复现调查

被调查的五个云端/单机 runtime 仓库不纳入本仓库版本控制（见 `.gitignore`），需要自行 clone 到本目录下：

```bash
cd ~/agenticAI
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/sgl-project/sglang.git
git clone https://github.com/NVIDIA/TensorRT-LLM.git
git clone https://github.com/ggml-org/llama.cpp.git
git clone https://github.com/ollama/ollama.git
```

`openvino_genai_agentic_evaluation_investigate.md` 调查的是本机已有的 `~/openvino.genai`（[openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai)），不在本目录下、也不受本仓库的 `.gitignore` 管理，如需在别处复现，单独 clone 该仓库即可：

```bash
git clone https://github.com/openvinotoolkit/openvino.genai.git
```

克隆完成后，既可以用审计脚本先快速扫一遍，也可以直接参照对应报告里的检索命令逐条复现验证——所有报告里的命令都是可以直接照抄执行的。

## 局限性说明

- 调查基于某一时间点的代码快照（各报告开头标注了检出日期/commit），这个领域迭代很快，结论有时效性，复现前建议先确认自己 clone 的版本
- 报告结论聚焦"当前 CI/测试是否覆盖某个组合场景"，不代表对应功能本身不能正常工作——很多时候是"能力存在但缺少自动化回归"，而不是"能力有 bug"
