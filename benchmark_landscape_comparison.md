# Agentic AI Workload 基准测试详解与对比：MMLU / MT-Bench / MTR-Bench / BFCL / τ-bench

> 
> 承接 `capability_x_systems_rigor_matrix.md` 的横轴（能力金字塔）——本文档把横轴上简化成一个箭头的几个节点，逐个展开成详细档案，并补充横轴上原本缺失的两级（MTR-Bench、BFCL 的完整版本演进史），最后回答一个和前面所有调查一脉相承的问题：**这些基准测试有没有一个考虑过"系统/Runtime 层优化是否影响这些能力"？**

## 一、逐个基准详细介绍

### 1. MMLU（Massive Multitask Language Understanding）

- **提出**：Hendrycks et al., 2020
- **形式**：57 个学科（STEM、人文、社科、职业等），四选一客观选择题，零样本/少样本测准确率
- **考察什么**：静态知识背诵 + 基础推理，单轮、无上下文依赖
- **评测方法**：精确匹配（选项对不对），完全客观、可自动化
- **现状：已饱和**——前沿模型普遍超过 90% 准确率，区分度大幅下降；且题目本身长期公开，[大量证据显示原题或高度相似文本已经出现在 Common Crawl 训练语料里](https://www.digitalapplied.com/blog/llm-benchmark-methodology-2026-contamination-leaderboard-guide)，"背过"和"推理出来"难以区分。选择题格式本身也测不出生成式/解释性能力
- **继任者**：MMLU-Pro（1.2 万题、10 个选项而非 4 个、降低蒙对概率、题目更难），但[也已经开始重演饱和轨迹](https://intuitionlabs.ai/articles/mmlu-pro-ai-benchmark-explained)（前沿模型已逼近 90%），行业正在转向 LiveBench、Humanity's Last Exam 这类持续更新/超高难度基准

### 2. MT-Bench

- **提出**：Zheng et al., 2023，《[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)》
- **形式**：80 组两轮对话问题，覆盖 8 个能力类别（写作、角色扮演、信息抽取、推理、数学、代码、知识 I/STEM、知识 II/人文社科），每类别 10 题
- **考察什么**：**对话连贯性和指令遵循**——"第二轮有没有顺着第一轮的意思往下接"，是主观质量判断，不是客观任务正确性
- **评测方法**：LLM-as-judge（用 GPT-4 当裁判），两种模式：成对比较（哪个回答更好）或单答打分（1-10 分，依据"有用性/相关性/准确性/深度/清晰度"综合打分）。论文验证了 GPT-4 判分和人类判分的一致率能到 [80%-85%](https://www.emergentmind.com/topics/mt-bench-benchmarks)，接近人与人之间的一致率
- **配套基准**：Chatbot Arena（众包实时对战，没有固定题库，覆盖更广但不可控）
- **局限**：只有两轮、题库仅 80 题，偏小——催生了 MT-Bench-101（更细粒度的分层能力分类）和 MT-Bench++（扩展到 8 轮对话）
- **关键定位**：这是"能力金字塔"横轴上唯一一个用**主观 LLM 打分**而非客观判定的基准，衡量的是"顺不顺"，不是"对不对"

### 3. MTR-Bench（Multi-Turn Reasoning Benchmark）

- **提出**：Li, Bao et al., 2025（中国科学技术大学 / 阿里巴巴 / 新加坡国立大学），《[MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation](https://arxiv.org/abs/2505.17123)》
- **形式**：**4 大类、40 个任务、3600 个实例**，每个任务标定 3 个难度等级，任务需要和"环境"做多轮交互才能完成——不是对话质量评测，是**交互式推理正确性**评测
- **评测方法**：全自动化的 **Generator-Monitor-Evaluator 框架**——数据集构造和模型评测全流程自动化，不需要人工标注/打分（这是它和 MT-Bench 最本质的区别：MT-Bench 靠 LLM 裁判做主观判断，MTR-Bench 靠环境自动判定客观对错）。报告的指标不只是准确率，还包括**效率、无效响应率、推理模式出现频次**这类更精细的过程性指标
- **测试规模与发现**：评测了 20 个推理/非推理模型，o3-mini 综合表现最好；核心发现是**难度和轮次一起往上堆时，即便最前沿的推理模型也会显著掉分**——多轮交互式推理目前对所有模型都是短板，不只是小模型的问题
- **关键定位**：这是横轴上介于 MT-Bench 和 Tool-Calling 之间、容易被忽略的一级——它比 MT-Bench 更客观（自动判定，不靠 LLM 主观打分），比 Tool-Calling/BFCL 更聚焦"推理过程"而非"调用协议格式对不对"，是一个和 Tool-Calling 平行、而不是被它包含的能力维度

### 4. BFCL（Berkeley Function Calling Leaderboard，Gorilla 项目）

这是四个基准里唯一经历了完整版本演进、能直接体现"能力金字塔怎么一步步爬高"的例子：

| 版本 | 新增内容 | 评测方法 |
|---|---|---|
| **V1** | `simple`/`multiple`/`parallel`/`parallel_multiple` 及多语言变体（Java/JS/Python） | **AST 匹配**——比较生成的函数调用抽象语法树结构和标准答案是否一致，纯格式/参数正确性 |
| **V2（Live）** | 改用真实来源数据；新增 **relevance/irrelevance 检测**——判断模型在"没有一个工具真的相关"时会不会正确拒绝调用，`live_` 前缀系列类别 | 同样是 AST 匹配，但数据更真实、规模更大 |
| **V3（Multi-Turn）** | 新增 `multi_turn_base`/`miss_func`/`miss_param`/`long_context`/`composite`——真正的**多轮、带状态追踪**的工具调用场景 | 引入状态一致性检查，不再是单次调用格式对错 |
| **V4（Agentic，2025 ICML）** | 新增 **Web Search**（200 例，含真实场景下随机注入的 503/429/403 等 6 种编程访问错误，测多跳推理+错误恢复）+ **Memory**（465 例，测跨会话的持久化/用户专属状态管理）+ **Format Sensitivity**（[5 个维度的输入格式变体](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)，包括输出格式如 JSON vs Python 函数调用语法，测模型对格式扰动的鲁棒性） | Web Search + Memory 合计构成 665 例 "Agentic" 评测集 |

- **重要提醒**：官方明确说明**各版本分数不可直接跨版本比较**——V4 比 V1 难得多，同一个模型在 V1 上接近满分，放到 V4 的 Agentic 类别上可能大幅下滑
- **关键定位**：BFCL 本身就是一条从"纯格式检查"（V1）到"真正的 agentic 能力"（V4：会不会用外部工具获取信息、会不会维护跨会话状态、格式变了会不会崩）的完整演进链——它自己内部就是一个微缩版的能力金字塔

### 5. τ-bench / τ²-bench（Sierra Research）

- **τ-bench**：Yao et al., 2024，《[τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)》
  - **形式**：retail（零售）和 airline（航空）两个领域，**单向控制**——只有 Agent 能调用工具改变世界状态，用户只是被动提供信息
  - **评测方法**：判定标准是**任务终态**——把最终数据库状态和标注好的目标状态做比较，而不是看回答像不像人话；同时引入了 **pass^k** 指标
  - **pass^k 的精确定义**（容易被搞混，需要澄清）：pass^k 衡量的是"**k 次独立重复试验全部成功**的概率"，是比标准 pass@k（k 次里至少 1 次成功）严格得多的**可靠性/最坏情况**指标。数学上 pass^k 随成功率 p 呈 p^k 指数衰减——一个单次成功率 90% 的模型，到 k=8 时"每次都成功"的概率会跌到 57% 左右，这正是 pass^k 存在的意义：暴露 pass@1/pass@k 掩盖掉的不稳定性。[这个指标后来被 Anthropic 的模型卡片、CORE-Bench 等基准直接借用](https://hippocampus-garden.com/pass_k/)
- **τ²-bench**：2025，《[τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982)》
  - **新增 Telecom（电信）领域，建模成 Dec-POMDP（去中心化部分可观测马尔可夫决策过程）**——**双向控制**：Agent 和用户都可以用工具去改变同一个共享环境的状态，测的是**协调与沟通**能力，不只是"Agent 自己会不会用工具"
  - **配套的用户模拟器**是一处方法论亮点：论文报告电信领域的环境耦合用户模拟器错误率只有 **16%**（其中 6% 是严重错误），远低于此前单向控制基准里用户模拟器 40%-47% 的错误率——这本身也是一项可以被借鉴的评测基础设施改进
  - **实测数据**（论文原始报告）：GPT-4.1 在 retail/airline/telecom 三个领域的 pass@1 分别是 **74% / 56% / 34%**——从单向控制切到双向控制，能力断崖式下降；且论文明确指出**当前公开排行榜上头部模型已经刷到 98% 以上**，说明这个领域进步极快，读论文数字时要注意时间戳
- **关键定位**：这是横轴上唯一一个**同时具备"多轮 + 工具调用 + 终态精确判定 + 可靠性量化（pass^k）"**四个要素的基准，是 BFCL（测格式对不对）和 SWE-bench（测开放式任务能不能做完）之间承上启下的一级——本系列 `agentic_test_design_proposal.md` 里提出的测试方案，就是把这一级的方法论往下嫁接到 Runtime 自测层面

### 6. SWE-bench / SWE-bench Verified / SWE-bench Lite / SWE-bench Pro

前四节和 τ-bench 都被反复引用作为"能力金字塔"的参照系，但此前没有给 SWE-bench 单独的详细档案——这里补上，同时说明它和前五个基准在判分哲学上的根本差异。

- **提出**：Jimenez et al., 2023（Princeton），《SWE-bench: Can Language Models Resolve Real-World GitHub Issues?》
- **形式**：从 12 个主流 Python 开源仓库的真实 GitHub Issue + 对应 PR 构造任务——给模型一个真实 issue 描述和完整代码库，要求生成一个补丁（patch/diff）去解决这个 issue。构造管线是三阶段过滤：抓取约 9 万个 PR → 按"是否关联 issue、是否修改了测试"过滤 → 只保留能通过"fail-to-pass"执行验证的实例 → 最终产出 2294 个高质量任务
- **考察什么**：**开放式真实软件工程任务**——没有预设的动作空间，agent 可以做任何事（读文件、改代码、跑测试、调用工具），这是它和 BFCL/τ-bench 最本质的区别：BFCL/τ-bench 的任务空间和终态都是预先定义好的（函数签名、数据库目标状态），SWE-bench 完全开放，怎么解决问题、改几个文件、用什么策略都不设限
- **评测方法**：**FAIL_TO_PASS + PASS_TO_PASS** 双重单测门禁——每个任务配一组人工编写的真实单测，生成的补丁必须让"修复前失败、修复后应该通过"的测试（FAIL_TO_PASS）全部通过，同时不能让"修复前后都应该通过"的测试（PASS_TO_PASS）退化。评测跑在带固定依赖版本的 Docker 容器里，保证观测到的分数差异反映的是模型能力而不是环境差异
- **规模与变体**：
  - **SWE-bench**（原始）：2294 个实例
  - **SWE-bench Lite**：300 个实例子集（11 个仓库），成本更低，常用于快速迭代
  - **SWE-bench Verified**：OpenAI 联合 SWE-bench 团队于 2024 年推出的 500 个人工校验子集——原始数据集里部分任务描述模糊、单测本身有问题，导致"模型做错"和"任务本身不公平"分不清，Verified 专门过滤掉这些歧义任务，是目前跨模型对比的主流子集
  - **SWE-bench Pro**（2026，Scale AI）：1865 个多语言任务，专门应对数据污染问题——在 Verified 上刷到 80%+ 的模型，在 Pro 上只有 46%-57%，这个落差被认为是"更真实地反映当前能力"，不是模型变差了
- **当前状态**：[Verified 榜单头部竞争已经极度压缩](https://benchmarkingagents.com/swe-bench/)——2026 年中头部 6 个模型分差在 1.3 个百分点以内，接近评测脚手架（scaffolding）差异带来的噪声范围；[也有对排行榜可信度的直接质疑](https://www.digitalapplied.com/blog/swe-bench-verified-june-2026-benchmark-vs-scaffolding-analysis)——某统计站点上 100 个模型里只有 1 个带独立验证标记，其余 99 个是厂商自报分数，而每家用的评测脚手架（工具定义、重试逻辑、上下文管理）都不一样，跨厂商可比性存疑；Gemini 3.1 Pro 从 Verified 的 80.6% 掉到 Pro 的 54.2%，Opus 4.5 从 Verified 80.9% 掉到 Scale SEAL 公开集的 45.9%，这类"断崖式下跌"进一步说明 Verified 分数的参考价值正在被行业质疑
- **关键定位**：这是横轴上目前唯一一个**开放式、无预设终态**的基准——BFCL 判"函数调用格式对不对"、τ-bench 判"数据库终态是否等于标注目标"，都是收敛、可精确定义的判分空间；SWE-bench 判"补丁能不能让真实单测通过"，判分标准客观但任务空间完全开放，代表能力金字塔里"真实开放式任务"这一档。但如上所述，这一档目前正经历"数据污染争议 + 评测脚手架差异主导分数差"的方法论危机——**分数客观（单测过没过是二元事实），可比性存疑（不同厂商的脚手架能贡献远超模型本身的分数差异）**，这是它和 BFCL/τ-bench 相比一个独特的、值得警惕的风险点

## 二、横向对比表

| 基准 | 提出时间 | 轮次 | 考察核心 | 评测方法 | 规模 | 客观/主观 | 当前状态 |
|---|---|---|---|---|---|---|---|
| MMLU | 2020 | 单轮 | 知识背诵+基础推理 | 精确匹配 | 57 学科海量题 | 客观 | **已饱和**，被 MMLU-Pro 接棒（也已逼近饱和） |
| MT-Bench | 2023 | 2 轮 | 对话连贯性/指令遵循 | LLM-as-judge 1-10 打分 | 80 题 × 8 类别 | 主观（人类一致率 ~85%） | 已被认为"太小、太浅"，衍生出 MT-Bench-101/++ |
| MTR-Bench | 2025 | 多轮（交互式环境） | 多轮**推理**正确性 | 自动化 Generator-Monitor-Evaluator，测准确率+效率+无效率 | 4 类 40 任务 3600 实例 | 客观（自动判定） | 新基准，前沿模型随难度/轮次增加显著掉分 |
| BFCL v1→v4 | 2023→2025 | v1-v2 单轮 → v3-v4 多轮/跨会话 | v1 格式正确性 → v4 真实 agentic 能力（联网/记忆/鲁棒性） | AST 匹配（早期）→ 状态/结果匹配（v3+） | v4 Agentic 类 665 例 | 客观 | 持续演进中，各版本分数不可比 |
| τ-bench / τ²-bench | 2024/2025 | 多轮 | 工具调用任务**终态**是否正确 + 可靠性 | 数据库终态匹配 + pass^k | retail/airline/telecom 三领域 | 客观 | 头部模型进步极快（论文 34% vs 公开榜单已超 98%） |
| SWE-bench (Verified/Lite/Pro) | 2023→2026 | 开放式（无轮次概念） | 真实 GitHub issue **能否被实际解决** | 真实单测门禁（FAIL_TO_PASS + PASS_TO_PASS），Docker 沙箱 | 原始2294 / Verified 500 / Lite 300 / Pro 1865(多语言) | 客观（单测过没过是二元事实），但**跨厂商可比性存疑** | Verified 榜单头部已压缩至 1.3pp 内；Pro 上分数普遍腰斩，暴露污染/脚手架问题 |

## 三、放回横轴：这几个基准该怎么排进能力金字塔

`capability_x_systems_rigor_matrix.md` 原有横轴是 `MMLU/GSM8K → MT-Bench → Tool-Calling/JSON → τ-bench → SWE-bench` 五级，现在可以把 MTR-Bench 和 BFCL 的完整版本史嵌进去，让这根轴更精细：

```
X1          X2         X2.5           X3(v1→v4 完整演进)              X4              X5
MMLU   →  MT-Bench  →  MTR-Bench  →  BFCL v1(格式)→v2(相关性)→v3(多轮状态)→v4(联网/记忆/鲁棒性)  →  τ-bench/τ²-bench  →  SWE-bench
客观选择    对话连贯性    交互式推理      纯AST匹配        真实数据         状态追踪          真agentic能力         终态匹配+pass^k        开放式代码修改
（背诵）   （LLM主观打分） （自动化客观判定）                                                （BFCL 内部的能力金字塔）
```

一个值得注意的细节：MT-Bench 和 MTR-Bench 虽然名字相似（都叫"多轮"），但**评测哲学完全不同**——MT-Bench 靠 LLM 当裁判做主观打分，MTR-Bench 靠环境自动判定客观对错。这和此前六份调查报告里反复出现的"命名相似但含义不同"陷阱（SGLang 的 `multi_round`、vLLM 的伪多轮）是同一类提醒：看到"多轮"两个字，永远要往下多问一句"是靠什么判的对错"。

## 四、一个共同的能力盲区：五个基准没有一个测多 Agent

除了下一节要讲的"系统维度缺失"，这五个基准还共享另一处能力维度上的空白：**全部是单 Agent 评测**。

需要特别澄清一处容易误读的地方：**τ²-bench 的"双向控制（dual-control）"不是多 Agent**——它是 Agent + 用户模拟器两方都能用工具改变共享环境状态，第二方是"模拟的人"，不是"另一个 Agent"。同理，BFCL v4 的 Memory 类别测的是**单 Agent 的跨会话状态**，不是多 Agent 之间的状态共享。

| 基准 | 参与方 | 是否多 Agent |
|---|---|---|
| MMLU / MT-Bench / MTR-Bench | 单模型（MTR-Bench 有环境交互） | 否 |
| BFCL v1-v3 | 单 Agent + 工具 | 否 |
| BFCL v4 | 单 Agent + 工具 + 跨会话记忆 | 否（记忆是单 Agent 自己的） |
| τ-bench | 单 Agent + 被动用户 | 否 |
| τ²-bench | Agent + **主动用户**（双向控制共享环境） | 否（第二方是模拟的人） |
| SWE-bench | 通常单 Agent | 否 |

这意味着"多个 Agent 协作完成一个任务，最终结果对不对"这件事，**在能力金字塔的横轴上目前连权威基准都还没有**——比 SWE-bench 更靠右的位置是空的。这个空白与 `agentic_test_design_proposal.md` 里 G5 子目标的第三个维度（多 Agent 协作正确性）正好对应：那一维之所以"全场空白"，一部分原因就是连可借鉴的评测方法论都不存在。

## 五、和整个调查系列的连接点：这五个基准，没有一个考虑过"Runtime 层"

把这五个基准放回整个系列的核心问题里看，结论和之前完全一致：

**MMLU、MT-Bench、MTR-Bench、BFCL、τ-bench/τ²-bench——全部是纯粹的模型能力评测，没有一个基准的官方评测协议里包含"这个分数是在什么 serving 配置下跑出来的"这个变量。** 它们默认假设推理是精确的、确定性的（或者假设采样噪声是评测本身要处理的问题，而不是系统引入的），完全没有涉及：

- 跑评测时开没开前缀缓存/KV cache 复用
- 权重/KV cache 是否被量化
- 是否用了投机解码
- 评测跑在单个请求下，还是高并发多租户场景下（batch-size 依赖的数值不确定性此前有研究讨论过，但和这几个基准的官方协议无关）

这正好和 `capability_x_systems_rigor_matrix.md` 纵轴要补的东西对上——这些基准分数目前默认都是在"Y0：无系统维度"的假设下报告的。而六份调查报告揭示的现实是：六个 Runtime 里没有一个把 BFCL/τ-bench 这类基准跑在"前缀缓存+量化+投机解码同时开启"的配置下做过日常回归。也就是说：**即便模型能力金字塔上的每一级都有了对应的权威基准，这些基准的分数在实际生产配置下是否依然成立，仍然是全行业没有人验证过的事**——这也是 `agentic_test_design_proposal.md` 里那套测试方案存在的意义：不是要再发明一个新基准，而是要把这些已经权威的基准，**在 Runtime 的真实优化配置矩阵下**重新跑一遍。

## 六、当前实践：面向 Agent 框架产品的评测组合（与 Runtime 层测试是两个不同的评测对象）

需要先划清一个容易混淆的边界：本文档前五节讨论的基准，评估对象是**模型/Agent 框架的能力**；本系列其余文档（六份 `*_agentic_evaluation_investigate.md`、`agentic_test_design_proposal.md`、`capability_x_systems_rigor_matrix.md`）讨论的是**Runtime（推理引擎）层该怎么测**——这是两个不同的评测对象。评估 Copilot、AutoGen、OpenAI Agent、Claude Agent、Microsoft Agent Framework 这类"Agent 框架产品"时，用的正是本节讨论的基准组合，不是前面 Runtime 测试方法论那一套。

### 6.1 产品级评测的基础链路：MMLU → MT-Bench → BFCL → τ-bench

结合多份 2026 行业评测指南的独立结论，四层结构反复收敛到同一个排序：MMLU/MMLU-Pro（已饱和，前沿模型普遍 >88%，区分度低）→ MT-Bench（LLM-judge 主观打分，测"顺不顺"不是"对不对"）→ BFCL v4（call-level 静态测试：函数选对没有、参数填对没有）→ τ-bench/τ²-bench（多轮工具调用 + 状态维持 + 用户中途修改 + 判断任务何时真正完成）。[Spheron 2026 Tool-Calling Benchmark Guide](https://www.spheron.network/blog/tool-calling-benchmarks-bfcl-tau-bench-latency-optimization/)、[FutureAGI 2026 Evaluation Guide](https://futureagi.com/blog/evaluating-llm-systems-metrics-benchmarks-2026/)、[Kili Technology 2026 AI Benchmarks Guide](https://kili-technology.com/blog/ai-benchmarks-guide-the-top-evaluations-in-2026-and-why-theyre-not-enough) 三份独立资料给出同一结构，且都明确指出**τ-bench 比 BFCL v4 更难**——原因是"模型必须在多轮工具调用结果之间维持状态、处理用户中途修改、自己判断任务到底完成没有"，这和第三节 X3(BFCL)→X4(τ-bench) 的排序结论一致。"MMLU/MT-Bench 更多被视为基础模型能力评测，τ-bench+BFCL 是 agentic 核心组合"这个定位是成立的。

### 6.2 MTR-Bench 该放在哪：一个尚无共识的位置，不宜写成确定的一级

MTR-Bench **不在**上述几份 2026 行业综述的常规名单里——它是 2025 年的学术论文（中科大/阿里/新国立），目前没有证据表明前述产品级评测已经把它纳入常规流程。外部评测资料（[EmergentMind](https://www.emergentmind.com/topics/mtr-bench)）明确把它和 BFCL 定位成**测不同维度**：MTR-Bench 测"通用交互式推理"（迭代解题、自适应规划、按搜索空间大小精确校准的难度分级），BFCL 测"工具调用/函数调用准确性"，两者被外部资料称为"互补"（complementary），不是谁比谁更难的线性关系。因此把 MTR-Bench 摆进金字塔里 BFCL 和 τ-bench 之间的固定一级，是一种简化——更准确的表述是"学术界提出的补充维度，与 BFCL 平行，尚无产品级评测采用的证据"，而不是"已经和 BFCL/τ-bench 一起成为核心组合"。

### 6.3 金字塔里没有 SWE-bench：多数情况下是合理省略，不是遗漏，但它本身也有独立的可信度问题

2026 年的行业综述普遍把评测拆成**并行的簇**，而不是一条单一链路：知识推理（MMLU-Pro/GPQA/AIME）、代码（HumanEval/MBPP 已饱和，SWE-bench Verified 是当前前沿）、Agentic 工具使用（BFCL v4、τ-bench/τ²-bench）、数学（AIME-25/FrontierMath）。也就是说，**SWE-bench 和 τ-bench 是两个并行的顶层，分别对应"代码智能体"和"通用工具智能体"两个应用域，不是叠在同一条金字塔上的上下级关系**。落到具体框架上：Copilot 是明确的代码智能体，评测它需要单独补 SWE-bench Verified 这一层；AutoGen/OpenAI Agent/Claude Agent/Microsoft Agent Framework 更偏通用工具调用场景，τ-bench/BFCL 这条链路已覆盖主要需求，可以不强制引入 SWE-bench。

SWE-bench 的详细档案见第一节新增的第 6 条——这里额外提醒一点，如果确实需要把它补进某个 Agent 框架的评测组合（比如评测 Copilot 类产品）：**SWE-bench Verified 榜单目前的可信度正在被行业质疑**——2026 年中头部模型分差已压缩到 1.3 个百分点以内，且换到刻意规避数据污染的 SWE-bench Pro 上，同一批模型的分数普遍从 80%+ 腰斩到 46%-57%。原因不是模型能力真的差这么多，而是 Verified 上的高分很大程度由"评测脚手架"（工具定义、重试逻辑、上下文管理）而非模型本身贡献，且大部分公开分数是厂商自报、缺乏独立验证。这意味着：**如果引入 SWE-bench 作为评测组合的一部分，报告出来的绝对分数应该谨慎解读，优先看同一套评测脚手架下的相对排名，而不是不同厂商自报分数的直接对比**——这一点和本系列此前反复强调的"打开代码验证，而不是相信自述"的方法论是一致的。

### 6.4 修订后的组合建议

| 层级 | 基准 | 定位 | 采用现状 |
|---|---|---|---|
| 基座能力 | MMLU/MMLU-Pro | 知识/推理，已饱和 | 成熟共识，仅作背景参考 |
| 对话质量 | MT-Bench | 主观对话连贯性 | 成熟共识，仅作背景参考 |
| 工具调用正确性 | BFCL v4 | call-level 函数选择/参数正确性 | **核心组合**，多份 2026 行业指南独立收敛到同一结论 |
| 通用交互式推理（平行维度，非线性叠加） | MTR-Bench | 自适应规划、按难度分级的迭代解题 | 学术界提出，尚无产品级评测采用证据，建议标注为"观察中的补充维度" |
| 端到端任务完成度 | τ-bench / τ²-bench | 多轮状态维持 + 用户中途修改 + 任务终态判定 | **核心组合**，公认比 BFCL 更难 |
| 代码域专项（与上面平行，非叠加） | SWE-bench Verified | 开放式真实代码库修改 | 仅当评测对象含代码智能体（如 Copilot）时需要补充 |

**结论**：τ-bench + BFCL 是当前有多份独立行业资料收敛支持的"核心组合"，MMLU/MT-Bench 作为"基础模型能力参考"而非"agentic 核心指标"的定位也成立。MTR-Bench 值得关注，但更适合定位为"和 BFCL 平行的补充维度"而非金字塔里确定的某一级；SWE-bench 的取舍取决于评测对象是否包含代码智能体——这些是把原始表述放进正式文档前需要做的修正，避免把一个尚处学术阶段的基准的采用成熟度写得比实际更靠前，也避免把两个并行应用域的基准误写成单一线性顺序。

## 参考链接

- [MMLU/MMLU-Pro 现状与饱和问题](https://www.digitalapplied.com/blog/llm-benchmark-methodology-2026-contamination-leaderboard-guide)
- [MMLU-Pro 详解](https://intuitionlabs.ai/articles/mmlu-pro-ai-benchmark-explained)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena（原始论文）](https://arxiv.org/abs/2306.05685)
- [MT-Bench 方法论详解](https://www.emergentmind.com/topics/mt-bench-benchmarks)
- [MTR-Bench 论文](https://arxiv.org/abs/2505.17123)
- [Berkeley Function Calling Leaderboard V4 官方页面](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BFCL V4 Format Sensitivity 博客](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)
- [BFCL 版本演进详解](https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/bfcl/index.html)
- [τ-bench 原始论文](https://arxiv.org/abs/2406.12045)
- [τ²-Bench 论文](https://arxiv.org/abs/2506.07982)
- [τ²-bench GitHub 仓库](https://github.com/sierra-research/tau2-bench)
- [pass^k vs pass@k 详细辨析](https://hippocampus-garden.com/pass_k/)
- [Spheron: AI Agent Tool Calling Benchmarks — BFCL v4, tau-Bench, and Function-Call Latency Optimization (2026 Guide)](https://www.spheron.network/blog/tool-calling-benchmarks-bfcl-tau-bench-latency-optimization/)
- [FutureAGI: Evaluating LLM Systems — Metrics and Benchmarks (2026)](https://futureagi.com/blog/evaluating-llm-systems-metrics-benchmarks-2026/)
- [Kili Technology: AI Benchmarks 2026 — Top Evaluations and Their Limits](https://kili-technology.com/blog/ai-benchmarks-guide-the-top-evaluations-in-2026-and-why-theyre-not-enough)
- [EmergentMind: MTR-Bench — Interactive LLM Benchmark](https://www.emergentmind.com/topics/mtr-bench)
- [SWE-bench 原始论文：Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [BenchmarkingAgents: SWE-bench Verified Explained — 2026 Methodology, Tiers, Caveats](https://benchmarkingagents.com/swe-bench/)
- [DigitalApplied: SWE-bench in 2026 — Benchmarks vs Scaffolding Reality](https://www.digitalapplied.com/blog/swe-bench-verified-june-2026-benchmark-vs-scaffolding-analysis)
- [DEV Community: SWE-bench Scores and Leaderboard Explained (2026)](https://dev.to/rahulxsingh/swe-bench-scores-and-leaderboard-explained-2026-54of)
