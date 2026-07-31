# 一份"真实有效"的 Agentic 测试设计方案

> 
> 背景：对 vLLM、SGLang、TensorRT-LLM、llama.cpp、Ollama、OpenVINO GenAI 六个系统的调查（见同目录下六份 `*_agentic_evaluation_investigate.md`）发现一条没有例外的规律——**没有一个系统的测试，同时做对"真实编排 + 正确判分标准 + 优化配置组合下验证 + 对抗性输入下的鲁棒性"这四件事**。本文档在六次调查的基础上，给出一套可以直接落地的测试设计方案，目标是把六家各自做对的那一小块拼成一份完整的、可执行的测试蓝图。

## 一、测试目标

回答一个到目前为止没有任何一个系统的 CI 能回答的问题：

> **当前缀缓存、KV 量化、投机解码这类主流性能优化打开时，Runtime 处理"多轮会话 + 工具调用"这类 agentic 负载的任务成功率，是否仍然保持在可接受的水平？**

这个问题必须拆成三个子目标，缺一个都回答不了：

| 子目标 | 对应的验证内容 | 为什么不能省 |
|---|---|---|
| G1. 编排真实性 | 第二轮请求必须依赖第一轮真实推理产出的 `tool_call`，工具必须被真实/仿真执行，不能预置历史 | 否则测的是"编排代码写得对不对"，不是"模型决策对不对"——这正是六次调查里反复出现的假象 |
| G2. 判分正确性 | 断言"任务终态对不对"，不是"格式合不合法"或"有没有崩" | 格式合法≠任务做对；六次调查里大多数"看似端到端"的测试止步于格式/崩溃检查 |
| G3. 配置敏感性 | 同一套任务，在"优化关闭"和"优化组合开启"两种配置下都要跑，且判分标准在两种配置下都要过 | 否则测的只是"默认配置下能力还行"，回答不了"优化到底有没有偷偷破坏 agentic 能力"这个真正的风险点 |
| G4. 鲁棒性（对抗性输入下不能被拖垮） | 工具 schema 本身可能是恶意/畸形的（巨大枚举、灾难性回溯正则、深层嵌套 JSON Schema），必须验证"不崩溃、掩码计算耗时可控"，这和"任务做没做对"是独立的一个维度 | agentic 场景里工具 schema 常来自第三方/MCP server，不完全可信；G1-G3 验证的是精度，G4 验证的是可用性——一个系统可以精度很高，同时被恶意 schema 轻易拖垮，两者互不覆盖 |

G4 是四个子目标里现状最差的一个——六个系统里只有 vLLM 有相关代码（一处对应真实安全公告 GHSA-rwxx-mrjm-wc2m 的 ReDoS 防护单测 + 一处会**主动过滤掉**复杂 schema 的 serving 压测），且两者从未被拼在一起验证过；另外五家零命中。详见 3.6 节的具体设计。

## 二、六个系统的现状：谁做对了哪一块

在设计新方案之前，先精确定位每家现有实践的边界——这决定了"借鉴谁的什么、还缺什么"。

| 系统 | 对 G1（编排真实性）的现状 | 对 G2（判分正确性）的现状 | 对 G3（配置敏感性）的现状 |
|---|---|---|---|
| vLLM | ✗ `tests/tool_use/` 用写死的三消息 fixture，非真实两轮 | 部分：断言参数值精确匹配，但仅单轮 | ✗ 11/12 配置显式关闭前缀缓存 |
| SGLang | 写对了（`_test_function_calling_multiturn`），但唯一调用点被注释掉，从未执行 | 同上（未执行） | 默认开启缓存复用（未显式关闭），但相关精度测试关闭了 radix cache |
| TensorRT-LLM | ✓ 唯一一处确认活跃在 CI 里的真两轮（Harmony 格式），第二轮真依赖第一轮真实输出 | 部分：断言最终 content 非空，未做终态精确判定 | ✓ 唯一把 `enable_block_reuse` 当精度实验变量做 A/B 的系统 |
| llama.cpp | 部分：真实模型推理，但工具结果仍写死注入（`test_calc_result`） | ✓ 断言具体数值结果（0.56），是六者中唯一断言"数值对不对"的工具调用测试 | 未测试；且该测试标记 `slow`，默认 CI 门禁不跑 |
| Ollama | ✗ 真实模型但仅单轮（21 模型矩阵），未验证工具结果回填后的续答 | ✗ 只验证触发了工具调用，未验证后续任务终态 | 未测试；且未接入公开 CI |
| OpenVINO GenAI | ✗ `test_react_sample_refs` 真推理，但判定目标是"跨语言输出一致"而非任务对错 | ✗（判分目标错位） | ✓ 但仅用于长文本 QA 场景（`test_kv_cache_eviction`），未延伸到工具调用；同一断言里"精度阈值"与"性能收益阈值"绑定，是六者里方法论最严谨的参照系 |

结论很清楚：**G1 最佳实践在 TRT-LLM，G2 最佳实践在 llama.cpp，G3 最佳实践在 OpenVINO GenAI**——没有一家三个都占，但三块拼图确实都已经分别被人做出来过，说明这不是技术不可行，而是没有人把它们放进同一个测试里。

## 三、测试架构设计

### 3.1 任务集设计：需要终态可判定，不能用文本相似度

任务集设计的核心约束：**每个任务的"对不对"必须能被程序化判定，不能靠"看起来像不像"**。参考 τ-bench 的数据库终态匹配思路，但下沉到 Runtime 自测这个更轻量的场景：

```python
TASK_SUITE = [
    {
        "id": "single_tool_unit_conversion",
        "prompt": "帮我查一下上海今天的气温，如果超过30度就用华氏度告诉我",
        "tools": [get_weather],
        "real_tool_impl": lambda location: {"celsius": 32},
        "check": lambda final_text: "89.6" in final_text or "89.6°F" in final_text,
    },
    {
        "id": "calculator_with_real_computation",
        "prompt": "单位圆上30度角对应点的y坐标，保留两位小数",
        "tools": [calculate],
        "real_tool_impl": lambda expression: eval_safely(expression),  # 真实计算，不是预置返回值
        "check": lambda final_text: re.search(r"0\.5[56]", final_text) is not None,
    },
    {
        "id": "multi_tool_sequential_dependency",
        "prompt": "查一下北京时间现在几点，然后告诉我纽约现在几点",
        "tools": [get_current_time, convert_timezone],
        "real_tool_impl": {...},   # 需要两次工具调用，且第二次依赖第一次结果
        "check": lambda final_text: check_time_conversion_correct(final_text),
    },
    {
        "id": "no_tool_needed_negative_case",
        "prompt": "1+1等于几？",
        "tools": [get_weather, calculate],
        "real_tool_impl": {},
        "check": lambda final_text, tool_calls: len(tool_calls) == 0 and "2" in final_text,
        # 反例场景：验证模型不会滥用工具——这一条在六次调查里完全没人测过
    },
    {
        "id": "long_history_tool_recall",
        "prompt_sequence": [...],  # 多轮铺垫后，第 N 轮要求模型回忆并使用第 1 轮工具调用的结果
        "check": lambda final_text: check_recall_correct(final_text),
        # 专门验证：上下文压缩/驱逐之后，早期工具调用结果有没有被正确保留
    },
]
```

最后一条 `long_history_tool_recall` 是专门为了填补 Ollama 报告里指出的空白设计的——`compactor.go` 的自动压缩机制目前完全没有"压缩后信息是否保真"的评测，这条任务就是直接测这个。

### 3.2 编排层：真实两轮，工具真实/仿真执行

```python
def run_agentic_task(server, task):
    messages = [{"role": "user", "content": task["prompt"]}]
    tool_calls_made = []

    for turn in range(MAX_TURNS):
        resp = server.chat(messages, tools=task["tools"])       # 真实推理，不是mock

        if not resp.tool_calls:
            return resp.content, tool_calls_made                 # 模型认为任务已完成

        for tool_call in resp.tool_calls:
            tool_calls_made.append(tool_call)
            args = json.loads(tool_call.arguments)
            result = task["real_tool_impl"](**args)               # 真实/仿真执行，不是预置在请求里
            messages.append({"role": "tool", "content": json.dumps(result),
                              "tool_call_id": tool_call.id})

        messages.append(resp.message)

    raise TimeoutError(f"Task {task['id']} did not finish within {MAX_TURNS} turns")
```

关键设计点：
- 工具调用是否发生、调用几次、传什么参数，全部由**模型的真实推理**决定，测试代码只负责"如果模型调用了，就真的执行并把结果喂回去"
- 循环到模型自己决定不再调用工具为止（或超过轮数上限），而不是像 vLLM/llama.cpp 现有测试那样固定死"只测两轮"

### 3.3 配置矩阵：优化组合是第一等公民，不是可选项

```python
CONFIG_MATRIX = [
    {"name": "baseline",              "prefix_caching": False, "kv_quant": None,  "spec_decode": False},
    {"name": "cache_only",            "prefix_caching": True,  "kv_quant": None,  "spec_decode": False},
    {"name": "cache_plus_quant",      "prefix_caching": True,  "kv_quant": "fp8", "spec_decode": False},
    {"name": "cache_quant_specdecode","prefix_caching": True,  "kv_quant": "fp8", "spec_decode": True},
]

@pytest.mark.parametrize("config", CONFIG_MATRIX, ids=lambda c: c["name"])
def test_agentic_task_suite_under_config(config):
    server = start_server(**config)
    results = [run_agentic_task(server, task) for task in TASK_SUITE]
    success_rate = sum(check_task(r, task) for r, task in zip(results, TASK_SUITE)) / len(TASK_SUITE)

    # 关键：像 OpenVINO GenAI 的 test_kv_cache_eviction 一样，性能和精度在同一条断言里
    perf_metrics = collect_perf_metrics(server)
    assert success_rate >= MIN_SUCCESS_RATE[config["name"]]
    if config["name"] != "baseline":
        assert perf_metrics.speedup_ratio >= MIN_SPEEDUP[config["name"]]
```

`cache_quant_specdecode` 这一档——"缓存复用 + KV 量化 + 投机解码同时开启，且要求 agentic 任务成功率不掉"——**是六次调查里没有任何一个系统测过的组合**，也是这套方案要补的核心空白。

### 3.4 稳定性验证：单次跑通不算数

LLM 输出有随机性，任何单次通过的测试都可能是运气。参考 τ-bench 的 `pass^k`：

```python
def test_agentic_task_stability(config, k=4):
    """同一任务重复跑 k 次，全部成功才算通过——而不是跑一次就下结论"""
    server = start_server(**config)
    for task in TASK_SUITE:
        successes = [check_task(run_agentic_task(server, task), task) for _ in range(k)]
        pass_at_k = all(successes)
        record_metric(f"{task['id']}_pass^{k}", pass_at_k)
    # CI 门禁可以设定："pass^1 必须 100%，pass^4 允许一定比例失败"这种分级阈值
```

### 3.5 缓存生效的独立验证：不能只信任配置开关

借鉴 SGLang `test_gsp_multi_turn` 的做法——解析服务端真实日志，验证"配置说开了缓存"和"缓存真的被命中了"是两回事，不能只凭配置项就下结论：

```python
def test_cache_actually_reused_during_agentic_task():
    server = start_server(prefix_caching=True)
    run_agentic_task(server, TASK_SUITE[0])
    logs = server.get_request_logs()
    assert any("cache hit" in log or log.get("cached_tokens", 0) > 0 for log in logs), \
        "配置开启了前缀缓存，但日志显示 agentic 会话过程中一次命中都没有——说明这项优化对这类负载实际没生效"
```

这一条特别重要：如果不做这个验证，"任务成功率在开缓存后没掉"这个结论可能只是因为**缓存压根没被命中**（比如工具调用改变了 prompt 结构导致前缀失配），那测试通过只是虚假的安全感。

### 3.6 G4 鲁棒性验证：对抗性 schema 不能拖垮 Runtime——这是当前最大的空白，六家里只有一半的零件

不同于 3.1-3.5（验证"任务做没做对"），这一条验证的是"面对恶意/畸形输入，系统会不会被拖垮"，判分标准是**可用性**而不是**正确性**。当前六个系统里只有 vLLM 有相关代码，且拆成了两个从未拼在一起的碎片：一处是对应真实安全公告（GHSA-rwxx-mrjm-wc2m）的 ReDoS 超时防护单测，但断言的是"超时包装器逻辑对不对"，背后用 mock 顶替了真实 xgrammar 编译器；另一处是真实的 serving 级结构化输出压测，但代码里显式 `dataset.filter(...not has_xgrammar_unsupported_json_features(schema))`，主动把复杂/极端 schema 过滤掉了。这套方案要把这两个零件拼起来，并推广到真实并发场景：

```python
ADVERSARIAL_SCHEMAS = [
    {"id": "catastrophic_backtracking_regex", "type": "regex", "value": r"(a+)+b"},
    {"id": "huge_enum", "type": "json_schema", "value": {"type": "string", "enum": [f"opt_{i}" for i in range(50_000)]}},
    {"id": "deeply_nested_json", "type": "json_schema", "value": build_nested_schema(depth=500)},
    {"id": "malformed_tool_definition", "type": "tool_schema", "value": {"type": "function", "function": {"parameters": {"type": "object", "$ref": "#/definitions/self"}}}},  # 自引用畸形schema
]

@pytest.mark.parametrize("schema_case", ADVERSARIAL_SCHEMAS, ids=lambda s: s["id"])
def test_structured_decoding_survives_adversarial_schema(schema_case, concurrency=50):
    server = start_server()  # 真实引擎，不 mock
    start = time.monotonic()
    # 真实并发发起请求，而不是单条 mock 调用——模拟真实场景下"一个恶意 schema 混进正常流量"
    results = run_concurrent_requests(server, schema_case, concurrency=concurrency, timeout=10)

    assert all(r.status in ("success", "graceful_rejection") for r in results), \
        f"{schema_case['id']} 导致请求既没有成功完成也没有被优雅拒绝（可能是挂起/崩溃）"
    assert time.monotonic() - start < MAX_ACCEPTABLE_LATENCY_S, \
        f"{schema_case['id']} 的掩码计算耗时超过可接受阈值，可能引发级联排队"
    # 关键：验证这一个恶意请求没有拖累同批次的正常请求（隔离性）
    normal_results = run_concurrent_requests(server, NORMAL_SCHEMA, concurrency=concurrency)
    assert median_latency(normal_results) < BASELINE_MEDIAN_LATENCY_S * 1.5, \
        "对抗性 schema 请求拖慢了同批次的正常请求，说明缺乏请求级隔离"
```

关键设计点，对应两处现有碎片各自的缺口：
- 用**真实引擎**跑对抗 schema（补 vLLM 安全单测用 mock 顶替真实编译器的缺口），而不是只验证超时包装器逻辑
- **不过滤**极端 schema，反而把它们当作专门的测试输入（补 vLLM 压测脚本主动过滤掉极端 case 的缺口）
- 加一条其他任何现有测试都没做的检查：**隔离性**——一个恶意请求不能拖累同批次的正常请求延迟，这是多租户 serving 场景下真正的风险点（对应到 agentic 场景就是：一个 Agent 的恶意/写坏的工具 schema，不应该拖垮同一 Runtime 上其他 Agent 会话的响应速度）

## 四、CI 分层落地策略

六次调查里反复出现的"规律 4"——真实测试因为太慢太贵被移出日常 CI——决定了这套方案如果不做分层，大概率会重蹈覆辙。参考 llama.cpp（`slow` marker + schedule 触发）和 TRT-LLM（L0/L1/L2 分级）的思路：

| 层级 | 触发时机 | 任务集规模 | 配置矩阵规模 | 模型规模 |
|---|---|---|---|---|
| **PR 门禁（必跑）** | 每次 PR | 精简至 3-5 个代表性任务（覆盖单工具/多工具/反例场景） | 仅 `baseline` + 1 个"全优化开启"档 | 1 个小模型 |
| **Nightly** | 每日定时 | 完整任务集（含长历史召回场景） | 完整配置矩阵 | 2-3 个不同规模/家族模型 |
| **Release 前** | 发布前手动触发 | 完整任务集 + pass^k 稳定性验证（k≥4） | 完整配置矩阵 | 参考 Ollama 21-模型矩阵的广度做法 |

PR 门禁层必须足够便宜才能真正被日常执行——这是这次六个系统调查里最大的教训：写得再好的测试，如果因为慢/贵被人手动移出默认路径，效果等于没写。

G4（鲁棒性）的分层策略要单独说明：单条 schema 的超时/崩溃防护（对应 vLLM 现有的那处安全单测）足够便宜，应该放进 **PR 门禁**；但完整的对抗 schema 集合 × 真实并发 × 隔离性验证（3.6 节的完整版本）成本更接近一次小型压测，应该放进 **Nightly**——这条正是"意识到问题存在"（PR 门禁）和"验证问题在真实负载下确实被解决"（Nightly）分开验证的具体案例。

## 五、借鉴清单（每一块具体来源与要补的缺口）

| 设计元素 | 借鉴自 | 需要补上的缺口 |
|---|---|---|
| 真实两轮编排（G1） | TensorRT-LLM `_test_openai_chat_harmony.py::test_tool_calls` | 从"仅 Harmony 格式"扩展到通用 tool-call 协议；从"仅两轮"扩展到"循环到模型自己收尾" |
| 工具结果真实执行而非预置 | 无人做到（llama.cpp `test_calc_result` 最接近，但结果仍硬编码在请求里） | 需要新建——这是当前最大的空白 |
| 终态精确判定（G2） | llama.cpp `test_calc_result`（断言具体数值） | 从单一任务扩展到任务集，覆盖单工具/多工具/反例/长历史召回 |
| 优化配置作为精度实验变量（G3） | TensorRT-LLM（`enable_block_reuse` A/B）+ OpenVINO GenAI（性能阈值与精度阈值同断言） | 从"单一优化项"扩展到"多优化组合"（缓存+量化+投机解码同时开） |
| 服务端日志核验缓存真实命中 | SGLang `test_gsp_multi_turn` | 从"验证前缀结构增长"扩展到"验证 agentic 场景下缓存确实被命中" |
| 真实模型广度覆盖 | Ollama（21 模型矩阵） | 从"仅单轮触发验证"扩展到"完整任务集 + 配置矩阵" |
| 稳定性验证（pass^k） | τ-bench 方法论（外部基准，非六个系统自带） | 六个系统里都没有对自己的工具调用测试做重复稳定性验证，需要引入 |
| CI 分层，防止真实测试被移出日常路径 | llama.cpp（`slow` marker）、TensorRT-LLM（L0/L1/L2） | 需要明确设计"最小可日常跑的子集"，而不是做完整套件后因为慢被搁置 |
| 对抗性 schema 鲁棒性验证（G4） | vLLM `test_regex_compilation_timeout.py`（安全公告驱动的防护意识）+ vLLM `benchmark_serving_structured_output.py`（真实并发压测框架） | 前者要去掉 mock、换成真实引擎；后者要去掉过滤逻辑、专测极端 case；且两者从未被拼在一起，还要新增"隔离性"验证（一个恶意请求不能拖累同批次正常请求） |

## 六、这套方案本身的局限

- **工具的"真实/仿真执行"需要额外的沙箱/桩实现**，这部分工作量不小，尤其涉及文件系统、网络请求类工具时需要考虑隔离与可重复性——建议先从纯计算型工具（无副作用、结果确定）开始覆盖，再逐步扩展到有状态工具
- **终态判定逻辑（`check` 函数）本身需要为每个任务单独设计**，不像文本相似度那样可以复用一套通用打分器——这是这套方案比现有"相似度评分"类方法更扎实、但也更需要前期投入的地方
- **pass^k 稳定性验证会显著增加运行时间**（k 倍），需要配合第四节的分层策略，只在 nightly/release 层做，不能塞进 PR 门禁
- 本方案聚焦"Runtime 层能验证的工具调用正确性"，不覆盖"整个 agent 任务是否最终完成"这类需要完整 agent 框架/沙箱的评测（如 SWE-bench）——那是更上层的验证责任，超出 Runtime 自测的合理范围，但两者应该衔接：Runtime 层测试通过是上层 agent 评测有意义的前提
- **G4 的对抗 schema 集合本身需要持续维护**——新的约束解码后端（xgrammar/outlines/llguidance）可能有各自不同的性能陷阱，3.6 节给出的四个样例不是穷尽列表，需要参考安全社区已披露的 ReDoS/DoS 模式持续补充，且不同引擎的"可接受耗时阈值"需要分别标定，不能用一套阈值套所有后端
