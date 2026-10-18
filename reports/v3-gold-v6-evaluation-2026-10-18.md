# fd-sentiment v3 满样本金标评测（gold-v6，2026-10-18）

> 60 条金标、54 条成功打分、6 条失败。prompt v3、model `deepseek-v4.1-flash`、
> `max_tokens=32000`、串行。**这是该服务第一次在满样本上跑出可判定的评测。**
> 此前归档记录里的「v3 命中率 0.591」是本地 22/60 样本的结论（38 条因失败缺失），
> 统计上不成立；本报告取代它。

## 一、为什么要重跑

生产库 `/data/fd_sentiment.db` 里 `gold_set` 0 条、`eval_runs` 0 条——三代 prompt 的全部分数与评测**从没进过生产库**，只存在于各人本地。换机器即失忆，线上没有任何"准不准"的证据。

本次用 `scripts/build_gold_set.py` 同一批 12 家公司、轮转采样重建 `gold-v6`（60 条：bullish 24 / neutral 24 / bearish 12），标注仍走显式人工规则表（非模型输出）。正文以 sha256 + 长度登记在 `gold/gold-v6.json`，可回源核对；`as_of` 跨 2004-03-11 → 2026-09-21，顺带覆盖 point-in-time。

## 二、三种口径的结果（必须并列看，单独看任一个都会误判）

| 口径 | 结果 | 说明 |
|---|---|---|
| **exact hit**（五档完全一致） | **22/54 = 40.7%** | 最严口径。单看这个会得出"不可用"的错误结论 |
| **±1 档内** | **53/54 = 98.1%** | 高度集中在相邻档 |
| **方向**（正/中/负三分类） | **43/54 = 79.6%** | 实用口径 |
| **校准** | **失败** | 高置信桶 0.333 < 低置信桶 0.615 |

校准分桶明细：

| 置信桶 | 样本 | 命中 | 准确率 |
|---|---|---|---|
| 0.50–0.70 | 13 | 8 | 0.6154 |
| 0.70–0.85 | 17 | 6 | 0.3529 |
| 0.85–1.00 | 24 | 8 | 0.3333 |

**高置信桶不如低置信桶 → `confidence` 字段不可用于加权决策。** 这一条直接决定下游怎么消费分数（见下）。

## 三、偏离结构：不是乱判，是标定偏移

全部 32 条偏离，集中在两个方向相反的模式：

| expected → predicted | 条数 | 性质 |
|---|---|---|
| `bullish → strong_bullish` | 13 | 过强 |
| `bearish → strong_bearish` | 8 | 过强 |
| `bullish → neutral` | 8 | 过弱（全是权益分派「方案」类） |
| `neutral → bearish` / `→ strong_bearish` | 3 | 过弱 |

按期望档看命中结构：

```
bullish   n=23 hit=2   → strong_bullish 13, bullish 2, neutral 8
neutral   n=19 hit=16  → neutral 16, bearish 2, strong_bearish 1
bearish   n=12 hit=4   → strong_bearish 8, bearish 4
```

- `bearish` 的 4 条命中全是精确档，另 8 条一致地跳到 `strong_bearish`——**方向全对，强度升一级**。
- `neutral` 16/19，是全套最准的档。
- `bullish` 最差（2/23），但差的形态是不同的：13 条过强 + 8 条掉到 neutral。

### 那 8 条 `bullish → neutral` 需要单独说

逐条看，**8 条全部是权益分派「方案」类**（分红方案、调整每股分红金额、回购价格上限调整）。同期 6 条「实施公告」标 `neutral` 的，模型 6/6 全判 neutral。

即：**模型对"实施=事务性"的理解完全正确，只是不认同"方案=利好"**——方案本身只是宣告既定政策的分多少钱，没有超预期的增量信息。

builder 的标注规则注释里写着「权益分派只有'方案'公告才带方向信息（首次宣布提高/调整分红是回报政策信号）；'实施公告'是事务性执行通知」，而同一份注释里也记着另一条被**模型纠正过**的标注错误（"v2 评测实证：把实施公告标成 bullish 是标注错误，11 条全被模型正确判成 neutral，模型对、金标错"）。

**本次不动金标。** 为通过评测而调标注会让评测变成自我实现。把「方案」类按事务性重标后 exact hit 会从 40.7% 升到 53.7%——这个数字仅作为"金标口径可争议"的证据记录，不作为成绩。

## 四、6 条失败样本

| 类型 | 条数 | 原文摘要 |
|---|---|---|
| 拒绝打分（理由成立） | 3 | 股东会法律意见书、「仅陈述会议召集」等——**正确拒绝，不是缺陷** |
| 网关不可达（读超时） | 1 | 重试 2 次后失败 |
| 其余 | 2 | 同属超时/拒绝类 |

**拒绝打分应当算作能力而非失败**：纯表格、法律意见书被正确识别为无方向性语句并拒绝，比返回一个可疑分数有价值。评测器目前把 `refused` 计入 failures 而不计入命中分母——这个处理是对的（不拿拒绝样本摊薄或注水命中率），但报告里应把它与真失败分开计数。

## 五、两次被误归因的失败（教训）

### 5.1 「模型输出没有可解析 JSON」不是 prompt 问题，是 token 预算

报错文本长这样，极易误导：

```
LLMError: 模型输出中没有可解析的 JSON 对象：'We need answer JSON only. Need analyze...'
```

看起来像模型不听话、在输出英文思考。实际根因是 `deepseek-v4.1-flash` 的 `reasoning_content` 极长，`max_tokens` 被推理吃光后 `content` 空或只剩半截推理，`chat()` 回退取 `reasoning_content` 再解析失败。受控实验：

| max_tokens | 结果（8 条） |
|---|---|
| 16000 | 7 clean + 1 reasoning-only |
| 32000 | **8/8 clean** |

默认值原为 4000，生产 env 是 16000——**两者都偏小**。已修为默认 32000（env 可覆盖，commit `067a6e7`）。此前归档记录把这类失败归因为「网关 503」，是误判；真正的网关失败（读超时）本次只占 1 条。

### 5.2 「网关限流」也要分清是哪一种

本次 54 条串行跑完约 60-70 分钟（单条 58-73 秒）。中途曾怀疑并发导致失败，做了串行 vs 并发 3 的对照：**两种都没有出现 reasoning 泄漏**，差异只在超时数（串行 2 次、并发 3 时 4 次）与耗时（847s vs 342s）。所以并发不是推理泄漏的原因，token 预算才是。

## 六、成本事实（design 未记，本次补）

- 单条打分 58-73 秒（max_tokens=32000）；16000 时约 58s、32000 时约 73s。
- `evaluate.py` 是串行 `for` 循环，无并发 → **60 条满样本评测 = 60-70 分钟**。
- 这意味着「改一次 prompt 就跑一次评测」的迭代成本约一小时。若要提效，方向是在 evaluate 引并发（会改变执行语义与失败率，需单独评估），不是降 max_tokens。

## 七、结论与下游消费口径

**服务可用；v3 prompt 的单点质量可信**（方向 79.6%、±1 档 98.1%、正确拒绝无方向语句、正确判「实施=事务性」）。

**但五档 exact 与 confidence 加权都不可用**（40.7%、校准失败）。所以：

1. **下游只应消费「方向」（正/中/负）**，不消费五档精档、不消费 `confidence` 做加权。
2. 「拒绝打分」是一等产出：计入覆盖面，不当 neutral、不打 0 分、不重试到出结果。
3. exact 档位要达标需 prompt v4，成本是每次评测 +1 小时；**不建议为凑 exact 而调金标**。
4. **金标与评测结果必须落生产库**——这是本次暴露的最大流程缺口：不落库等于没评。

## 八、复现方式

```bash
# 1) 建金标（巨潮真实公告，显式人工规则标注）
python3 scripts/build_gold_set.py --set-version gold-v6 --target 60 --db /tmp/gold-v6.db

# 2) 满样本评测（必须设 MAX_TOKENS，否则大量误失败）
FD_SENTIMENT_MAX_TOKENS=32000 \
FD_SENTIMENT_LLM_BASE_URL=https://token.finddatatech.cloud/v1 \
FD_SENTIMENT_LLM_API_KEY=<gateway key> \
FD_SENTIMENT_MODEL=deepseek-v4.1-flash \
python3 - <<'PY'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'../fd-cn-report')
from fd_sentiment.evaluate import run_eval
from fd_sentiment.llm import LLMClient, LLMConfig
from fd_sentiment.prompt import load_prompt
from fd_sentiment.storage import Store
run_eval(LLMClient(LLMConfig.from_env()), Store('/tmp/gold-v6.db'),
         set_version='gold-v6', prompt=load_prompt('v3'))
PY
```

金标元数据（doc_id / as_of / expected_band / 正文 sha256 / 字符数）见 `gold/gold-v6.json`。
