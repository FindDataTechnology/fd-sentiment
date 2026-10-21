#!/usr/bin/env python3
"""金标集构建：≥50 条已标注公告，五档齐全。

标注原则（写在这里，避免"标注者口径随心情变"）：
  - 以**公告对相关主体经营/财务的影响方向**为准，不以市场反应为准。
  - 强度档按**正文量化锚点**判定，与打分 prompt 的强档门槛使用同一套判据：
    正文载有净利/营收同比 ≥50% 的明确数字，或出现「大幅上升/大幅增长/
    显著增长/大幅下降/大幅亏损」等强度词，且理由指向经营实质
    （销量/价格/毛利率/订单/产能/行业地位）→ 强档。
  - 一次性损益（政府补贴、资产处置、公允价值变动、税收返还）驱动的增长
    **不进强档**，最高到偏档。
  - 纯事务性（会议通知、章程修订、分红**实施**公告）或正负相抵 → neutral。
  - **不看标题判强度**：标题是否有"大幅"字样不构成强度依据。实测
    gold-v6 的 13 条误判全部源于此——标题无强度词但正文写着 +50%~+500%，
    金标按标题判 bullish，打分器按正文判 strong_bullish，两边各按自己的
    规则做对了，却在同一张评测表上对撞。

用法：
    python3 scripts/build_gold_set.py --set-version gold-v1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fd-cn-report"))

# 标注规则表：按 (类别, 标题关键词) → band。这是**人工标注的显式规则**，
# 不是模型输出——金标的独立性靠这条保证（打分器不能自己给自己出题）。
#
# 每条规则后标注其依据 prompt 的哪一条（sentiment-gold-coverage：标注依据
# 与判据一致性）。档位判定不得没有出处：
#   [强档门槛] bands.md 第一档：方向明确 + 量化锚点（≥50% 或强度词）+ 经营实质
#   [中性定义] bands.md neutral 条：纯事务性 / 正负相抵
#   [拒答边界] system.md「何时拒答」：只对读不出内容者拒答（不属于标注范畴）
RULES: list[tuple[str, tuple[str, ...], str]] = [
    # [同比方向] 预盈/扭亏：语义是「从亏到盈」，方向与强度都由同比决定。
    # 必须排在「预增」之前——标题同时含「预盈」与「增长」时，按后者会走
    # 普通增长分支，把「上年亏 12 亿、本期盈 6 亿」这种扭亏判成偏档（实测 2 条）。
    ("业绩预告", ("预盈", "扭亏"), "__trend__"),
    # [强档门槛] 标题命中"预增"等只是方向；强度再交给 _strengthen 按正文判
    ("业绩预告", ("预增", "大幅上升", "同向上升", "增长"), "bullish"),
    # [同比方向] 预亏/预减的**方向由同比变化决定，不由标题决定**。
    # 标题「业绩预亏公告」是法定披露类别名，不是情绪表述：一家公司从亏 386 亿
    # 收窄到亏 9 亿，公告标题仍叫「预亏」，但它对主体的影响是显著改善。
    # 一律按「预亏→负面」会把 10 条亏损收窄的公告错标成强负面（gold-v8 实测）。
    # 故此处只给「待判方向」标记，实际档位由 _trend_band() 按本期 vs 上年同期判。
    ("业绩预告", ("预减", "预亏", "首亏", "续亏", "大幅下降", "同向下降"), "__trend__"),
    # 风险提示类：几乎全是负面（诉讼/处罚/退市风险/资金占用），用来补负面档
    ("风险提示", ("退市", "暂停上市", "重大违法", "资金占用", "违规担保", "立案"), "strong_bearish"),
    ("风险提示", ("诉讼", "处罚", "警示", "问询"), "bearish"),
    # [中性定义] "股票交易异常波动"本身只是披露义务触发，方向取决于同期消息面 →
    # 归中性（不能把披露义务当利空，这是标注口径问题，不是模型问题）。
    ("风险提示", ("异常波动", "风险提示"), "neutral"),
    # [中性定义] 权益分派：**方案公告带方向信息**（首次宣布或上调分红金额是
    # 回报政策信号），**实施公告是事务性执行通知**（按既定方案发钱）。
    # 2026-10-19 裁决：口径分歧已由用户拍板——方案判 bullish，并已同步
    # prompt v5 的 bands.md（v4 及以前把两者都归 neutral，导致 8 条对不上）。
    # v2 评测实证补充：把"实施公告"标成 bullish 是标注错误（11 条全被模型
    # 正确判成 neutral），模型对、金标错。
    ("权益分派", ("方案", "调整"), "bullish"),
    ("权益分派", ("实施",), "neutral"),
    # [中性定义] 纯事务性公告 → 中性（补 neutral 档）
    ("股东大会", (), "neutral"),
    ("董事会", (), "neutral"),
    ("监事会", (), "neutral"),
    ("公司治理", (), "neutral"),
]

# 亏损收窄幅度达到此比例才认定「显著改善」。低于它（含小幅扩大/基本持平）
# 视为方向不明 → neutral：公司仍在亏损，改善幅度不足以构成方向性利好。
# 依据 prompt v5 的强档门槛（≥50%）与「宁可少说，不可多说」原则。
TREND_IMPROVE_PCT = 50.0
# 亏损扩大超过此比例 → 强负面；小于它 → 偏负面（温和恶化）。
TREND_WORSEN_PCT = 50.0


def rule_sources() -> list[tuple[str, str, str]]:
    """规则 → 依据的对照表，供审计与测试核对（每条规则都有出处）。"""
    src = "[强档门槛]"
    neu = "[中性定义]"
    return [
        ("业绩预告/预增类", "bullish→_strengthen", src),
        ("业绩预告/预减类", "bearish→_strengthen", src),
        ("风险提示/退市类", "strong_bearish", src),
        ("风险提示/诉讼类", "bearish", neu),
        ("风险提示/异常波动", "neutral", neu),
        ("权益分派/方案", "bullish", neu),
        ("权益分派/实施", "neutral", neu),
        ("股东大会", "neutral", neu),
        ("董事会", "neutral", neu),
        ("监事会", "neutral", neu),
        ("公司治理", "neutral", neu),
    ]

# 强度词。**不含「重大」**：公告开头的法定免责套话「保证本公告内容不存在
# 任何虚假记载、误导性陈述或者重大遗漏」几乎每份公告都有——实测 gold-v7 的
# 90 条里 72 条的前 400 字符含这种「重大」，于是这个判据对几乎所有公告恒为真，
# 任何业绩预减都被标成 strong_bearish（gold-v7 里 bearish 只剩 2 条就是这么来的）。
# 强度词必须表达**方向 + 强度**，不能是套话里的形容词。
STRONG_WORDS_BULL = ("大幅增长", "大幅上升", "显著增长", "显著上升", "翻倍", "激增",
                     "大幅提高", "大幅增加")
STRONG_WORDS_BEAR = ("大幅下降", "大幅下滑", "大幅亏损", "显著下降", "显著下滑",
                     "大幅减少", "巨额亏损")
# 兼容旧调用：任一方向的强度词
STRONG_WORDS = STRONG_WORDS_BULL + STRONG_WORDS_BEAR

# 法定免责套话：判定强度前必须从正文里剔除，否则「重大」「完整」这类词
# 会被当成本文自己的强度表述。实测 90 条中 72 条前 400 字含「重大遗漏」。
BOILERPLATE = (
    "重大遗漏", "虚假记载", "误导性陈述", "真实、准确、完整", "真实性、准确性和完整性",
    "准确性、完整性", "连带责任", "连带的法律责任", "敬请广大投资者注意投资风险",
    "不存在影响本次业绩预告内容准确性", "本公告内容不存在任何",
    # 财报格式语：出现在几乎每份业绩预告里，不构成本文对利空来源的判断
    "扣除非经常性损益后的", "扣除非经常性损益事项后", "扣除非经常性损益的",
)

NEUTRAL_TITLE = ("通知", "章程", "制度", "议事规则", "管理办法", "股东大会", "董事会决议")

# 一次性损益：由这些来源驱动的增长不构成"经营实质改善"，不进强档。
# 与 prompt v5 的强档第 ③ 条一致（理由须指向经营实质）。
ONE_OFF_SOURCES = ("政府补助", "政府补贴", "资产处置", "处置收益", "公允价值变动",
                   "税收返还", "债务重组", "投资收益", "非经常性损益")
# 经营实质来源：出现这些词说明增长来自主业。
OPERATIONAL_SOURCES = ("销量", "销售", "价格", "毛利率", "订单", "产能", "产量",
                       "营业收入", "主营业务", "市场份额", "行业地位", "成本")

# 同比增幅的量化门槛，与 prompt v5 bands.md 的 50% 锚点一致。
STRONG_PCT_THRESHOLD = 50.0

# 弱档的最小可信样本量：低于此值的档位在报告中标为「样本不足」。
WEAK_BAND_MIN = 10


def _max_yoy_pct(text: str, direction: str | None = None) -> float | None:
    """从正文里取最大同比变化幅度（百分数）。

    中文公告的同比表述有四种常见写法，缺一种就会漏掉整批样本：

      · 「同比增长 62%」          —— 动词直接接数字（覆盖最多）
      · 「将增加 88.78 亿元或 124.67%」 —— **金额在前、百分比用「或」引出**
      · 「增幅约 68%」            —— 名词「增幅」
      · 「增长 50%以上」          —— 带后缀

    第二种最容易漏：兖矿、紫金这类公告几乎全部用「增加 X 亿元或 Y%」，
    旧正则只认动词后紧跟数字，实测 50 条 `pct_up` 为空的样本里有多条属于此类。

    `direction` 给出时只取该方向的表述：一份亏损公告正文里常同时写着
    「营业收入增长 30%」，不区分方向会把利好数字当成利空的强度锚点。
    """
    if not text:
        return None
    # 先去掉所有空白：PDF 抽取常把词拆到两行（实测「增\n长50%以上」「同比分别下降
    # 34.3% 和 42.6%」），不归一化会让动词与数字被换行隔断而漏匹配。
    text = re.sub(r"\s+", "", text)
    if direction == "up":
        verbs = r"(?:增长|增加|上升|提高)"
        nouns = r"(?:增幅|增长幅度|增长率)"
    elif direction == "down":
        verbs = r"(?:下滑|下降|减少|降低|亏损)"
        nouns = r"(?:降幅|下降幅度|减少幅度)"
    else:
        verbs = r"(?:增长|增加|上升|提高|下滑|下降|减少|降低|亏损)"
        nouns = r"(?:增幅|增长幅度|增长率|降幅|下降幅度|减少幅度)"

    pats = (
        # ① 动词 … X%
        re.compile(verbs + r"[^0-9%]{0,14}(\d{1,4}(?:\.\d+)?)\s*%"),
        # ② 「增加/减少 X 亿元或 Y%」：金额在前、百分比由「或」引出。
        #    必须锚定在同一句的增减动词之后，否则会把「实际税负为 20%」
        #    「占比为 30.0%」这类非同比数字当成增幅（实测误伤 6 条分红公告）。
        re.compile(verbs + r"[^。；;]{0,40}?或(?:约)?(\d{1,4}(?:\.\d+)?)%"),
        # ③ 名词「增幅/降幅 … X%」
        re.compile(nouns + r"[^0-9%]{0,8}(?:约)?(\d{1,4}(?:\.\d+)?)%"),
    )
    vals = []
    for pat in pats:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            # 上限放宽到 10000%：真业绩预告里「增加 770%」是常见表述（基数低所致），
            # 旧上限 500% 会把它当笔误丢掉——而它恰恰是最强的正向数量级。
            if 0 < v <= 10000:
                vals.append(v)
    return max(vals) if vals else None


def _strip_boilerplate(text: str) -> str:
    """剔除法定免责套话。

    **必须在判强度之前做。** 公告开头的「保证本公告内容不存在任何虚假记载、
    误导性陈述或者重大遗漏」是法定格式，不是本文的情绪表述；不剔除就会让
    「重大」这个强度词对 80% 的公告恒为真，把所有预减都顶上强档。
    """
    if not text:
        return ""
    for b in BOILERPLATE:
        text = text.replace(b, "")
    return text


def _big_loss(text: str) -> bool:
    """强档条件② 的第三条命中方式：亏损金额绝对值重大（≥5 亿元）。

    prompt v5 的 bands.md 明确列了这一种，但旧标注函数只认「同比 ≥50%」与
    「强度词」，漏掉了它——于是同样亏 135 亿的两份公告，写法带「下降 84.88%」
    的判强档，只写「净亏损约 135 亿元」的就判偏档。同一事实因表述不同被标成
    两个档，这是标注器与 prompt 判据不一致。

    金额有三种写法，都要认：
      · 净亏损约为 135 亿元                         （「亏损」+ 正数）
      · 净利润为人民币-102 亿元                     （带负号）
      · 净利润预计为人民币-7,907 百万元             （百万元口径，需换算）
    """
    if not text:
        return False

    def _val(s: str, scale: float = 1.0) -> float:
        try:
            return float(s.replace(",", "")) * scale
        except ValueError:
            return 0.0

    # 带负号的金额（亿元 / 百万元）；百万元 ÷100 换算成亿元
    for m in re.finditer(r"-([\d,]+(?:\.\d+)?)\s*亿元", text):
        if _val(m.group(1)) >= 5:
            return True
    for m in re.finditer(r"-([\d,]+(?:\.\d+)?)\s*百万元", text):
        if _val(m.group(1), 0.01) >= 5:
            return True
    # 「亏损 … X 亿元」的正数写法
    for m in re.finditer(r"亏损[^0-9]{0,10}([\d,]+(?:\.\d+)?)\s*亿元", text):
        if _val(m.group(1)) >= 5:
            return True
    return False


def _amount_yi(pattern_text: str) -> tuple[float, float] | None:
    """从一段文字里取出金额区间（亿元），返回 (低, 高)；单值时两者相同。

    认两种口径：亿元，以及百万元（÷100 换算）。老公告常用「2,651 百万元」，
    只认亿元会让整条样本解析失败而退回保守判定。
    """
    if not pattern_text:
        return None
    flat = re.sub(r"\s+", "", pattern_text)
    m = re.search(
        r"(-?[\d,]+(?:\.\d+)?)\s*(亿元|百万元)"
        r"(?:\s*到\s*(-?[\d,]+(?:\.\d+)?)\s*(?:亿元|百万元))?", flat)
    if not m:
        return None
    scale = 1.0 if m.group(2) == "亿元" else 0.01
    try:
        lo = float(m.group(1).replace(",", "")) * scale
        hi = float(m.group(3).replace(",", "")) * scale if m.group(3) else lo
    except ValueError:
        return None
    return (min(lo, hi), max(lo, hi))


def _period_amounts(text: str) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """取 (本期净利润区间, 上年同期净利润区间)，单位亿元。

    本期：在「上年同期」小节**之前**的正文里找第一个「净利润/净亏损」金额。
    上年同期：定位「上年同期业绩」小节，取该节内的第一个金额。

    顺序很重要：老公告（如 2008 年中石化预减）正文前半段没有本期金额，
    从全文取第一个会抓到上年同期那个数，导致「本期=上年」而判成持平。

    亏损的两种写法都归一成负数：
      · 「净亏损约为 9 亿元到 13 亿元」   → -13 ~ -9
      · 「净利润为人民币-35 亿元到-47 亿元」 → -47 ~ -35
    """
    flat = re.sub(r"\s+", "", text)

    def _signed(seg: str) -> tuple[float, float] | None:
        amt = _amount_yi(seg)
        if amt is None:
            return None
        # 「净亏损」字样表示这是亏损，但数字可能写成正数
        is_loss = ("净亏损" in seg[:40] or "亏损" in seg[:40]) and amt[1] > 0
        if is_loss:
            return (-amt[1], -amt[0])
        return amt

    # 定位「上年同期业绩」小节（真正列出上年数字的地方）。
    # 不能退回到裸的「上年同期」——「与上年同期数据相比，减少约 29 亿元」
    # 里的 29 是**变动额**不是上年值（兖矿实测：本期 46.5 亿被读成上年 29 亿）。
    prev_idx = flat.find("上年同期业绩")
    cur_seg = flat[:prev_idx] if prev_idx > 0 else flat

    cur = None
    for m in re.finditer(
            r"(?:归属于上市公司股东的)?(?:净亏损|净利润)[^。]{0,60}", cur_seg):
        amt = _signed(m.group(0))
        if amt is not None:
            cur = amt
            break

    # 上年值两条路：① 有「上年同期业绩」小节就直接读；② 否则用同比变动额反推
    # （「本期 X 亿，与上年同期相比减少 Y 亿」→ 上年 = X + Y，盈利口径）。
    prev = None
    if prev_idx >= 0:
        prev = _signed(flat[prev_idx:prev_idx + 180])
    if prev is None and cur is not None:
        delta = re.search(r"(?:减少|下降|降低|增加|增长)(?:约)?([\d,]+(?:\.\d+)?)亿元", cur_seg)
        if delta:
            try:
                v = float(delta.group(1).replace(",", ""))
                sign = 1 if re.search(r"(?:减少|下降|降低)", delta.group(0)) else -1
                prev = (cur[0] + sign * v,) * 2
            except ValueError:
                prev = None
    return cur, prev


def _trend_band(text: str, fallback: str = "bearish") -> str:
    """按「本期 vs 上年同期」判方向与强度（业绩预告的预亏/预减/预盈/扭亏）。

    标题只说「预亏」，但公告的实际含义取决于同比：

      · 上年同期亏 386 亿、本期亏 9–13 亿  → 收窄 97%，是显著改善 → bullish
      · 上年同期亏 10 亿、本期亏 13–19 亿  → 扩大，是恶化 → bearish/strong_bearish
      · 上年同期盈利、本期亏损              → 由盈转亏 → strong_bearish
      · 上年同期亏损、本期盈利              → 扭亏为盈 → strong_bullish

    解析不出上期数字时退回 `fallback`（由调用方按标题方向给出：预亏→bearish，
    预盈→bullish）。**不能一律退回 bearish**：预盈公告常常只写「预计盈利 X 亿」
    而不列上年同期小节，一律退回 bearish 会把工行、柳钢这类正常预盈判成负面
    （实测 6 条误翻）。
    """
    cur, prev = _period_amounts(text)
    if cur is None or prev is None:
        return fallback

    cur_mid = sum(cur) / 2
    prev_mid = sum(prev) / 2

    # 由盈转亏 / 扭亏为盈：方向反转，直接给强档
    if prev_mid > 0 and cur_mid < 0:
        return "strong_bearish"
    if prev_mid < 0 and cur_mid > 0:
        return "strong_bullish"

    # 两端都是亏损：比较亏损幅度
    if prev_mid < 0 and cur_mid < 0:
        loss_prev, loss_cur = abs(prev_mid), abs(cur_mid)
        if loss_prev == 0:
            return "bearish"
        change = (loss_cur - loss_prev) / loss_prev * 100
        if change <= -TREND_IMPROVE_PCT:
            return "bullish"          # 亏损显著收窄
        if change >= TREND_WORSEN_PCT:
            return "strong_bearish"   # 亏损显著扩大
        if change > 0:
            return "bearish"          # 亏损扩大（含温和扩大）
        # 收窄但不足 50%：仍在亏损、改善有限，方向不明
        return "neutral"

    # 两端都盈利：按同比方向判（预增/预减都走这里，以正文数字为准）
    if prev_mid > 0 and cur_mid > 0:
        change = (cur_mid - prev_mid) / prev_mid * 100
        if change <= -TREND_WORSEN_PCT:
            return "strong_bearish"
        if change < 0:
            return "bearish"
        if change >= TREND_IMPROVE_PCT:
            return "strong_bullish"
        return "bullish"
    return "bearish"


def _has_operational_basis(text: str) -> bool:
    """强档第 ③ 条：理由指向经营实质，且不能只来自一次性损益。

    判定前先剔免责套话：「扣除非经常性损益」本身是财报格式语，出现在几乎
    每份业绩预告里；把它当成"利空来自非经常性项目"会让真正的经营恶化被降档。
    """
    if not text:
        return False
    clean = _strip_boilerplate(text)
    has_op = any(s in clean for s in OPERATIONAL_SOURCES)
    one_off_hits = sum(1 for s in ONE_OFF_SOURCES if s in clean)
    # 只出现一次性损益来源、没有经营实质词 → 不满足 ③
    return has_op or one_off_hits == 0


def label(title: str, category: str, text: str = "") -> str | None:
    """按标题 + 正文共同判定档位。

    text 为空时退化为标题判据（向后兼容旧调用），但**正常构建必须传正文**：
    强度档靠正文的量化锚点判，标题只用于识别公告类型与排除事务性公告。

    强度判定只对**业绩预告**做。分红方案、股东大会这类公告的正文里会引用
    经营背景（「2025 年集团实现净利润…同比增长 1.0%」），那是背景信息不是
    这份公告自身的信号——按正文判强度会让一份分红方案因为去年利润涨了 1%
    升到强档。这类公告的方向由规则表直接给定，不参与强度判定。
    """
    for kw in NEUTRAL_TITLE:
        if kw in title:
            return "neutral"
    for cat, kws, band in RULES:
        if cat != category:
            continue
        if not kws:  # 空关键词 = 整类按该档
            return band
        for kw in kws:
            if kw in title:
                # __trend__：预亏/预减/预盈/扭亏类，方向由同比变化决定，不看标题。
                # 解析不出上期数字时按**标题方向**回退：预盈/扭亏 → bullish，
                # 预亏/预减 → bearish（见 _trend_band 的 fallback 说明）。
                if band == "__trend__":
                    if category != "业绩预告":
                        return "bearish"
                    fallback = "bullish" if kw in ("预盈", "扭亏") else "bearish"
                    return _trend_band(text, fallback=fallback)
                if band not in ("bullish", "bearish"):
                    return band
                # 只有业绩预告的正文才承载"这份公告自身的量化锚点"
                if category != "业绩预告":
                    return band
                return _strengthen(band, title, text)
    return None


def _strengthen(band: str, title: str, text: str) -> str:
    """偏档 → 强档的三条件判定（与 prompt v5 的强档门槛同源）。

    ① 方向明确：由调用方已保证（进了 bullish/bearish 分支）
    ② 命中量化锚点（**三种任一**，与 bands.md 第二条逐项对齐）：
       - 正文净利/营收同比 ≥50%
       - 亏损/盈利金额绝对值重大（数亿元量级）
       - 正文明确出现**该方向**的强度词
    ③ 理由指向经营实质：非一次性损益驱动

    旧实现在这里有两处偏离：
      · 强度词用「重大」，而它几乎只出现在免责套话里 → 判据恒真
      · 强度词不分方向（「大幅增长」能命中一份亏损公告）→ 方向与强度脱钩
      · 漏掉 bands.md 明列的「亏损金额绝对值重大」→ 同额亏损因表述不同而分档
    """
    pct = _max_yoy_pct(text, "up" if band == "bullish" else "down")
    words = STRONG_WORDS_BULL if band == "bullish" else STRONG_WORDS_BEAR
    # 标题不参与强度词判定：套话已在正文，标题里的「预亏」是方向不是强度；
    # 且实测标题带"大幅"而正文无锚点的情形会被误判（gold-v6 的 13 条误判源于此）。
    clean_head = _strip_boilerplate(text[:600])
    anchor_hit = (
        (pct is not None and pct >= STRONG_PCT_THRESHOLD)
        or (band == "bearish" and _big_loss(text))
        or any(w in clean_head for w in words)
    )
    if not anchor_hit:
        return band
    if not _has_operational_basis(text):
        return band
    return "strong_" + band


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-version", default="gold-v1")
    ap.add_argument("--target", type=int, default=54)
    ap.add_argument("--db", default=str(ROOT / "e2e.db"))
    ap.add_argument(
        "--tickers", default="",
        help="逗号分隔的额外公司代码（追加到默认池）。补偏档时用中小市值公司："
             "大盘股的预减按「亏损金额绝对值重大」判据几乎必然升到 strong_*。",
    )
    args = ap.parse_args()

    from fd_sentiment.sanitize import flatten
    from fd_sentiment.storage import Store
    import cninfo_client as c
    import cnreport_tools as t
    import httpx

    store = Store(args.db)
    # 公司池：前六个是蓝筹（业绩预告天然偏正面，用于 bullish/neutral）；
    # 后六个是当期真实的预亏/预减公司（2026Q1 由 akshare 业绩预告快照筛出，
    # 137 家负面前景里取前几个）——三档均衡必须靠真实负面样本，不能靠把
    # 中性公告标成负面来凑。
    # 扩充到 20 家：五档齐全需要足够的强档样本，而单家公司的预增/预亏公告
    # 有限（gold-v6 的 12 家只凑出三档）。前 8 家蓝筹给正面/中性，后 12 家
    # 含周期股与已知预亏公司给负面与强负面。
    #
    # `--tickers` 可追加公司（逗号分隔）。补**偏档**（bullish/bearish）时必须
    # 追加中小市值公司：大盘股的预减动辄数亿元，按 prompt v5 的「亏损金额
    # 绝对值重大」判据几乎必然升到 strong_*，于是偏档永远只剩零星几条。
    # 这不是判据的问题——判据与 prompt 同源且正确——是取样池的市值偏差。
    tickers = [
        "600519", "000333", "000001", "600036", "000858", "601318", "600030", "601899",
        "601111", "600029", "000002", "600048", "601088", "600188", "000898", "600019",
        "601398", "600028", "000063", "002415",
        # 2026-10-19 扩样：负面样本原先全部来自国航(601111)+南航(600029)，
        # 其余 18 家源里根本没有预减/预亏公告，导致 bearish 只有 1–4 条。
        # 下面这批是实测确有负面预告的周期性主体（钢铁/煤炭/地产/航空/面板），
        # 用 `query_announcements(category='业绩预告')` 逐家筛出来的：
        #   600808 马钢 10 条、601003 柳钢 7 条、600115 东航 8 条、
        #   601225 陕煤 3 条、600221 海航 3 条、600703 三安 3 条、
        #   600606 绿地 2 条、600362 江铜 1 条
        "600808", "601003", "600115", "601225", "600221", "600703", "600606", "600362",
    ]
    if args.tickers:
        for extra in args.tickers.split(","):
            extra = extra.strip()
            if extra and extra not in tickers:
                tickers.append(extra)
    counts: dict[str, int] = {}
    added = 0
    seen: set[str] = set()

    # 轮转采样：先每家公司取 1 条，再取第 2 条……避免前几家公司（蓝筹）
    # 把配额吃满而负面公司池永远轮不到——三档均衡靠的是覆盖面，不是排序。
    #
    # 但"按公司轮转"仍会让**类别**失衡：每家的 bucket 里业绩预告排在最前，
    # 中小市值公司一加进来，前几轮全是业绩预告，neutral 类（权益分派/股东
    # 大会/董事会/公司治理）一条都轮不到。所以 bucket 按**类别交错**排列
    # （各取第 1 条、各取第 2 条……），让轮转同时覆盖公司与类别两个维度。
    pending: dict[str, list[tuple[str, dict]]] = {}
    for ticker in tickers:
        co = c.lookup_company(ticker)
        if not co:
            continue
        by_category: dict[str, list[dict]] = {}
        for category in ("业绩预告", "风险提示", "权益分派", "股东大会", "董事会", "公司治理"):
            try:
                rows = c.query_announcements(co["stock_code"], co["org_id"], category=category, limit=25)
            except Exception as exc:
                print(f"  ! {ticker}/{category}: {exc}")
                continue
            by_category[category] = [r for r in rows if r.get("pdf_url")]
        bucket: list[tuple[str, dict]] = []
        depth = max((len(v) for v in by_category.values()), default=0)
        for i in range(depth):
            for category, rows in by_category.items():
                if i < len(rows):
                    bucket.append((category, rows[i]))
        pending[ticker] = bucket

    # 类别配额：金标要覆盖**五档**，而 neutral 类公告（权益分派/股东大会/
    # 董事会/公司治理）在真实公告里占绝对多数。纯轮转采样会被它们淹没
    # （实测 100 条里 neutral 占 82），强档只剩个位数；反过来只取业绩预告
    # 又会让 bearish 全被「亏损金额重大」推到 strong_bearish。
    # 配额按「信号类别 : 中性类别」大致 2:1 分配，各档才有机会达标。
    SIGNAL_CATEGORIES = ("业绩预告", "风险提示")
    category_quota = {
        "业绩预告": max(1, int(args.target * 0.55)),
        "风险提示": max(1, int(args.target * 0.15)),
    }
    # 中性类共享一个总上限：它们只在金标里扮演"不该被判成有方向"的对照组，
    # 数量再多也不增加信息量，却会挤掉信号类（实测不设限时 neutral 占 83%）。
    neutral_budget = max(1, int(args.target * 0.30))
    used_by_category: dict[str, int] = {}
    used_neutral = 0

    round_idx = 0
    while added < args.target and any(round_idx < len(v) for v in pending.values()):
        for ticker in tickers:
            if added >= args.target:
                break
            bucket = pending.get(ticker, [])
            if round_idx >= len(bucket):
                continue
            category, r = bucket[round_idx]
            aid = r["announcement_id"]
            if aid in seen:
                continue
            quota = category_quota.get(category)
            if quota is not None and used_by_category.get(category, 0) >= quota:
                continue
            if category not in SIGNAL_CATEGORIES and used_neutral >= neutral_budget:
                continue
            # 正文必须先取：强度档靠正文的量化锚点判定（sentiment-gold-coverage:
            # 标注依据与判据一致性）。先判档再取正文会让标注意义上"看不见数字"，
            # 这正是 gold-v6 那 13 条误判的成因。
            try:
                with httpx.Client(timeout=60, follow_redirects=True) as cl:
                    text = flatten(t.extract_pdf_text(cl.get(r["pdf_url"]).content))
            except Exception as exc:
                print(f"  ! {aid} PDF: {exc}")
                continue
            if len(text) < 100:
                continue
            band = label(r["title"], category, text)
            if band is None:
                continue
            # 按**实际判出的档位**限额，而不是按类别。
            # 按类别限不住 neutral：风险提示类里绝大多数是「一般风险提示/异常波动」，
            # 实测 18 条全判 neutral，且它属 SIGNAL_CATEGORIES 不占中性预算，
            # 于是 neutral 膨胀到 47%，把负面档挤到只剩 8 条。
            # neutral 只是「不该被判成有方向」的对照组，够用即可。
            if band == "neutral" and counts.get("neutral", 0) >= neutral_budget:
                continue
            seen.add(aid)
            store.upsert_gold(
                set_version=args.set_version, doc_id=aid, as_of=r["published"][:10],
                expected_band=band, text=text,
                note=f"{ticker}｜{category}｜{r['title'][:40]}",
            )
            counts[band] = counts.get(band, 0) + 1
            used_by_category[category] = used_by_category.get(category, 0) + 1
            if band == "neutral":
                used_neutral += 1
            added += 1
            print(f"  + {aid} {band:15s} {r['title'][:34]}")
        round_idx += 1

    # 覆盖度检查：五档必须齐全，缺档就失败（sentiment-gold-coverage）。
    # 金标缺档会让模型输出该档时被记成"错误"，从而系统性低估打分器——
    # gold-v6 缺 strong_* 整整两档，导致 exact 被压低 23 个百分点。
    ALL_BANDS = ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish")
    missing = [b for b in ALL_BANDS if counts.get(b, 0) == 0]

    print(f"\n金标集 {args.set_version}：{added} 条，分布 {counts}")
    print("五档覆盖：" + " ｜ ".join(f"{b}={counts.get(b, 0)}" for b in ALL_BANDS))
    # 分层统计：金标按**真实公告类型分布**构建，不人为凑均衡（2026-10-19 裁决）。
    # 理由：业绩预告的法定类型只有预增/预减/预亏/扭亏，增幅中位数就是 54%，
    # 人为拉平会让金标偏离真实数据分布，评测出来的数不代表实际使用场景。
    # 代价是弱档样本天然稀少 —— 故报告必须**分层呈现**，弱档单独说明样本量不足。
    cat_counts: dict[str, dict[str, int]] = {}
    for r in store.list_gold(args.set_version):
        cat = (r.get("note") or "?｜?｜?").split("｜")[1]
        cat_counts.setdefault(cat, {}).setdefault(r["expected_band"], 0)
        cat_counts[cat][r["expected_band"]] += 1
    print("\n按类别分层：")
    for cat, dist in cat_counts.items():
        total = sum(dist.values())
        thin = [b for b, n in dist.items() if n < WEAK_BAND_MIN]
        print(f"  {cat}: n={total} ｜ " + " ".join(f"{b}={n}" for b, n in sorted(dist.items()))
              + (f"  ← 样本不足档: {'、'.join(thin)}" if thin else ""))

    fail = False
    if missing:
        print(f"✗ 档位覆盖不全，缺：{'、'.join(missing)}", file=sys.stderr)
        print("  缺档会让模型输出该档时被记成错误（gold-v6 即因此把 exact 压低 23 个百分点）",
              file=sys.stderr)
        fail = True
    if added < 50:
        print(f"✗ 样本量不足 50（实为 {added}）", file=sys.stderr)
        fail = True
    if fail:
        return 1
    thin_bands = [b for b, n in counts.items() if n < WEAK_BAND_MIN]
    if thin_bands:
        print(f"⚠ 以下档位样本偏少（<{WEAK_BAND_MIN} 条），该档的评测结论不具统计意义："
              f"{'、'.join(thin_bands)}", file=sys.stderr)
    print("✓ 满足 ≥50 条且五档齐全（分布按真实公告类型，非人为均衡）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
