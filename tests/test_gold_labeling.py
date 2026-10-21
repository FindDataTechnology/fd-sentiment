"""金标标注规则回归：dominated by 免责套话的强度词判据。

实测缺陷（gold-v7）：`STRONG_WORDS` 含「重大」，而 90 条里 72 条正文前 400 字
都有法定的「保证本公告内容不存在任何虚假记载、误导性陈述或者重大遗漏」。
于是这个判据对几乎所有公告恒为真，任何业绩预减都被顶成 strong_bearish——
gold-v7 里 bearish 只剩 2 条就是这么来的。

同一份预减公告，写法带「下降 84.88%」的判强档、只写「净亏损约 135 亿元」的
判偏档：同一个事实因表述不同落进两个档，这是标注器与 prompt 判据不一致。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_gold_set as g  # noqa: E402

# 真实的公告开头（法定免责套话）
BOILER = ("本公司董事会及全体董事保证本公告内容不存在任何虚假记载、误导性陈述"
          "或者重大遗漏,并对其内容的真实性、准确性和完整性承担个别及连带责任。")


def test_boilerplate_removed_before_strength_judgment():
    """免责套话里的「重大」不得被当成本文的强度表述。"""
    clean = g._strip_boilerplate(BOILER)
    assert "重大" not in clean
    assert "重大遗漏" not in clean


def test_big_word_not_in_strong_words():
    """「重大」必须从强度词表里移除——它只出现在套话里。"""
    assert "重大" not in g.STRONG_WORDS_BULL
    assert "重大" not in g.STRONG_WORDS_BEAR
    assert "重大" not in g.STRONG_WORDS


def test_small_decline_stays_weak_despite_boilerplate():
    """小幅预减 + 免责套话 → 偏档，不能因套话升成强档。"""
    text = BOILER + ("重要内容提示：预计本期归属于上市公司股东的净利润同比减少 12%，"
                     "主要因产品销量下滑、销售价格下降。")
    assert g.label("某公司2025年半年度业绩预减公告", "业绩预告", text) == "bearish"


def test_large_decline_goes_strong():
    """同比降幅 ≥50% → 强档。"""
    # 亏损显著扩大（47 亿 → 102 亿）→ 强负面
    text = (BOILER + "预计本期净亏损约为人民币102亿元，主要因销量下滑。"
            "上年同期业绩情况 归属于上市公司股东的净亏损:人民币46.88亿元。")
    assert g.label("某公司2020年年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_big_loss_amount_alone_qualifies():
    """只写「净亏损约 135 亿元」、没有同比数字 → 仍是强档。

    prompt v5 的 bands.md 明列「亏损金额绝对值重大（数亿元量级）」是条件②的
    一种命中方式；旧标注函数漏掉它，导致同额亏损因表述不同而分档。
    """
    text = (BOILER + "预计2020年度归属于上市公司股东的净亏损约为135亿元到155亿元。"
            "上年同期业绩情况 归属于上市公司股东的净利润:人民币64.09亿元。")
    assert g.label("某公司2020年年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_negative_sign_amount_qualifies():
    """「净利润为人民币-102 亿元」这种带负号的写法也要认。"""
    text = (BOILER + "预计半年度归属于上市公司股东的净利润为人民币-102亿元到人民币-121亿元。"
            "上年同期业绩情况 归属于上市公司股东的净亏损:人民币46.88亿元。")
    assert g.label("某公司2022年半年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_small_loss_stays_weak():
    """小额亏损（<5 亿）且无其他锚点 → 偏档。"""
    text = BOILER + "预计本期归属于上市公司股东的净亏损约为 0.8 亿元。"
    assert g.label("某公司2025年半年度业绩预亏公告", "业绩预告", text) == "bearish"


def test_growth_word_does_not_strengthen_a_loss():
    """亏损公告里出现「营业收入增长 30%」不得被当成利空的强度锚点。"""
    text = BOILER + ("预计本期净亏损约为 1 亿元；报告期内营业收入增长 30%。")
    assert g._max_yoy_pct(text, "down") is None
    assert g._max_yoy_pct(text, "up") == 30.0
    assert g.label("某公司2025年半年度业绩预亏公告", "业绩预告", text) == "bearish"


def test_large_growth_above_old_500_cap_is_kept():
    """「增加 770%」是真业绩预增表述，不得被 500% 上限当笔误丢掉。"""
    text = BOILER + "预计2016年度实现归属于上市公司股东的净利润将增加770%左右。"
    assert g._max_yoy_pct(text, "up") == 770.0
    assert g.label("某公司2016年度业绩预增公告", "业绩预告", text) == "strong_bullish"


def test_one_off_growth_stays_weak():
    """增长理由是一次性损益 → 不进强档（prompt v5 条件③）。"""
    text = (BOILER + "预计本期净利润同比增长 120%，主要因政府补助增加及资产处置收益。"
            "报告期内公司主营业务收入与销量基本持平。")
    # 有经营实质词（主营业务/销量）→ 满足③；用纯一次性损益且无经营词才降档
    text_no_op = BOILER + "预计本期净利润同比增长 120%，主要因政府补助及资产处置收益。"
    assert g.label("某公司2025年年度业绩预增公告", "业绩预告", text_no_op) == "bullish"


# ==== 同比方向判据（v6 / gold-v9）====

def test_loss_narrowing_is_positive():
    """亏损大幅收窄 → 偏正面。

    实测：国航 2023 年报从亏 386 亿收窄到亏 9–13 亿，公告标题仍叫「业绩预亏」，
    旧规则按标题一律判强负面，而正文写「经营表现同比大幅好转」。
    """
    text = (BOILER + "预计2023年度归属于上市公司股东的净亏损约为人民币9亿元到人民币13亿元。"
            "二、上年同期业绩情况 归属于上市公司股东的净亏损:人民币386.19亿元。")
    assert g.label("某公司2023年年度业绩预亏公告", "业绩预告", text) == "bullish"


def test_loss_widening_is_negative():
    """亏损扩大 → 负面；扩大 ≥50% → 强负面。"""
    mild = (BOILER + "预计净亏损约为人民币4亿元。"
            "上年同期业绩情况 归属于上市公司股东的净亏损:人民币3.8亿元。")
    assert g.label("某公司2021年年度业绩预亏公告", "业绩预告", mild) == "bearish"

    severe = (BOILER + "预计净亏损约为人民币102亿元。"
              "上年同期业绩情况 归属于上市公司股东的净亏损:人民币46.88亿元。")
    assert g.label("某公司2022年半年度业绩预亏公告", "业绩预告", severe) == "strong_bearish"


def test_big_loss_widening_hits_amount_anchor():
    """亏损扩大幅度温和、但金额 ≥5 亿元 → 仍进强档（金额锚点等价）。

    prompt v6 条件② 把「单期亏损 ≥5 亿元」列为强档锚点；spec 要求标注器
    同等地实现全部锚点形态。实测南航 2025 半年度：亏 13.38–17.56 亿，
    上年亏 12.28 亿（扩大 26%，未过 50%），按金额锚点应判强负面。
    """
    text = (BOILER + "预计公司2025年半年度实现归属于上市公司股东的净利润为人民币"
            "-13.38亿元到人民币-17.56亿元。"
            "二、上年同期业绩情况 归属于上市公司股东的净利润:人民币-12.28亿元。")
    assert g._period_amounts(text)[0] == (-17.56, -13.38)
    assert g.label("南方航空2025年半年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_profit_to_loss_is_strong_negative():
    """由盈转亏 → 强负面（与标题方向一致，但依据是同比反转）。"""
    text = (BOILER + "预计净亏损约为人民币135亿元。"
            "上年同期业绩情况 归属于上市公司股东的净利润:人民币64.09亿元。")
    assert g.label("某公司2020年年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_loss_to_profit_is_strong_positive():
    """扭亏为盈 → 强正面。"""
    # 预盈类走 bullish 分支，由 _trend_band 的「上年亏本期盈」判强正面；
    # 标题须命中「预盈」关键词
    text = (BOILER + "预计净利润约为人民币6亿元。"
            "上年同期业绩情况 归属于上市公司股东的净亏损:人民币12亿元。")
    assert g._trend_band(text) == "strong_bullish"


def test_small_loss_narrowing_is_neutral():
    """亏损收窄不足 50% → 中性（仍亏损，方向不明）。"""
    text = (BOILER + "预计净亏损约为人民币13亿元到人民币19亿元。"
            "上年同期业绩情况 亏损总额:人民币16.05亿元。")
    assert g.label("某公司2025年年度业绩预亏公告", "业绩预告", text) == "neutral"


def test_prior_year_not_confused_with_delta():
    """「与上年同期数据相比，减少约 29 亿元」里的 29 是变动额，不是上年值。

    实测缺陷：把变动额当成上年净利润，算出「本期 46.5 上年 29」→ 误判为增长。
    正确是上年 76 亿，本期 46.5 亿，同比 -38.8% → 偏负面。
    """
    text = (BOILER + "预计2025年半年度实现归属于上市公司股东的净利润约46.5亿元,"
            "与上年同期数据相比,减少约29亿元或38%。")
    cur, prev = g._period_amounts(text)
    assert cur == (46.5, 46.5)
    assert prev is not None and abs(prev[0] - 75.5) < 1.0, f"上年应约 75.5，实得 {prev}"
    assert g.label("某公司2025年半年度业绩预减公告", "业绩预告", text) == "bearish"


def test_million_yuan_unit_parsed():
    """老公告用「2,651 百万元」，需换算成 26.51 亿元。"""
    text = (BOILER + "预计本期净亏损约为人民币79.07亿元。"
            "上年同期业绩情况 归属于上市公司股东的净利润:人民币2,651百万元。")
    cur, prev = g._period_amounts(text)
    assert cur == (-79.07, -79.07)
    assert prev is not None and abs(prev[0] - 26.51) < 0.01
    assert g.label("某公司2020年年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


# ==== 预盈/扭亏 走同比分支（v6-final）====

def test_profit_turnaround_routes_through_trend():
    """预盈/扭亏标题必须先于「预增」匹配，否则扭亏被判成普通增长。

    实测：海航 2025 年报「上年亏、本期盈」，旧路由走 bullish 分支判成偏档，
    正确应是 strong_bullish（扭亏为盈是方向反转）。
    """
    text = (BOILER + "预计2025年度实现归属于上市公司股东的净利润约为人民币6亿元。"
            "上年同期业绩情况 归属于上市公司股东的净亏损:人民币12亿元。")
    assert g.label("海航控股：2025年年度业绩预盈的公告", "业绩预告", text) == "strong_bullish"


def test_missing_prior_year_falls_back_to_title_direction():
    """没有「上年同期业绩」小节时按标题方向回退，不能一律退回 bearish。

    实测：工行/柳钢的预盈公告只写「预计盈利 X 亿」不列上期，一律退回 bearish
    会把 6 条正常预盈判成负面。
    """
    text = BOILER + "预计2021年一季度归属于母公司股东的净利润预计人民币160亿元到180亿元。"
    assert g.label("2021年一季度业绩预盈公告", "业绩预告", text) == "bullish"

    loss_text = BOILER + "预计本期将出现亏损。"
    assert g.label("某公司2021年年度业绩预亏公告", "业绩预告", loss_text) == "bearish"


def test_amount_anchor_is_loss_only():
    """金额锚点只适用于亏损，不适用于盈利。

    v6 曾写成「亏损或盈利金额 ≥ 5 亿元」，导致大盘蓝筹的温和增长
    （如工行 +10.4%、中信 +32%）因「盈利金额大」被判成强正面。
    """
    prompt_text = (Path(__file__).resolve().parent.parent
                   / "fd_sentiment" / "prompts" / "v6" / "bands.md").read_text(encoding="utf-8")
    assert "只适用于亏损" in prompt_text
    assert "亏损或盈利金额" not in prompt_text


# ==== 金额解析：兜底正则与口径（gold-v9 重标前修正）====

def test_single_amount_without_range_does_not_crash():
    """单值金额走两捕获组的兜底正则，不得 IndexError。

    主正则要求「X 到 Y」区间；单值公告（「净亏损约为 0.8 亿元」）落到只有
    两个捕获组的兜底正则。旧代码用 `m.lastindex and m.group(3)` 判空，而
    `m.lastindex == 2` 为真，于是仍去取 group(3) → IndexError，整轮建集中断。
    """
    assert g._amount_yi("净亏损约为人民币0.8亿元") == (0.8, 0.8)
    assert g._amount_yi("净利润为人民币2,651百万元") == (26.51, 26.51)


def test_wan_yuan_unit_parsed():
    """「85,000 万元」= 8.5 亿元：航空/钢铁类公告常用万元口径。"""
    assert g._amount_yi("净亏损85,000万元") == (8.5, 8.5)
    lo, hi = g._amount_yi("净亏损85,000万元至92,000万元")
    assert abs(lo - 8.5) < 1e-9 and abs(hi - 9.2) < 1e-9


def test_explicit_current_period_beats_delta_form():
    """正文明确写了本期金额时，用它；差额式只用于反推。

    实测陕煤 2023 半年度：既写本期 112–118 亿，又写「将减少 137 亿元到 143 亿元」。
    先算差额会把本期算成 103–109 亿（虽然同档），但显式值才是公告口径。
    """
    text = (BOILER + "预计2023年半年度实现归属于上市公司股东的净利润与上年同期相比，"
            "将减少137亿元到143亿元，同比减少53%到56%。"
            "预计2023年半年度实现归属于上市公司股东的净利润为112亿元到118亿元。"
            "二、上年同期业绩情况 归属于上市公司股东的净利润:245.98亿元。")
    cur, prev = g._period_amounts(text)
    assert prev == (245.98, 245.98)
    assert cur == (112.0, 118.0)
    assert g.label("陕西煤业股份有限公司2023年半年度业绩预减公告", "业绩预告", text) == "strong_bearish"


def test_delta_only_announcement_derives_current_period():
    """只给差额、不给本期值时，用「上年 + 差额」反推。

    实测江铜 2018 预增：归母只写「增加 8.02 亿元，同比增加 50%」，本期未直接给；
    正文另有一句「经重述后，2017 年…净利润为 16.06 亿元」——显式口径会把上期
    重述数当本期（+0.1%），与公告自报 +50% 自相矛盾。反推得 24.06 亿。
    """
    text = (BOILER + "预计2018年1-12月实现归属于上市公司股东的净利润与上年同期相比,"
            "将增加8.02亿元左右,同比增加50%左右。"
            "2018年1-12月,公司实现归属于上市公司股东的扣除非经常性损益后的净利润"
            "为15.31亿元左右,与上年同期相比,将减少8.55亿元左右,同比减少35.83%左右。"
            "经重述后,2017年1-12月归属上市公司股东的净利润为16.06亿元。"
            "二、上年同期业绩情况(重述前) 归属于上市公司股东的净利润:16.04亿元。")
    cur, prev = g._period_amounts(text)
    assert prev == (16.04, 16.04)
    assert cur is not None and abs(cur[0] - 24.06) < 0.01 and abs(cur[1] - 24.06) < 0.01
    assert g._trend_band(text) == "strong_bullish"


def test_koufei_net_profit_not_taken_as_attributable():
    """「扣除非经常性损益后的净利润」不得当归母净利润。

    实测柳钢 2025 半年度：归母 +530%~641%、扣非 +729%~887%，两者都过 50%，
    但拿扣非金额配归母上年值会算出错误的同比；江铜那条更是 +50% 与 -35.83%
    方向相反，取错口径直接把利空当利好。
    """
    text = (BOILER + "预计2025年半年度实现归属于母公司所有者的净利润约3.4亿元至4亿元,"
            "与上年同期相比,将增加2.86亿元至3.46亿元,同比增加530%至641%。"
            "预计2025年半年度实现归属于母公司所有者的扣除非经常性损益后的净利润"
            "约3.15亿元至3.75亿元。"
            "二、上年同期经营业绩和财务状况(一)利润总额:6,236万元。"
            "归属于母公司所有者的净利润:5,414万元。")
    cur, prev = g._period_amounts(text)
    assert cur == (3.4, 4.0)
    assert prev is not None and abs(prev[0] - 0.5414) < 0.001


def test_hyphen_range_parsed_as_range_not_negative():
    """「145.5-154.5 亿元」是区间，不是 -154.5 亿元。

    实测紫金 2024/2025 预增：连字符区间若走单值正则，会从「-154.5亿元」处
    匹配成负数，与上年正利润相减就成了「由盈转亏」。
    """
    assert g._amount_yi("145.5-154.5亿元") == (145.5, 154.5)
    text = (BOILER + "预计2024年半年度实现归属于上市公司股东的净利润约人民币145.5-154.5亿元,"
            "与上年同期103.02亿元相比,将增加约42.48-51.48亿元,同比增加约41%-50%。"
            "二、上年同期业绩情况 2023年半年度归属于上市公司股东的净利润:103.02亿元。")
    cur, prev = g._period_amounts(text)
    assert cur == (145.5, 154.5)
    assert prev == (103.02, 103.02)


# ==== 数字解析失败时的显式表述与规则误命中（gold-v9 重标后修正）====

def test_explicit_turnaround_phrase_beats_title_fallback():
    """正文写明「实现扭亏为盈」时，即使解析不出上期数值也判强正面。

    实测中石化 2021Q1/H1 预盈公告：只写本期 160–180 亿 / 365–385 亿，不列上年
    数值，旧码退回标题方向判 bullish；而正文写「较2020年同期预计将实现扭亏
    为盈」，按 prompt v6 的同比方向表是 strong_bullish。
    """
    text = (BOILER + "2021年一季度归属于母公司股东的净利润预计人民币160亿元到180亿元,"
            "较2020年同期预计将实现扭亏为盈,较2019年同期预计将增加8.1%到21.6%。")
    assert g._period_amounts(text) == ((160.0, 180.0), None)
    assert g.label("2021年一季度业绩预盈公告", "业绩预告", text) == "strong_bullish"


def test_buyback_price_cap_adjustment_is_neutral():
    """分红实施后调整回购价格上限是机械调整，不是分红方案 → neutral。

    实测五粮液「关于2025年度分红派息实施后调整回购股份价格上限的公告」被
    「调整」关键词误命中判成 bullish；正文只写回购价上限 153.59 → 151.01 元。
    """
    text = (BOILER + "特别提示:1.调整前回购股份价格上限:153.59元/股。"
            "2.调整后回购股份价格上限:151.01元/股。3.调整生效日期:2026年7月16日。")
    title = "关于2025年度分红派息实施后调整回购股份价格上限的公告"
    assert g.label(title, "权益分派", text) == "neutral"


# ==== prompt v7：扭亏金额门槛 + 非业绩预告类判据 ====

def test_small_turnaround_is_bullish():
    """小额扭亏 → 偏正面（prompt v7 金额门槛）。

    实测柳钢 2024H1：上年同期亏 1.45 亿，本期预计盈 0.50–0.58 亿。
    方向改善是真的，但金额不重大——判强正面会与第三档「拿不准取温和」冲突。
    """
    text = (BOILER + "预计2024年半年度实现归属于上市公司股东的净利润为5000万元到5800万元。"
            "上年同期业绩情况 归属于上市公司股东的净利润:-14472万元。")
    assert g._period_amounts(text) is not None
    assert g.label("柳钢股份2024年半年度业绩预盈公告", "业绩预告", text) == "bullish"


def test_large_turnaround_stays_strong():
    """大额扭亏仍进强档：上年亏 202 亿 → 本期盈 3.0–4.5 亿。"""
    text = (BOILER + "预计2023年年度实现归属于上市公司股东的净利润为3亿元到4.5亿元。"
            "上年同期业绩情况 归属于上市公司股东的净利润:-202.47亿元。")
    assert g.label("海航控股：2023年年度业绩预盈公告", "业绩预告", text) == "strong_bullish"


def test_turnaround_without_any_amount_is_bullish():
    """正文只说「扭亏为盈」且解析不出任何金额 → 偏正面（无法证明金额重大）。"""
    text = BOILER + "报告期内公司经营情况改善，较上年同期预计将实现扭亏为盈。"
    assert g._period_amounts(text) == (None, None)
    assert g.label("某公司2025年半年度业绩预盈公告", "业绩预告", text) == "bullish"


def test_reduce_plan_is_bearish():
    """拟减持/预披露 → 偏负面（未来供给增加）。"""
    for title in (
        "关于股东减持公司股份计划的公告",
        "紫金矿业集团股份有限公司高级管理人员减持股份计划公告",
        "关于大股东减持股份的预披露公告",
        "股东减持股份计划公告",
    ):
        assert g.label(title, "股权变动", "") == "bearish", title


def test_reduce_execution_is_neutral():
    """减持执行回执 → 中性（既定计划兑现，无新信息）。"""
    for title in (
        "关于职工监事股份减持计划实施完成的公告",
        "关于股东减持公司股份计划实施结果的公告",
        "关于部分高级管理人员减持计划时间过半的进展公告",
        "关于部分高级管理人员减持计划实施进展暨减持期限届满的公告",
        "关于2021年度员工持股计划减持完毕的公告",
    ):
        assert g.label(title, "股权变动", "") == "neutral", title


def test_no_reduce_commitment_is_neutral():
    """「承诺不减持」是无方向承诺 → 中性，不占负面档。"""
    title = "中国南方航空集团有限公司关于不存在减持公司股票行为或减持计划的承诺函"
    assert g.label(title, "股权变动", "") == "neutral"


def test_involving_litigation_is_bearish():
    """涉及诉讼/仲裁 → 偏负面，不进强档（金额与结果不确定）。"""
    title = "绿地控股关于公司及控股子公司2022年至2025年上半年涉及诉讼的补充公告"
    assert g.label(title, "补充更正", "公司及控股子公司涉及诉讼。") == "bearish"
    assert g.label("关于涉及仲裁的公告", "补充更正", "涉及仲裁。") == "bearish"


def test_earnings_correction_uses_trend_not_title():
    """业绩预告更正公告按同比判方向，不因标题含「更正」直接判负面。"""
    text = (BOILER + "预计2025年年度实现归属于母公司所有者的净利润约为-261亿元至-266亿元。"
            "上年同期业绩情况 归属于母公司所有者的净利润:-155.52亿元。")
    assert g.label("绿地控股2025年年度业绩预告更正公告", "补充更正", text) == "strong_bearish"
    # 上修（亏损收窄）不该判负面
    text2 = (BOILER + "预计本期净亏损约为人民币9亿元至13亿元。"
             "上年同期业绩情况 归属于上市公司股东的净利润:-386亿元。")
    assert g.label("某公司业绩预告修正公告", "补充更正", text2) == "bullish"


def test_earnings_correction_without_numbers_is_neutral():
    """更正公告解析不出两个数 → 中性（更正本身不表方向）。"""
    text = BOILER + "公司对前次业绩预告中的部分表述予以更正，具体内容详见下文。"
    assert g.label("某公司2025年年度业绩预告更正公告", "补充更正", text) == "neutral"


def test_other_correction_is_neutral():
    """非业绩预告的更正/补充（年报更正、章程修订）→ 中性。"""
    assert g.label("关于2024年年度报告更正的公告", "补充更正", "") == "neutral"
    assert g.label("关于补充公告的说明", "补充更正", "") == "neutral"


def test_rule_sources_cover_new_categories():
    """新规则必须能回指 prompt 条目（sentiment-gold-coverage）。"""
    names = [r[0] for r in g.rule_sources()]
    for expect in ("股权变动/减持", "补充更正/业绩预告", "补充更正/涉及诉讼", "补充更正/其他"):
        assert expect in names, f"规则来源缺 {expect}"
