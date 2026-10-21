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
    mild = (BOILER + "预计净亏损约为人民币113亿元。"
            "上年同期业绩情况 归属于上市公司股东的净亏损:人民币108.42亿元。")
    assert g.label("某公司2021年年度业绩预亏公告", "业绩预告", mild) == "bearish"

    severe = (BOILER + "预计净亏损约为人民币102亿元。"
              "上年同期业绩情况 归属于上市公司股东的净亏损:人民币46.88亿元。")
    assert g.label("某公司2022年半年度业绩预亏公告", "业绩预告", severe) == "strong_bearish"


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
