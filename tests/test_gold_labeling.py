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
    text = BOILER + "预计本期净利润同比减少 84.88%，主要因销量下滑。"
    assert g.label("某公司2020年年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_big_loss_amount_alone_qualifies():
    """只写「净亏损约 135 亿元」、没有同比数字 → 仍是强档。

    prompt v5 的 bands.md 明列「亏损金额绝对值重大（数亿元量级）」是条件②的
    一种命中方式；旧标注函数漏掉它，导致同额亏损因表述不同而分档。
    """
    text = BOILER + "预计2020年度归属于上市公司股东的净亏损约为135亿元到155亿元。"
    assert g.label("某公司2020年年度业绩预亏公告", "业绩预告", text) == "strong_bearish"


def test_negative_sign_amount_qualifies():
    """「净利润为人民币-102 亿元」这种带负号的写法也要认。"""
    text = BOILER + "预计半年度归属于上市公司股东的净利润为人民币-102亿元到人民币-121亿元。"
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
