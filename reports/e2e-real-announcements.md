# 端到端：真实公告批量打分（task 7.1）

- 运行时间：2026-10-17T16:40:47+00:00
- 模型：`deepseek-v4.1-flash`｜prompt：`v1`｜库：`/Users/chengsishi/finddata/fd-sentiment/e2e.db`
- 公告来源：fd-cn-report / cninfo（真实公告 PDF）；公司池 600519/000333/000001/600036/000858

## 批量结果

- 取回公告 20 条；可打分 20 篇；跳过 0 篇
- 打分 ok=8 / skipped=0 / failed=12

## evidence 抽查（每条都在原文定位）

- 片段总数 13；可定位 13；切片与 quote 逐字一致 13

| doc_id | quote（截断） | 可定位 | 切片一致 | 区间 |
|---|---|---|---|---|
| 1204375966 | 2017 年度业绩预增公告 | ✓ | ✓ | 65..78 |
| 1204375966 | 同比增加58%左右 | ✓ | ✓ | 208..217 |
| 62065601 | 净利润与上年同期相比增加50%左右 | ✓ | ✓ | 276..293 |
| 60458343 | 预计公司2011年度归属于上市公司股东的净利润较上年同期增长65%以上。 | ✓ | ✓ | 739..775 |
| 1200136385 | 同向上升 | ✓ | ✓ | 194..198 |
| 1200136385 | 比上年同期增长: 118%-140% | ✓ | ✓ | 261..279 |
| 1200054100 | 同向上升 | ✓ | ✓ | 187..191 |
| 1200054100 | 比上年同期增长: 45%-60% | ✓ | ✓ | 814..830 |
| 63751509 | 业绩预告情况:同向大幅上升 | ✓ | ✓ | 196..209 |
| 63538838 | 比上年同期增长52%-69% | ✓ | ✓ | 399..413 |
| 63538838 | 公司预计2013 年度实现的备考利润将超过预测的69.31 亿元 | ✓ | ✓ | 723..755 |
| 1212172931 | 经营业绩稳健增长 | ✓ | ✓ | 1208..1216 |
| 1212172931 | 同比增长25.6% | ✓ | ✓ | 1293..1302 |

## 聚合视图（分布统计，非情绪指数）

```json
{
  "title": "已打分文档的情绪分布统计（非市场情绪指数、非行情指标）",
  "window": {
    "since": null,
    "until": null
  },
  "doc_type": null,
  "prompt_version": null,
  "scored_count": 8,
  "band_counts": {
    "strong_bullish": 8,
    "bullish": 0,
    "neutral": 0,
    "bearish": 0,
    "strong_bearish": 0
  },
  "mean_score": 0.8375,
  "mean_confidence": 0.93,
  "coverage": {
    "scored": 8,
    "universe_total": 20,
    "ratio": 0.4,
    "note": "覆盖说明：已打分数与统计范围内文档总数；universe_total 未知时为 null，不推断、不补算。"
  },
  "caveat": "本视图是已打分文档的分布统计，不是情绪指数，不用于行情或交易判断。"
}
```

## 失败/跳过明细

- 失败 60713041：LLMError: 模型输出中没有可解析的 JSON 对象："We need answer only JSON. Need analyze Chinese document. It's 贵州茅台 2011业绩快报. Financial data: revenue +58.19%, operating profit +72.27%, net profit +73.49%, EPS +73.49%, ROE increase. Strong positive. "
- 失败 37685821：拒绝：模型输出的 score/confidence 不合法：{'quote': '净利润 \n301,623.26 \n166,105.67 \n81.59', 'char_start': 429, 'char_end': 463}
- 失败 13771222：拒绝：模型拒绝打分（无可用情绪证据）：文本为乱码（编码损坏）的财经资讯片段，除日期、股票代码、'50%'、版权信息等零散可读字符外，所有承载语义与情绪方向的中文语句均无法辨认。无法据此确定任何表达正面/负面影响的原文语句，也就无法给出可复核的逐字证据区间，故拒绝打分。
- 失败 1205769097：拒绝：evidence_spans[0].quote 不是原文逐字子串（疑似编造引用）：'预计的经营业绩:同向上升'
- 失败 1200323678：LLMError: 模型输出中没有可解析的 JSON 对象：'We need answer only JSON. Need analyze document sentiment. Need provide evidence spans with exact quotes and char positions. Need determine score. Document is Midea 2014 annual earnings pre-announceme'
- 失败 1215626113：LLMError: 模型输出中没有可解析的 JSON 对象：'我们需要输出JSON。需要判断情绪。文本是平安银行2022业绩快报。营收+6.2，净利润+25.3，营业利润+25，每股收益+27.2，ROE上升，资产总额+8.1，存款+11.8，不良率略升0.03，拨备覆盖率上升，逾期偏离度下降。整体明确正面，措辞“稳健增长”“风险抵补能力保持较好水平”。强度：净利润大幅增长25.3%，强正面？ band strong_bullish likely score'
- 失败 1207246377：拒绝：evidence_spans[0].quote 不是原文逐字子串（疑似编造引用）：'整体经营稳中趋好,资产质量持续改善。'
- 失败 1205708465：LLMError: 模型输出中没有可解析的 JSON 对象：'We need answer only JSON. Need analyze document sentiment. Need score -1 to 1, band, confidence, evidence spans with exact quotes and char indices. We need compute char_start/end accurately. Need dete'
- 失败 1201929343：拒绝：evidence_spans[1].quote 不是原文逐字子串（疑似编造引用）：'比上年同期增长:5% ~15%'
- 失败 60475802：拒绝：evidence_spans[0].quote 不是原文逐字子串（疑似编造引用）：'2.本报告期本公司(深发展)单体净利润同比大幅增长。'
- 失败 60060081：拒绝：evidence_spans[0].quote 不是原文逐字子串（疑似编造引用）：'比上年同期增长:60% – 70%'
- 失败 59667627：拒绝：evidence_spans[0].quote 不是原文逐字子串（疑似编造引用）：'比上年同期增长:50% - 60%'
