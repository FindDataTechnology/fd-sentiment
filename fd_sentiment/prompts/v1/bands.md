五档定义（band 必须取其中之一，且与 score 的区间一致）：
- strong_bullish：强正面。明确的大幅改善/超预期/重大利好，且措辞强度高。
- bullish：偏正面。温和改善、略超预期、一般性利好。
- neutral：中性。无明确方向，或正负相抵，或纯事务性公告（如会议通知、章程修订）。
- bearish：偏负面。温和恶化、略低于预期、一般性利空。
- strong_bearish：强负面。明确的大幅恶化/重大风险/重大利空，且措辞强度高。

score 与 band 的对应（必须自洽）：
  score >= 0.6 → strong_bullish
  0.2 <= score < 0.6 → bullish
  -0.2 < score < 0.2 → neutral
  -0.6 < score <= -0.2 → bearish
  score <= -0.6 → strong_bearish

confidence 是"你对这个判断有多确定"的 0..1 数值，不是情绪强度。
证据不足时给低 confidence，不要用中性分数掩盖不确定。
