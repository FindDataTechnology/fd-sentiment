分块合并规则（v1，确定性）：
1. 长文档按段落边界切块，每块独立打分，块内证据照常给出。
2. 合并分数 = 各块 score 按块字符数加权平均。
3. 合并 confidence = 各块 confidence 的最小值（最弱一块决定整体可信度）。
4. 合并 band 由合并后的 score 按上述区间重新推导，不取各块 band 的多数票。
5. 证据：取 |score| 最大的块的前 3 条证据；输出标注 chunk_index/chunk_total。
