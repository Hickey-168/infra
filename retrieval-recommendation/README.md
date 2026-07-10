# Retrieval Recommendation

这个目录保留脑图里的搜推分支。它不是 LLM infra 主线的第一优先级，但和 embedding、向量检索、特征存储、在线服务、排序重排、LLM4Rec 有交叉。

## Structure

- `retrieval`: embedding、ANN、HNSW/IVF/PQ、recall metrics。
- `ranking`: cross-encoder、learning-to-rank、calibration。
- `recall-strategies`: i2i、u2i、u2i2i、vector recall、hybrid recall。
- `features`: online/offline consistency、feature drift。
- `llm4rec`: LLM rerank、explanation、session understanding。

## Priority Notes

如果你的短期目标是大模型 infra，搜推可以作为 P1/P2；如果你的业务主线是搜索推荐，它可以提前。建议先学 embedding retrieval metrics 和 ANN index，因为它们和 RAG/LLM serving 也共用。

