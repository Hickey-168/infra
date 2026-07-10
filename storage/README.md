# Storage

存储目录关注模型、数据、checkpoint、缓存。它不像 kernel 那么显眼，但经常决定训练能不能恢复、serving 能不能快速扩容、数据 pipeline 会不会饿死 GPU。

## Structure

- `weights`: safetensors、bin、metadata、mmap。
- `datasets`: JSONL、Parquet、Arrow、WebDataset。
- `checkpoint`: single-rank/distributed checkpoint、resharding。
- `cache`: model cache、dataset cache、eviction、versioning。
- `object-store`: S3/OSS/GCS consistency、retry、multipart。

## Exit Bar

你应该能设计一个可恢复训练 checkpoint，解释哪些状态缺失会导致不可复现，并能 benchmark 不同数据格式的读取吞吐。

