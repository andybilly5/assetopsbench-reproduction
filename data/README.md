# 数据说明

本目录从用户已有的 2026-11-14 实验备份复制，不在此次整理中联网下载。第三方数据的适用许可沿用其原始来源；本仓库未对第三方数据重新授权。

| 目录 | 内容 | 来源与限制 |
|---|---|---|
| normalized | assets、sensors、readings 三张 CSV | 既有公开样例的规范化结果：3 个资产、45 个传感器、136,805 条有效读数 |
| scenarios | all_utterance.jsonl | 备份中的 152 个公开场景；不等同于论文报告的 141 个场景 |
| kb | FMSR 知识库与 TSFM 能力目录 | 自建或重构的知识/能力描述；各文件内保留 provenance，不代表上游完整知识库或预训练权重 |
| wo | 合成工单、生成脚本与 PROVENANCE.json | 自建 26 条工单，不是上游约 4.2K 工单 |

`normalized` 表头：

```text
assets.csv:   site,asset_id,asset_class
sensors.csv:  site,asset_id,sensor,unit
readings.csv: site,asset_id,sensor,ts,value
```

缺失数据由 DataUnavailable 显式返回，不填充虚构实测值。数据筛选、知识库切换与工具扩展均会影响任务覆盖，不能直接视为模型能力变化。

自建工单生成：在仓库根目录执行 `python data/wo/gen_workorders.py`。该脚本会覆盖同目录的 synthetic_workorders.json；运行前如有自定义修改请另存。它只读取本地目录字段，不联网。

独立的冒烟夹具位于 examples/smoke/，全部为合成数据，用于离线工程验证。
