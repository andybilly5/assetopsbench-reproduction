# 代码整理与验证记录

验证时间：2026-11-14。本记录针对从既有实验备份整理出的代码快照；没有重新训练或重新执行真实模型评测。

## 已完成的工程整理

- 采用 src 布局，生产代码、测试、数据和文档分开管理。
- Ruff 统一格式及导入顺序，启用 E4、E7、E9、F、I 规则；行宽目标 100。
- 根级旧版 ReAct 文件改成兼容转发，修复原先无效的父级相对导入，消除重复实现。
- 删除未使用的局部变量与导入，修改测试中含混的变量名。
- 两项测试使用 pytest tmp_path，取消写死的 Linux /tmp 路径。
- 更新旧版 README、CLI 注释，明确当前编排已实现、判官替换及工具扩展范围。
- 增加中文架构说明、Mermaid 调用图、复现边界、开发规范和环境变量示例。
- 增加 GitHub Actions，覆盖代码检查、格式、单元测试和合成数据端到端冒烟。
- 使用单独仓库目录和忽略规则，不纳入私钥、个人文档、虚拟环境、模型权重、日志及原始服务器备份。

## 本机验证结果

| 检查 | 结果 |
|---|---|
| Ruff lint | 通过 |
| Ruff format --check | 通过，57 个 Python 文件 |
| pytest | 135 passed |
| 离线冒烟 | 通过：运行 → 轨迹 → 静态评估，1 条合成样例通过 |
| wheel 构建与安装后验证 | 构建成功，安装后的包通过全部 135 项测试和离线冒烟 |
| 数据覆盖 | 152 total / 52 ready / 28 generic / 72 blocked / 80 executable |
| 原场景静态标准答案 | 0；静态冒烟成功不代表公开场景完成率 |
| 常见凭据模式扫描 | 0 个命中；仅为本次检查，不代表正式安全审计 |

开发工具版本：pytest 9.2.0、Ruff 0.17.5。2026-11-15 已上传至 GitHub 私有仓库；提交 ef56c769ed41b0b38d1282a2f0fb7a05c4628672 的远程 CI 全部通过，包含 lint、格式检查、135 项单元测试和离线冒烟验证。

仓库：[andybilly5/assetopsbench-reproduction](https://github.com/andybilly5/assetopsbench-reproduction)

远程验证：[GitHub Actions 运行记录](https://github.com/andybilly5/assetopsbench-reproduction/actions/runs/43653313894)。

现有实验实现仅做格式、兼容导入和文档整理，测试的临时路径修复不改变实验算法。判官数值逻辑、策略实现、工具算法保持原样；数据只统一文本换行，字段与记录不变。数据文件校验值见 DATA_SHA256.json。
