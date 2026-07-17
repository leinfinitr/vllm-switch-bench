# Result artifact policy

`results/` 只保留每个实验族的最新 curated 输出，不是所有本地运行的日志仓库。

## 可提交内容

- 小型 `metadata.json`、`summary.json/CSV` 和机器可读 curated analysis；
- 论文图和生成该图所需的最小输入；
- 明确记录失败/unsupported 的结果，避免只保留成功样本。

每个可引用 run 至少应记录：命令参数、模型、代码 commit/dirty 状态、软件/硬件环境、原始指标、自动 correctness assertions。物理内存回收结论必须同时包含 application accounting 和 worker RSS/host `MemAvailable`。

没有公开 raw artifact、独立重复和离散度的单点 A/B 只能标记为本地
observational evidence，不得使用 causal 或 paper-grade 措辞。论文级比较应发布
带 checksum 的原始 summary/steps，并报告多次独立运行的中位数和 IQR 或置信区间。

## 不直接提交的内容

- JSONL event stream、server/controller log、临时 YAML；
- 重复运行和被 supersede 的 timestamp 目录；
- 模型、trace 或其他大文件。

这些内容应放在本地 ignored 目录，或发布到带 checksum/永久 URL 的 artifact archive。论文引用的 raw run 若不在仓库内，curated summary 必须标记其 availability；本地相对路径不能视为公开可复现证据。

## 历史 schema

历史 run 保持原始字段和文件名，不为适配新代码而改写。当前 repeated-sleep harness 输出：

```text
repeated_sleep_l1_summary.json
repeated_sleep_l1_steps.csv
```

旧 `phase1_two_model_*` 文件只用于审计既有实验。后处理工具应显式选择输入，不依赖“最新目录”猜测。
