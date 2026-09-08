# 足球准确率改造：实现与运行说明

更新日期：2026-09-08。生产版本 `football-v2026.09.08-agent-research-18`，预测逻辑版本 `2026-09-08-agent-research-v44`。

本次在现有流程上增加可验证的训练、情报和评估闭环。没有改动主比赛卡片的三栏布局和整场胜平负百分比；深度报告的融合权重改为实际执行口径。以下功能已落地到本地代码，不能据此宣称未来比赛命中率已经提高。

## 已实现的四项改造

1. **真实训练与融合。** 训练和推理共用 40 项特征合同，采用真实 5/10 场及主客场历史窗口。缺字段、样本不足、训练截止未知、旧制品不兼容时停用 ML，不用零值伪造特征。修正训练准确率维度问题。最终比分矩阵实际执行通过门禁的 5% ML 融合，胜平负、让球和比分等输出随后从同一矩阵计算；记录加载、执行、准入、应用及实际权重。是否提高融合比例必须另行验证，不能只看样本数量。
2. **概率校准与时间隔离。** 修正 Platt 无操作参数不保持原概率的问题，避免重复校准；历史校准使用预测截止前已结算的数据。相似盘口查询支持排除本场及未知/未来可用时间。旧版未记录可用时点的相似盘口库不能作为严格历史重放的证据，现有运行中的相似盘口仍主要作参考和风险输入。
3. **结构化赛前情报。** `MatchContext` 汇集伤停、首发、新闻、赛程、H2H、天气，逐条保存来源、发布时间、采集时间、确认状态和冲突。主流程读取已完成缓存并投递后台研究任务；缺失、超时和不合格证据会明确降级。可接本地 Ollama 做文本提取，LLM 不输出最终概率、影响分或融合权重。只有可核实事实进入统计特征。
4. **冻结、结算、对照和选权。** 每场第一次真正赛前保存的预测作为固定评估事件，后续重算追加观察，不能覆盖原事件；已结算记录不能被赛后分析改写。重复同步不重复喂入 Elo/校准。增加 Top1/3/5/10、真实比分排名、Brier、LogLoss、分母和缺失原因；使用同一赛前事件比较纯统计 A、统计加市场 B、B 加情报 C、去水市场 M 和实际生产结果。情报残差模型采用按日期划分的训练/验证/测试与可用时间隔离，验证段选权，测试段只评价。

## 影子实验与准入

- A 是不读盘口的球队 Poisson；B 在 A 上复用现有盘口锚定。球队样本不足时 A/B/C 标记不可用，不把生产结果冒充纯统计基准。
- A/B/C 使用同一版本的恒等校准协议，以便隔离信息增量；实际生产流程保持已有独立校准，并另存生产分布。
- 当前没有合格情报制品时 **C = B、实际情报权重为 0**。报告区分全部回退样本与真正应用情报的样本，不能把两者混为 Agent 提升。
- 情报候选需真实训练至少 200 场、选权至少 100 场、独立测试至少 1,000 场且至少 30 个开赛日期，并通过按日期配对的置信区间门禁。候选选权范围 0/2.5%/5%/10%。制品有字段、版本、时间和指纹检查；不足则不启用。CLI 训练只写候选，不自动上线。
- ML 使用现有门禁：制品测试至少 200 场，同制品赛前冻结影子配对至少 100 场，5% 混合 LogLoss 改善且 Brier 不变差。老导出中未经冻结的 ML 预测不能充数。
- 评估按生产版本拆分，提供按开赛日期的配对区间。描述性统计不等于完整模型发布验收，`release_qualified` 不会仅因有命中率而变为真。

## 本地操作

在项目根目录执行，示例文件名须替换为真实导出文件：

```powershell
.venv/Scripts/python.exe scripts/tools/football_research.py status
.venv/Scripts/python.exe scripts/tools/football_research.py evaluate --input reports/new-frozen-export.json --output reports/research-evaluation.json
.venv/Scripts/python.exe scripts/tools/football_research.py train-intelligence --input reports/new-frozen-export.json --model-version football-v2026.09.08-agent-research-18 --output data/intelligence-candidate.json
.venv/Scripts/python.exe scripts/tools/research_match_context.py --status
```

不传 `--input` 时，评估/训练读取项目当前配置的历史存储。训练命令不会修改运行中的模型。API 历史统计包含 `frozen_event_evaluation`，导出保留完整事件；详细诊断无需重新放回主 Web 页面。

可选环境配置：

| 配置 | 用途 |
| --- | --- |
| `FOOTBALL_INTELLIGENCE_ENABLED=0` | 停止提交新的后台研究任务 |
| `FOOTBALL_INTELLIGENCE_SOURCES` | 结构化 JSON/RSS 来源配置文件 |
| `FOOTBALL_OLLAMA_MODEL` | 已安装的本地模型名；没有配置则不调用 LLM |
| `FOOTBALL_OLLAMA_URL` | 默认 `http://127.0.0.1:11434`，只允许回环地址 |
| `FOOTBALL_INTELLIGENCE_MODEL` | 经验证的情报残差 JSON 制品路径 |
| `FOOTBALL_ML_FUSION_ENABLED=0` | 保留 ML 影子预测，关闭生产融合 |
| `FOOTBALL_INTELLIGENCE_GDELT=0` / `FOOTBALL_INTELLIGENCE_WEATHER=0` | 关闭默认免费新闻发现/有球场坐标时的天气查询 |

GDELT 的发现时间不冒充文章发布时间，Open-Meteo 的预报有效时间不冒充发布时间。它们在时间证据不足时仅留作未确认情报，不能提高置信度。配置源可补充可靠发布时间和确认状态。Ollama 原文提取固定为 reported，独立确认前也不参与统计特征。

## 当前验证和边界

- 最终足球核心与 API 联合回归：926 项测试、19,368 个子测试通过。覆盖常规五路数据源与官方竞彩独立路径、缺模型回退、校准执行、概率一致性、真实比分排名、冻结防篡改、结算幂等、证据冲突及训练/测试隔离；历史球队统计与 ML 共用结果观察时点和排除规则，拒绝给历史短日期猜年份。
- 用户提供的旧导出 837 条全部因没有真实冻结事件而排除出新实验，不能回填成赛前记录。旧指标可用于描述问题，不能证明本次改造效果。
- 本轮未安装 Ollama 或下载模型；状态检查显示尚未配置本地 LLM，也没有合格的情报残差模型。代码支持接入，但不能说情报模型已经生效。
- 本轮未执行 MySQL 迁移、生产服务重启或部署。正在运行的旧服务需重新加载本地代码后，新预测才会开始积累新版本事件；历史数据不伪造补齐。
- 结算派生数据采用现有单进程锁和持久化领取标记保证不重复应用，不是跨数据库事务；中断或赛果订正会留待重建标记，不悄悄重复训练。

入口索引：`pipeline.py`（接入）、`research.py`（A/B/C/M 与最终 ML）、`research_model.py`（情报训练/门禁）、`research_runtime.py`（运行适配）、`prediction_events.py`（不可变事件）、`../domain/sports/football/prediction_evaluation.py`（配对评估）、`intelligence/`（证据采集）、`ml_features.py`（训练推理合同）。
