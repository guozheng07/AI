# 词汇总揽
<img width="2162" height="942" alt="image" src="https://github.com/user-attachments/assets/302ad95c-2638-4d05-b948-367d800b9009" />

# 调用关系一：Agent → Prompt → SKILL → Sub-Agent → Sub-Skill
<img width="684" height="1542" alt="image" src="https://github.com/user-attachments/assets/91b499ef-b990-40bf-b351-a93f255da55d" />

# 调用关系二：SKILL 决策 → CLI / MCP+Tool / Script
<img width="890" height="1290" alt="image" src="https://github.com/user-attachments/assets/f5e16f9c-e227-4da8-aaf4-1973770742b4" />

# 两图关键理解
图一：调度层 — Agent 是决策中枢，接单后拆解任务，按专业方向分派给各 Sub-Agent；每个 Sub-Agent 持有各自专属的 SKILL，SKILL 的 Meta 提纲决定命中哪个 Sub-Skill；Sub-Agent 按 Sub-Skill 执行完毕后，结果统一汇总回主 Agent 交付客户。

图二：执行层 — SKILL 读取 Meta 提纲后做路径决策，三条路各有侧重：
- CLI 直联：直接拨通外部 API，快速临时，无封装
- MCP + Tool：SKILL 指定 MCP 目录中的具体 Tool，Tool 封装后调用 API，标准规范可复用
- Script 直调：手册附带专用表单脚本，定制化直接对接 API，灵活独立

两图衔接：图一中每个 Sub-Agent 打开 SKILL 后，SKILL 的内部决策逻辑即为图二所示。

# 完整流程串联
客户发来需求说明书（Prompt），主 Agent 接单后自主分析任务，判断是否需要拆分：
- 简单任务 → Agent 直接查阅自己的 SKILL，SKILL 根据情况（图二）选择 CLI / MCP+Tool / Script
- 复杂任务 → Agent 拆分子任务（图一），分派给多个专业同事（Sub-Agent），每人持有各自的 SKILL；各 Sub-Agent 的 SKILL 同样走图二的决策路径，必要时再指定 Sub-Skill 处理更细分领域

所有 Sub-Agent 完成后，结果统一汇总回主 Agent，由主 Agent 整合输出最终结果交付客户。整个过程消耗的工时和费用就是 Token，嵌套层级越深，消耗越多。
