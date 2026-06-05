# UI 驱动 Harness E2E 执行指南

> **适用项目**：`osg-fe-store-web` — 仓容概览看板（Capacity Dashboard）  
> **最后更新**：2026-06-04  
> **关联资源**：
> - PRD：https://km.sankuai.com/collabpage/2749938366
> - 技术文档：https://km.sankuai.com/collabpage/2761200038
> - 设计稿：https://imd.sankuai.com/file/189270363591816?file=189270363591816&page_id=21%3A29770&shareId=8c7ad16d-eb97-41bc-8069-497b2cec5fdf&layer_id=21%3A32747%2F26%3A22955
> - 手动验收 / 自动化测试页面（Chrome Beta，SSO 登录）：http://localhost:8418/store-alias/dashboard.html
> - 兜底泳道环境（本地接口异常时使用）：https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store/dashboard.html
> - 源码目录：`src/pages/dashboard`
> - 泳道：`baihaoxiang-eutnc`（已在 `config/config.ts` 配置代理）

---

## 零、前置约束与全局记忆

> **最高优先级**：本节所有约束是 Agent 执行 Harness 循环的**硬性前提条件**，任何步骤均不得违反。在开始执行前，Agent 必须完整读取并记忆本节内容。

### 0.1 端口约束（关键，禁止混用）

| 用途 | 端口 | 启动方式 | SSO | 说明 |
|------|------|---------|-----|------|
| **手动验收** / Chrome Beta 调试 | **8418** | `pnpm run dev:sso`（`PORT=8418 o dev`） | 必须 | 通过真实美团 SSO 登录，接 `baihaoxiang-eutnc` 泳道真实数据 |
| **Playwright 自动化测试（Layer 1/2/3）** | **8418** | 由 `playwright.config.ts` 的 `webServer` 自动调用 `pnpm run dev:sso` | 必须 | `global-setup` 预先完成 SSO 登录，Cookie 保存至 `tests/.auth/user.json`，测试通过 `storageState` 复用 |
| **CI 烟雾测试（可选）** | **8420** | `pnpm run dev:test`（`DISABLE_SSO=true PORT=8420`） | 无需 | 仅做页面骨架可访问性验证，接口数据为空，**不用于正式 Harness** |

**强制规定**：
- Agent 执行 Harness 期间，**Playwright 测试命令统一使用 8418 端口**
- `playwright.config.ts` 的 `webServer` 配置会在测试前自动启动 `pnpm run dev:sso`（即 `PORT=8418 o dev`），若 8418 已有服务则直接复用（`reuseExistingServer: true`）
- 推荐在运行测试前先手动执行 `pnpm run dev:sso`，节省 webServer 启动时间
- 若 8418 端口已有服务运行（`reuseExistingServer: true`），Playwright 会直接复用，**无需重复启动**

### 0.2 SSO 登录约束

- **Playwright 自动化测试**（Harness 主流程，Layer 1/2/3）：**必须经过真实美团 SSO 登录**，所有测试均通过 `storageState: 'tests/.auth/user.json'` 复用登录态 Cookie，接口才能返回真实泳道数据
- **global-setup.ts**：在所有测试开始前自动运行一次，检查 `tests/.auth/user.json` 是否在 8 小时有效期内。若不存在或已过期，弹出有界面浏览器（`headless: false`）等待用户手动完成美团 SSO 登录，登录成功后保存 Cookie
- **Layer 1 `dashboard.spec.ts`**：直接调用真实泳道接口，**不使用 `page.route()` Mock**，断言逻辑验证元素存在性与结构，数值断言依赖真实数据
- **Layer 3 Midscene AI 视觉测试**：同样使用真实 SSO Cookie + 真实泳道数据，**严禁使用 `page.route()` Mock**
- **`DISABLE_SSO=true` / 8420 端口**：仅用于 `dev:test` 模式的烟雾测试，**不属于正式 Harness 流程**

### 0.3 泳道与代理约束

- 泳道标识：`baihaoxiang-eutnc`
- 代理配置文件：`config/config.ts`（已配置，**不得修改**）
- 代理目标：`https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com`
- 代理路径：`/api/*`、`/shepherd/*`、`/shop/*`
- **禁止**：为了让测试通过而修改代理配置、切换泳道或注入假数据到 Layer 3

### 0.4 Agent 上下文轮换约束（主 Agent + SubAgent 协作模式）

**上下文压缩次数限制**：单个主 Agent 会话的 **Context Compaction（上下文压缩）次数不得超过 3 次**。

| 压缩次数 | 动作 |
|---------|------|
| 第 1 次压缩发生 | 继续执行，在当轮结束后更新 `docs/harness-progress.json` |
| 第 2 次压缩发生 | 继续执行，每次代码修改后立即更新进度文件 |
| 第 3 次压缩发生 | 完成当前最小可完成任务后，立即执行**主动切换流程**（见 9.5 节） |
| 第 4 次压缩即将发生 | **禁止继续当前 Agent**，必须已在第 3 次压缩后完成切换 |

**SubAgent 拆分规范**：
- 当 Layer 1 出现跨模块失败时，主 Agent 拆分 SubAgent 并行处理，各自修复后汇总
- 每个 SubAgent 完成后必须向主 Agent 汇报：`修改了哪些文件 + lint/type-check 结果`
- SubAgent 的上下文压缩次数独立计算，**不影响主 Agent 计数**

**关键资源（全局记忆，每次 Agent 恢复时必须重新确认）**：

```
PRD：         https://km.sankuai.com/collabpage/2749938366
技术文档：    https://km.sankuai.com/collabpage/2761200038
设计稿：      https://imd.sankuai.com/file/189270363591816
手动验收：    http://localhost:8418/store-alias/dashboard.html（pnpm run dev:sso）
自动化测试：  http://localhost:8418/store-alias/dashboard.html（Playwright 统一走 8418，SSO 必须）
源码目录：    src/pages/dashboard
泳道：        baihaoxiang-eutnc
SSO Cookie：  tests/.auth/user.json（global-setup 登录后生成，有效期 8 小时）
```

---

## 一、整体概念：什么是 Harness

**Harness**（测试驱动开发脚手架）是指将"测试用例 → 代码修改 → 验证 → 继续修复"组织成一个**自动闭环**的开发工作流，目标是让 Agent/开发者能在**无人工干预**的情况下，完成从初始缺陷到所有用例全绿的完整循环。

```
            ┌─────────────────────────────────────────────────┐
            │                  Harness 循环                    │
            │                                                 │
            │   运行测试（Layer 1 → 2 → 3）                    │
            │      ↓                                          │
            │   收集失败用例 + 截图 + Service 状态日志          │
            │      ↓                                          │
            │   定位根因（DOM? 逻辑? API?）                     │
            │      ↓                                          │
            │   修改 src/pages/dashboard/ 代码                 │
            │      ↓                                          │
            │   lint + type-check 门禁                         │
            │      ↓                                          │
            │   重新运行测试                                    │
            │      ↓                                          │
            │   ✅ Layer 1~3 全绿 → 进入 Layer 4 UI 对比       │
            │      ↓                                          │
            │   截图对比设计稿 → 标注样式偏差 → 修复 TSX/CSS    │
            │      ↓                                          │
            │   ✅ UI 对比通过则退出循环 / ❌ 否则继续修复       │
            └─────────────────────────────────────────────────┘
```

### 与 TMS 项目实践的对照

参考美团 TMS 项目 Harness 实践文档（https://km.sankuai.com/collabpage/2764956067）的核心经验：

| TMS 实践经验 | 本项目对应落地 |
|-------------|--------------|
| 代码输出环节通过 lint + ts-checker 确保规范 | `pnpm run lint && pnpm run type-check` 作为每轮修复后的强制门禁 |
| Agent 通过 web mcp 操作 RSJS 状态、截图获取反馈 | `window.__dashboardService` 全局挂载 + `page.evaluate()` 直接断言 |
| 先冒烟用例、再 web mcp 验证，避免低效视觉操作 | 四层测试策略：精确断言 → Service 状态断言 → AI 视觉回归 → **设计稿 UI 对比** |
| 技术方案质量是效率关键 | 技术方案 + PRD + 设计稿三文档全局记忆，代码修改有依据 |
| 使用 evaluate_script 比 click 等操作效率更高 | Playwright `page.evaluate()` 优先于 `aiAct()` 访问内部状态 |
| UI 还原整体接近但仍存在间距、大小、字号问题 | Layer 4 专项对照设计稿截图逐模块检查并修复样式偏差 |

---

## 二、四层测试策略

### Layer 1：真实接口精确断言（~5 分钟）

**工具**：Playwright 原生 (`@playwright/test`)，Chrome Beta（`headless: false`）  
**文件**：`tests/dashboard.spec.ts`  
**命令**：`pnpm test:e2e:offline`

| 特点 | 说明 |
|------|------|
| 真实接口 | 直接调用 `baihaoxiang-eutnc` 泳道后端，**不使用 `page.route()` Mock** |
| SSO Cookie | 通过 `storageState: tests/.auth/user.json` 注入登录态，接口返回真实数据 |
| 精确 DOM 断言 | CSS 选择器、role、text，验证元素存在性、结构和颜色类名 |
| Service 状态断言 | `page.evaluate(() => window.__dashboardService?.xxx)` 直接验证 RSJS 状态 |
| Chrome Beta | `channel: 'chrome-beta'`，`headless: false`，可见浏览器运行 |
| 异常态降级 | 接口返回空数据时 Service 层降级展示 `--` 占位符，测试验证不崩溃 |

**适用场景**：每次代码修改后的第一道门禁（需先完成 SSO 登录，Cookie 有效期 8 小时）。

---

### Layer 2：RSJS Service 状态断言（< 1 分钟）

**工具**：Playwright + `window.__dashboardService`  
**文件**：`tests/dashboard.spec.ts` Suite H  
**命令**：`pnpm test:svc`

这是参考 TMS Harness 实践中 `evaluate_script` 最佳方案的核心落地：

```typescript
// 直接访问 RSJS Service 状态，不依赖视觉识别
const dimension = await page.evaluate(() => {
  return window.__dashboardService?.currentDimension;
});
expect(dimension).toBe('2');
```

**`window.__dashboardService` 可访问的关键状态**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `currentPoiId` | `string` | 当前选中门店 ID，初始为 `'10010'` |
| `currentDimension` | `'1' \| '2'` | 当前维度，`'1'` = PCS，`'2'` = 体积 |
| `utilizationData` | `UtilizationData \| null` | 利用率接口返回数据 |
| `alertsData` | `AlertsData \| null` | 预警数据 |
| `capacityDetailData` | `CapacityDetailData \| null` | 容量详情（温区+库区） |
| `floorInfoData` | `FloorInfoData \| null` | 楼层信息 |
| `overviewData` | `OverviewData \| null` | 概览信息（门店名、楼层数等） |
| `cadFloorMapData` | `CadFloorMapData \| null` | CAD 解析后的楼层地图数据 |

**暴露机制**（`src/pages/dashboard/index.tsx` + `dashboard.service.ts`）：

> ⚠️ **重要**：`@osgfe/rs-react` 的 RSJS Service 基类**没有** `init()` 生命周期钩子！
> 必须在消费该 Service 的 React 组件中，通过 `useEffect` 手动调用 `mountToWindow()` 来挂载：

```typescript
// src/pages/dashboard/index.tsx（消费端组件）
useEffect(() => {
  // 将 Service 实例挂载到 window.__dashboardService，供 E2E 测试访问
  dashboardService.mountToWindow();
}, []);
```

```typescript
// src/pages/dashboard/dashboard.service.ts（Service 侧提供方法）
mountToWindow() {
  if (process.env.NODE_ENV !== 'production') {
    (window as any).__dashboardService = this;
  }
}
```

> ⚠️ 生产环境 webpack DefinePlugin 会将 `process.env.NODE_ENV` 替换为 `'production'`，此代码段会被 tree-shaking 掉，**不会暴露到线上**。
> 若 `page.evaluate(() => window.__dashboardService)` 返回 `undefined`，优先检查 `index.tsx` 的 `useEffect` 是否调用了 `mountToWindow()`，以及 `useEffect` 依赖数组是否为空（`[]`）确保仅在挂载时执行一次。

---

### Layer 3：Midscene AI 视觉回归（~10 分钟）

**工具**：Midscene.js (`@midscene/web`) + Playwright  
**文件**：`tests/dashboard-midscene.spec.ts`  
**命令**：`pnpm test:e2e:ai`

| 特点 | 说明 |
|------|------|
| 视觉模型驱动 | 不依赖 CSS 选择器，AI 理解截图后判断 |
| 自然语言描述 | `aiAssert('Alerts 模块包含 Temp zones 预警区域')` |
| 结构化数据提取 | `aiQuery<number>('获取利用率百分比数字')` |
| 每步自动截图 | 生成 `midscene_run/report/` 可视化报告 |
| 直连真实数据 | 不 Mock，直接对接测试环境或本地 dev server |
| UI 变更不影响 | 布局调整不需要修改测试脚本 |

**fixture 速查（`tests/fixture.ts`）**：

| Fixture | 用途 | 示例 |
|---------|------|------|
| `ai(action)` | 自然语言操作 | `await ai('点击 Volume 切换按钮')` |
| `aiAssert(assertion)` | 视觉断言 | `await aiAssert('按钮处于高亮状态')` |
| `aiQuery<T>(dataDef)` | 提取结构化数据 | `await aiQuery<number>('获取利用率数字')` |
| `aiWaitFor(condition)` | 等待条件成立 | `await aiWaitFor('数据加载完成', { timeoutMs: 8000 })` |
| `aiTap(target)` | 点击元素 | `await aiTap('排序按钮')` |
| `aiInput(text, target)` | 输入文本 | `await aiInput('RT', '搜索框')` |
| `recordToReport(title)` | 记录截图快照 | `await recordToReport('切换后状态')` |

---

### Layer 4：设计稿 UI 对比与样式修复（~15 分钟）

**工具**：Midscene.js 截图 + 设计稿人工/AI 比对  
**触发时机**：Layer 1/2/3 全部通过后执行  
**设计稿链接**：`https://imd.sankuai.com/file/189270363591816`  
**命令**：`pnpm test:layer4:ui`（独立 `layer4` project，依赖 `midscene` project 通过；若想跳过依赖链直接运行用 `pnpm test:layer4:ui --no-deps`）

| 特点 | 说明 |
|------|------|
| 截图驱动 | 通过 Midscene `recordToReport()` 对每个模块截图，生成 `midscene_run/report/` |
| 逐模块比对 | 对照设计稿检查间距、字号、颜色、布局是否还原 |
| 样式偏差修复 | 直接修改 TSX 的 TailwindCSS 类名或 CSS Module，**不修改功能逻辑** |
| 迭代收敛 | 修复后重新截图比对，直至视觉差异在可接受范围内 |
| 结果记录 | 将各模块 UI 对比结论记录到 `docs/harness-progress.json` 的 `uiReview` 字段 |

**Layer 4 对比维度（每个模块必须检查）**：

| 维度 | 检查项 | 设计稿参考位置 |
|------|--------|---------------|
| 颜色 | 预警色（橙/红）、背景色、文字色与设计稿色值一致 | 设计稿标注面板 |
| 间距 | padding、gap、margin 与设计稿标注一致（允许 ±4px 误差） | 设计稿 inspect 模式 |
| 字号/字重 | 标题、数值、描述文字的 fontSize、fontWeight | 设计稿 inspect 模式 |
| 圆角/边框 | 卡片圆角、分割线颜色与粗细 | 设计稿 inspect 模式 |
| 图标 | SVG 图标尺寸、颜色与设计稿一致 | 设计稿资源面板 |
| 响应式布局 | 在 1440px 宽度下与设计稿比例匹配 | 设计稿画布尺寸 |

**各模块设计稿对照位置**：

| 模块 | TSX 文件 | 设计稿 Layer |
|------|---------|-------------|
| 整体布局 | `index.tsx` | 主画布 Dashboard |
| 利用率看板 | `utilization-module.tsx` | Utilization 组件 |
| 预警模块 | `alerts-module.tsx` | Alerts 组件 |
| 容量详情 | `capacity-detail-module.tsx` | Capacity Detail 组件 |
| 楼层可视化 | `floor-visualization-module.tsx` | Floor Visualization 组件 |
| 门店切换头部 | `header-module.tsx` | Header 组件 |

**Layer 4 执行步骤**（Agent 必须按顺序执行）：

```
Step 1：用 Midscene recordToReport() 对所有模块截图
        await recordToReport('Layer4-整体布局');
        await recordToReport('Layer4-利用率看板');
        await recordToReport('Layer4-预警模块');
        await recordToReport('Layer4-容量详情');
        await recordToReport('Layer4-楼层可视化');
        await recordToReport('Layer4-门店切换头部');

Step 2：打开 midscene_run/report/ 截图，与设计稿逐模块对比
        重点关注：颜色、间距、字号、圆角、图标尺寸

Step 3：用 aiAssert 进行 AI 辅助验证
        await aiAssert('预警模块的标题 Alerts 字号明显大于正文，左侧有橙色闪电图标');
        await aiAssert('利用率模块的进度条颜色与利用率百分比匹配（高利用率偏红）');
        await aiAssert('容量详情卡片有明显的圆角和边框，各指标数据对齐');

Step 4：列出所有发现的样式偏差，按严重程度分级
        🔴 严重（明显色差、布局错乱）→ 必须修复
        🟡 一般（间距偏差 > 8px、字号差 2 级以上）→ 本轮修复
        ⚪ 轻微（间距偏差 ≤ 4px）→ 记录但本轮可接受

Step 5：修复 🔴 严重 和 🟡 一般 级别的样式问题
        - 优先修改 TailwindCSS utility class（如 text-base → text-xl）
        - 必要时查阅 sailor-keeta-design-pc Skill 确认组件 Token
        - 修改后执行 pnpm run lint 确认无错误

Step 6：重新截图，确认修复效果
        重复 Step 1 截图 → 与设计稿再次比对

Step 7：将结论写入 harness-progress.json
        更新 uiReview 字段，记录各模块状态和遗留问题
```

**适用场景**：Layer 1~3 全绿后的最后一道 UI 质量门。TMS 实践经验表明，AI 开发完成功能后 UI 整体接近设计稿但仍存在细节偏差，通过此层可系统性消灭间距/字号/颜色等样式问题。

---

## 三、完整 Harness 工作流

### 3.1 环境准备

#### 步骤 1：启动开发服务器

**场景 A：手动验收 / Playwright 自动化测试（统一 8418 端口，真实 SSO）**

```bash
# 终端 1 — 推荐先手动启动，再执行测试命令
cd osg-fe-store-web
pnpm run dev:sso
# 等待输出 "webpack 5.x compiled successfully"
# 手动验收 / 自动化测试页面（需 SSO 登录）：http://localhost:8418/store-alias/dashboard.html
```

> **注意**：`playwright.config.ts` 的 `webServer` 配置会在运行测试命令时自动以 `pnpm run dev:sso`（`PORT=8418 o dev`）启动服务，若 8418 端口已有服务则直接复用。

**场景 B：烟雾测试（可选，8420 端口，无 SSO）**

```bash
# 仅用于 CI 烟雾测试或页面骨架验证，不属于正式 Harness 流程
pnpm run dev:test
# DISABLE_SSO=true PORT=8420 — 接口无 Cookie，数据为空，显示 "--"
# 页面（骨架验证）：http://localhost:8420/store-alias/dashboard.html
```

> **端口分工总结**：
> - `8418`：**手动验收 + Playwright 自动化测试（Layer 1/2/3）**，真实 SSO 登录，泳道 `baihaoxiang-eutnc` 代理真实后端，接口返回真实数据
> - `8420`：CI 烟雾测试专用，无 SSO，接口无数据（仅验证页面可访问、骨架不崩溃）

#### 步骤 2：配置 Midscene 模型（首次/换机）

在项目根目录创建 `.env` 文件（已在 `.gitignore` 中，**不会提交**）：

```bash
# 接入美团 Friday（内网，无需科学上网）
MIDSCENE_MODEL_BASE_URL=https://aigc.sankuai.com/v1/openai/native
MIDSCENE_MODEL_API_KEY=<你的 Friday App ID>
MIDSCENE_MODEL_NAME=gpt-4o
MIDSCENE_MODEL_FAMILY=openai
```

- App ID 申请：https://friday.sankuai.com/budget/serviceManage  
- 必须选择**支持视觉（多模态）的模型**：`gpt-4o` / `gpt-4.1` / `gpt-4o-mini`

#### 步骤 3：初始化认证状态（SSO 登录，必须）

```bash
# 首次运行或 Cookie 过期（8 小时）时执行
# 会弹出 Chrome 浏览器窗口，等待用户完成美团 SSO 登录
# 登录成功后自动保存 Cookie 到 tests/.auth/user.json
pnpm test:seed

# 也可以直接运行测试命令，global-setup 会自动触发 SSO 登录流程
pnpm test:e2e:offline
```

> ⚠️ **Cookie 有效期为 8 小时**。超过 8 小时后再运行测试，`global-setup` 会自动重新弹出浏览器要求重新登录。

---

### 3.2 浏览器选择规范

**Playwright 测试（Layer 1 精确断言 + Layer 2 SVC）** 使用系统安装的 **Chrome Beta**（`channel: 'chrome-beta'`），**默认以无头模式（`headless: true`）运行**，适合 CI 和 Agent 自动化；本地调试时通过 `HEADLESS=false` 环境变量或 `pnpm test:e2e:headed` 快捷脚本切换为有界面模式。

**Midscene AI 视觉测试（Layer 3 / Layer 4）** 由 `playwright.config.ts` 中 `midscene` / `layer4` project 配置的 `headless: false` 运行——Midscene 需要有界面渲染，特别是 Canvas / WebGL 内容，**不可改为 headless**。

**规则总结**：

| 层次 | headless 设置 | 浏览器 | SSO | 原因 |
|------|------------|--------|-----|------|
| Layer 1 精确断言 | **默认 `true`**（无头），`HEADLESS=false` 切换有界面 | Chrome Beta | 必须 | 默认无头适合 CI/Agent；本地调试加 `HEADLESS=false` |
| Layer 2 SVC 状态断言 | **同 Layer 1**，受 `HEADLESS` 变量控制 | Chrome Beta | 必须 | 同属 `chromium` project，规则一致 |
| Layer 3 Midscene AI 视觉 | **固定 `false`**，不受 `HEADLESS` 变量影响 | Chrome Beta | 必须 | AI 视觉需要渲染上屏、Canvas 需要有界面 |
| Layer 4 UI 视觉对比 | **固定 `false`**，不受 `HEADLESS` 变量影响 | Chrome Beta | 必须 | Midscene `recordToReport()` 需要有界面渲染 |
| `global-setup.ts` SSO 登录 | **固定 `false`** | Playwright Chromium | — | 需要用户在弹出窗口中完成美团 SSO 登录 |

> ℹ️ **headless 切换速查**：Layer 1/2 默认无头（`HEADLESS=true`）；本地调试用 `HEADLESS=false pnpm test:e2e:offline` 或 `pnpm test:e2e:headed`。Layer 3/4 及 global-setup 固定 `headless: false`，**不可修改**。

---

### 3.3 执行命令速查

| 命令 | headless | 适用场景 | 预计耗时 |
|------|---------|---------|---------|
| `pnpm test:e2e:offline` | ✅ 无头（默认） | Layer 1 精确断言门禁（CI / Agent 自动化） | < 2min |
| `pnpm test:e2e:headed` | ❌ 有界面 | Layer 1 本地调试，可观察浏览器 UI 变化 | < 2min |
| `pnpm test:svc` | ✅ 无头（默认） | Layer 2 Service 状态断言（验证 RSJS 逻辑） | < 1min |
| `pnpm test:svc:headed` | ❌ 有界面 | Layer 2 本地调试 | < 1min |
| `pnpm test:e2e:ai` | ❌ 有界面（固定） | Layer 3 AI 视觉回归 | ~10min |
| `pnpm test:layer4:ui` | ❌ 有界面（固定） | Layer 4 设计稿 UI 视觉对比（依赖 Layer 3 通过） | ~15min |
| `pnpm test:layer4:ui --no-deps` | ❌ 有界面（固定） | Layer 4 独立运行，跳过依赖链 | ~15min |
| `pnpm test:harness` | ✅ 无头（默认） | **完整 Harness 循环**（Layer 1 → 2 → 3） | ~12min |
| `pnpm test:harness:headed` | ❌ 有界面 | 完整 Harness 循环（有界面调试版） | ~12min |
| `pnpm test:midscene` | ❌ 有界面（固定） | 单独运行 Midscene 测试（不依赖 Layer 1） | ~10min |
| `pnpm test:wh:layer3` | ❌ 有界面（固定） | **仓容线上化** Layer 3 Midscene AI 视觉测试（28 个用例） | ~15min |
| `pnpm test:wh:layer4` | ❌ 有界面（固定） | **仓容线上化** Layer 4 设计稿 UI 对比（16 个用例） | ~15min |
| `pnpm test:wh:harness` | ❌ 有界面（固定） | **仓容线上化** Layer 3 + Layer 4 完整循环 | ~30min |
| `pnpm test:ui` | — | Playwright UI 模式（查看时间轴/截图） | 交互式 |
| `pnpm test:seed` | — | 重新初始化认证状态 | < 1min |

> ℹ️ **一键切换 headless**：在任意命令前加 `HEADLESS=false` 前缀即可将 Layer 1/2 切换为有界面模式，例如 `HEADLESS=false pnpm test:e2e:offline`。

---

### 3.4 Harness 循环执行步骤

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: 确认 dev server 在 8418 端口运行                          │
│         curl http://localhost:8418/store-alias/dashboard.html   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ Step 1: Layer 1 离线门禁                                          │
│         pnpm test:e2e:offline                                   │
│         ✅ 全绿 → 进入 Step 3                                     │
│         ❌ 有失败 → 进入 Step 2                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ 失败
┌────────────────────────────▼────────────────────────────────────┐
│ Step 2: 分析失败原因并修复代码                                      │
│                                                                 │
│  A. 查看 Playwright HTML 报告：                                  │
│     npx playwright show-report                                  │
│                                                                 │
│  B. 用 evaluate_script 检查 Service 状态：                       │
│     window.__dashboardService?.currentDimension                 │
│     window.__dashboardService?.utilizationData                  │
│                                                                 │
│  C. 修改 src/pages/dashboard/ 对应文件                           │
│                                                                 │
│  D. 强制门禁（修改完立即执行）：                                   │
│     pnpm run lint && pnpm run type-check                        │
│                                                                 │
│  E. 回到 Step 1 重新运行                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ Layer 1 全绿
┌────────────────────────────▼────────────────────────────────────┐
│ Step 3: Layer 2 Service 状态断言                                  │
│         pnpm test:svc                                           │
│         ✅ 全绿 → 进入 Step 4                                     │
│         ❌ 有失败 → 检查 RSJS Service 逻辑，修复后回到 Step 1      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ Step 4: Layer 3 AI 视觉回归                                       │
│         pnpm test:e2e:ai                                        │
│                                                                 │
│  运行完成后查看报告：                                               │
│     · Midscene 报告：midscene_run/report/（含每步 AI 截图）        │
│     · Playwright 报告：npx playwright show-report               │
│                                                                 │
│         ✅ 全绿 → Harness 循环完成                                │
│         ❌ 有失败 → 查看 Midscene 截图定位 UI 问题，修复后          │
│                    回到 Step 1                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、测试文件结构与用例分布

### 4.1 `tests/dashboard.spec.ts` — 离线精确测试

| Suite | 用例范围 | 核心验证点 |
|-------|---------|----------|
| **Suite A** · 页面基础加载与布局 | TC-F-001 ~ TC-F-002 | 四大模块可见、概览信息展示 |
| **Suite B** · Utilization 利用率看板 | TC-F-003 ~ TC-F-009 | PCS/Volume 维度、趋势图、满载天数、Unactivated 标签 |
| **Suite C** · Alerts 预警模块 | TC-F-010 ~ TC-F-016 | 三分区（温区/楼层/库区）、daysToFull 标签、预警计数 |
| **Suite D** · Capacity Detail 容量详情 | TC-F-017 ~ TC-F-022 | 温区卡片、库区搜索过滤、排序、状态标签 |
| **Suite E** · Floor Visualization 楼层可视化 | TC-E-001 ~ TC-E-008 | 楼层 Tab、信息栏数据、缩放控制 |
| **Suite F** · 维度切换 PCS ↔ Volume | TC-F-025 ~ TC-F-027 | 并发 API 触发、单位切换、Tab 高亮 |
| **Suite G** · 异常与边界场景 | TC-ERR-001 ~ TC-ST-004 | 接口 500、空数据降级、颜色阈值边界 |
| **Suite H** · RSJS Service 状态断言 | TC-SVC-001 ~ TC-SVC-006 | `window.__dashboardService` 状态验证 |
| **Suite I** · Header 门店切换 | TC-HDR-001 ~ TC-HDR-005 | poiName 显示、poiSearch 触发、空列表降级 |

---

### 4.2 仓容线上化测试文件总览

| 文件 | 类型 | 用例数 | 运行方式 | 说明 |
|------|------|---------|---------|------|
| `tests/warehouse-capacity-smoke-test-cases.md` | Markdown 文档 | 38 | 手动参考 | 冒烟测试用例，最小覆盖集 |
| `tests/warehouse-capacity-full-test-cases.md` | Markdown 文档 | 99 | 手动参考 | 完整测试用例，覆盖所有路径 |
| `tests/warehouse-capacity-layer3.spec.ts` | Playwright spec | 28 | `pnpm test:wh:layer3` | 仓容看板 Midscene AI 视觉测试 |
| `tests/warehouse-capacity-layer4.spec.ts` | Playwright spec | 16 | `pnpm test:wh:layer4` | 仓容看板设计稿 UI 对比 |

> **执行层次**：`warehouse-midscene` project 依赖 `chromium`（Layer 1），`warehouse-layer4` project 依赖 `warehouse-midscene`（Layer 3）。运行 `pnpm test:wh:harness` 会按顺序执行 Layer 1 → Layer 3 → Layer 4。

---

### 4.3 `tests/dashboard-midscene.spec.ts` — AI 视觉测试

| Suite | 用例范围 | 核心验证点 |
|-------|---------|----------|
| **Suite A** · 页面布局与加载 | TC-AI-F-001 ~ TC-AI-F-002 | 四大模块 AI 视觉识别、loading 状态等待 |
| **Suite B** · Utilization 利用率看板 | TC-AI-F-003 ~ TC-AI-F-007 | 容量指标可见性、维度切换视觉反馈 |
| **Suite C** · Alerts 预警模块 | TC-AI-F-008 ~ TC-AI-F-012 | 三分区结构、预警条目格式 |
| **Suite D** · Capacity Detail 容量详情 | TC-AI-F-013 ~ TC-AI-F-016 | 温区卡片、搜索、排序、状态标签 |
| **Suite E** · Floor Visualization 楼层可视化 | TC-AI-F-017 ~ TC-AI-F-021 | 楼层 Tab 切换、信息栏、缩放控制 |
| **Suite F** · 维度切换联动验证 | TC-AI-F-022 ~ TC-AI-F-023 | 切换后多模块单位同步 |
| **Suite G** · 视觉质量与一致性 | TC-AI-VIS-001 ~ TC-AI-VIS-005 | 预警颜色编码、CAD 画布渲染 |
| **Suite H** · 数据提取与精确断言 | TC-AI-D-001 ~ TC-AI-D-002 | 全量指标快照、跨模块数据一致性 |
| **Suite I** · Header 门店切换 | TC-AI-HDR-001 ~ TC-AI-HDR-004 | 选择器展开、搜索过滤、门店切换数据刷新 |
| **Suite J** · 空数据与降级展示 | TC-AI-EMPTY-001 ~ TC-AI-EMPTY-003 | 无预警空态、无 CAD 数据空态 |
| **Suite K** · 多楼层切换联动 | TC-AI-FL-001 ~ TC-AI-FL-003 | Floor 2 切换、维度切换后单位同步、楼层切换不影响全局数据 |

---

### 4.4 `tests/seed.spec.ts` — 全局环境初始化

| 步骤 | 职责 |
|------|------|
| Step 1 | 确认开发服务器（8418 端口，真实 SSO 模式）可访问（失败则提前报错） |
| Step 2 | 验证 SSO Cookie 已注入（`storageState` 生效，页面可正常访问） |
| Step 3 | Dashboard 页面可正常加载（页面骨架可访） |
| Step 4 | home 页面可用性探针（优雅降级，非必须） |
| Step 5 | Dashboard React 应用成功挂载（`#app` 非空） |

---

## 五、失败诊断手册

### 5.1 常见失败类型与处理

| 失败现象 | 诊断方法 | 修复方向 |
|---------|---------|---------|
| 接口返回 `{"status":401,"data":{"message":"auth failed"}}` | 检查 `tests/.auth/user.json` 是否存在、是否在 8 小时有效期内 | 执行 `pnpm test:seed` 重新完成 SSO 登录 |
| 接口返回 404 / 数据为 `--` | 检查泳道后端服务是否正常，确认 `poiId` 在泳道中存在 | 联系后端确认 `baihaoxiang-eutnc` 泳道服务部署状态 |
| `expect(page.getByText('75%')).toBeVisible()` 失败 | 检查 `utilizationData?.utilRate` | `utilization-module.tsx` 数值渲染逻辑 |
| Service 状态 `TC-SVC-xxx` 失败 | `page.evaluate(() => window.__dashboardService)` 返回 `undefined` | 检查 `index.tsx` 的 `useEffect` 是否调用了 `dashboardService.mountToWindow()`，依赖数组是否为 `[]` |
| `aiAssert` 失败 | 查看 `midscene_run/report/` 对应截图 | 根据截图定位 UI 问题，修改对应 module tsx 文件 |
| `aiWaitFor` 超时 | 检查网络请求是否卡住 | 确认 dev server 运行正常，SSO Cookie 有效，检查泳道接口是否返回 |
| Midscene `TypeError: Cannot read properties of undefined` | 检查 `.env` 文件配置 | 确认 `MIDSCENE_MODEL_API_KEY` 有效 |
| `storageState` 相关认证失败 | 删除 `tests/.auth/user.json` 后重跑 `pnpm test:seed` | 重新完成美团 SSO 登录 |
| CAD 可视化相关 Midscene 用例不稳定 | 检查 `cadFloorMapData` 状态 | 测试环境是否有 CAD 数据，无数据时 `aiQuery<boolean>` 判断走空态分支 |

### 5.2 接口响应与数据健全性验证

> **原则**：接口数据是 Dashboard 所有功能的基础。在进行任何 UI 断言之前，**必须先确认接口调用是否正确响应、是否返回了有效数据**。不得因为接口无数据就直接认定测试通过或跳过。

#### 5.2.1 接口响应状态快速核查

运行测试前或测试失败后，通过 `page.evaluate()` 核查关键 Service 字段的接口返回情况：

```javascript
// 在 page.evaluate() 或 Chrome DevTools 中执行
const apiHealth = await page.evaluate(() => ({
  utilizationData:    window.__dashboardService?.utilizationData,   // null = 接口未响应或返回空
  alertsData:         window.__dashboardService?.alertsData,        // null = 接口未响应
  capacityDetailData: window.__dashboardService?.capacityDetailData,// null = 接口未响应
  floorInfoData:      window.__dashboardService?.floorInfoData,     // null = 接口未响应
  overviewData:       window.__dashboardService?.overviewData,      // null = overview 接口未响应
}));
// 若某字段为 null → 对应接口未返回有效数据
```

**接口健康度判断标准**：

| 字段 | 预期状态 | 为 null 时说明 | 排查方向 |
|------|---------|-------------|---------|
| `utilizationData` | 非 null，含 `utilRate` | `/utilization` 接口失败或返回空 | 检查 SSO Cookie、泳道服务、poiId 是否有效 |
| `alertsData` | 非 null，含 list 字段 | `/alerts` 接口失败或返回空 | 同上 |
| `capacityDetailData` | 非 null | `/detail` 接口失败 | 同上 |
| `floorInfoData` | 非 null | `/floorInfo` 接口失败 | 同上 |
| `overviewData` | 非 null，含 `poiName` | `/overview` 接口失败 | 同上 |

#### 5.2.2 接口无数据时的用例处理规则

> ⚠️ **核心规则**：接口无数据 **≠ 用例可以通过或 skip**。需区分"空数据降级"与"接口故障"。

| 情形 | 判定 | 处理方式 |
|------|------|---------|
| **接口故障**（401/500/网络超时/SSO 过期） | 环境问题 | 修复环境后重跑，**不记为用例失败** |
| **泳道真实空数据**（该 poiId 下本就无数据） | 合理空态 | 用例必须验证空态 UI 不崩溃（显示 `--` 或 `No alerts`），属于**合理通过** |
| **Service 字段为 null 但 DOM 仍断言数据展示** | 代码缺陷 | 修复渲染逻辑，使其正确降级为占位符，**不允许 skip** |
| **接口有数据但页面展示错误** | 功能 Bug | 修复代码直到断言通过，**严禁降低断言精度** |

#### 5.2.3 接口数据相关用例的连续失败处理

对于与接口数据直接相关的用例（TC-F-003、TC-F-008、TC-SVC-006 等），当**同一用例连续 3 轮失败**且根因不明时：

```
Round 1 失败 → 诊断接口状态 → 尝试修复代码
Round 2 失败 → 重新读 PRD + 技术文档 → 调整修复策略
Round 3 失败 → 深度诊断（读 Service 实现 + 接口返回原始数据）→ 第三次不同策略修复

Round 3 仍失败且为接口环境问题（非代码问题）→ 在 harness-progress.json 的 failedCases 中
  补充字段：
    "failureType": "env_dependency",
    "apiEndpoint": "/shop/wms/api/warehouse/capacity/utilization",
    "lastApiResponse": "<记录最后一次接口返回值>",
    "blockedReason": "泳道 baihaoxiang-eutnc 该 poiId 无容量数据",
    "nextAction": "联系后端确认泳道数据，环境恢复后重跑"

  同时在 Playwright 报告注解中记录：test.info().annotations.push(...)
  → 绝不允许因为接口无数据而 skip 或删除断言
```

### 5.3 调试利器

**Playwright UI 模式**（本地调试首选）：
```bash
pnpm test:ui
# 打开浏览器 UI，可查看每步截图、时间轴、网络请求
```

**evaluate_script 实时状态检查**（Chrome DevTools MCP / Playwright evaluate）：
```javascript
// 在浏览器控制台或 page.evaluate() 中运行
window.__dashboardService?.currentDimension        // '1' | '2'
window.__dashboardService?.currentPoiId            // 'default' | 'poi-xxx'
window.__dashboardService?.utilizationData         // 利用率数据对象
window.__dashboardService?.alertsData?.zoneAlertList  // 温区预警列表
window.__dashboardService?.cadFloorMapData         // CAD 解析结果
```

**Midscene 可视化报告**：
```bash
# 测试完成后，用浏览器打开
open midscene_run/report/
```

---

## 六、代码修改规范

修改 `src/pages/dashboard/` 目录下的文件时，需遵守以下约束：

### 6.1 架构规范

| 规范 | 说明 |
|------|------|
| 状态管理 | 统一使用 `@osgfe/rs-react`，禁止引入 Redux/Zustand/MobX |
| UI 组件 | 优先 `@sailor/keeta-design-pc`；业务基础设施组件优先 `@osgfe/keemart-components` |
| 请求工具 | 使用 `@osgfe/tools-ajax` 的 `requestGet`/`requestPost` |
| 国际化 | 面向用户的文案通过 `@sailor/i18n-web`，不硬编码字符串 |
| 样式 | 优先 TailwindCSS utility class，CSS Modules/SCSS 仅做补充 |
| 文件位置 | 页面私有实现留在 `src/pages/dashboard/` 内，跨页面复用后才提升到全局 |

### 6.2 每次代码修改后强制执行的门禁

```bash
# 修改代码后，必须全部通过才算完成
pnpm run lint          # ESLint 检查
pnpm run type-check    # TypeScript 类型检查

# 可选（格式化）
pnpm run format
```

### 6.3 文件修改影响范围

| 修改的文件 | 可能影响的测试 | 重点验证 |
|----------|-------------|---------|
| `dashboard.service.ts` | Suite H (TC-SVC) + Suite F | Service 状态断言、维度切换 API 调用 |
| `utilization-module.tsx` | Suite B (TC-F-003~009) + Suite B AI | 利用率数值、单位切换、Unactivated 标签 |
| `alerts-module.tsx` | Suite C (TC-F-010~016) + Suite C AI | 三分区结构、预警颜色、daysToFull |
| `capacity-detail-module.tsx` | Suite D (TC-F-017~022) + Suite D AI | 搜索、排序、状态标签颜色 |
| `floor-visualization-module.tsx` | Suite E (TC-E) + Suite E AI | 楼层切换、缩放、信息栏数据 |
| `header-module.tsx` | Suite I (TC-HDR) + Suite I AI | 门店名显示、poiSearch 触发 |

---

## 七、Agent 执行规范

当由 AI Agent（CatPaw）自动执行 Harness 循环时，需遵守以下规范：

### 7.1 上下文管理

- **全局记忆**：PRD、技术方案、设计稿链接、测试页面地址始终保持在全局记忆中（已通过 `update_memory` 写入）
- **压缩次数限制**：单个主 Agent 会话 Context Compaction **不超过 3 次**（见第 0.4 节详细规定）
- **第 3 次压缩后**：完成当前最小可完成任务 → 更新进度文件 → WIP commit → 输出恢复 Prompt → 等待用户开启新 Agent
- **SubAgent 拆分时机**：Layer 1 出现跨模块失败时（而非上下文耗尽时），按模块拆分 SubAgent 并行处理
- **子任务粒度**：按功能模块拆分（Utilization / Alerts / Capacity Detail / Floor / Header），各自独立修复
- **SubAgent 完成后**：必须汇报「修改了哪些文件 + lint/type-check 是否通过」，主 Agent 汇总后再全量运行测试

### 7.2 浏览器操作规范

- **Layer 1/2 默认无头模式（headless: true）**：CI / Agent 自动化执行时无需显示器，接口调用真实泳道数据；本地调试时可用 `HEADLESS=false` 或 `pnpm test:e2e:headed` 切换为可见模式
- **Layer 3 Midscene 固定 `headless: false`**：AI 视觉需要有界面渲染和 Canvas 支持，**不可修改**
- **不干扰用户窗口**：测试使用独立的浏览器上下文（`playwright.config.ts` 管理），不影响用户正在使用的 Chrome 窗口
- **Midscene 操作**：优先 `aiQuery/aiAssert`，避免频繁 `aiAct` 导致的 AI token 消耗
- **直接状态验证**：涉及 RSJS 逻辑验证时，优先 `page.evaluate(() => window.__dashboardService?.xxx)` 而非视觉断言

### 7.3 修复策略优先级

```
1. 先检查 Service 状态（page.evaluate）
   → 确认 API 数据是否正确返回、RSJS 状态是否更新

2. 再检查 DOM 渲染（page.locator / page.getByText）
   → 确认数据是否正确绑定到 UI

3. 最后做 AI 视觉回归（aiAssert）
   → 验证视觉呈现是否符合设计稿
```

### 7.4 验证完成标准（强制红线）

> ⛔ **以下两条为硬性约束，任何情况下不得违反：**
>
> **准则一：禁止虚假通过**
> 不得通过删除断言、注释掉 `expect`、降低断言精度（如把精确值改为宽泛条件）、Mock 绕过等手段让测试"变绿"。每个用例必须因为**功能代码本身正确**而通过，而不是因为测试被削弱。
>
> **准则二：禁止 Skip，验证覆盖率必须 100%**
> 所有 93 个用例（离线 53 个 + AI 视觉 40 个）必须全部**实际执行并通过**，不允许使用 `test.skip()`、`test.fixme()`、条件跳过（`if (condition) return`）等任何跳过手段。Playwright 报告中 `skipped` 数量必须为 **0**，`passed` 必须等于总用例数。

**失败处理边界**（在上述准则范围内）：

- 连续 3 轮 Harness 循环后同一用例仍失败 → **修改代码**使功能正确，直到用例通过；不允许跳过该用例
- 遇到环境问题（dev server 挂了、API key 失效）→ 停止循环，输出诊断信息请求人工介入，待环境恢复后继续
- 遇到设计稿与 PRD 描述冲突 → 以 PRD 为准，修改代码后重新验证
- 遇到数据依赖型用例（如 TC-AI-HDR-004 需要多个门店）→ **修改代码使其满足测试条件**，不允许 skip

---

## 八、测试用例完整清单

### 8.1 离线精确测试（`dashboard.spec.ts`）— 53 个

| ID | 用例名 | Suite |
|----|--------|-------|
| TC-F-001 | 页面标题和四大模块均可见 | A |
| TC-F-002 | 概览信息展示门店名/楼层数/库区数/预警数 | A |
| TC-F-003 | PCS维度：利用率/总量/已用/可用/待激活均正确展示 | B |
| TC-F-004 | 7D Trend 图表区域渲染 | B |
| TC-F-005 | 切换至 Volume 维度后利用率和单位更新 | B |
| TC-F-006 | PCS Tab 默认高亮，Volume Tab 可点击 | B |
| TC-E-001 | 利用率为 0% 时环形图正确渲染 | B |
| TC-E-002 | 利用率 100% 爆仓时正确展示 | B |
| TC-F-007 | Alerts 模块：Temp zones/Floors/Storages 三区均可见 | C |
| TC-F-008 | 温区预警展示利用率百分比和满载天数标签 | C |
| TC-F-009 | 楼层预警：Floor 1 和 Floor 2 均展示利用率 | C |
| TC-F-010 | Storages 预警徽标显示正确数量 | C |
| TC-F-011 | Storages 预警列表展示库区编码和利用率 | C |
| TC-E-003 | 无楼层预警时显示 "No alerts" | C |
| TC-E-004 | 无库区预警时 Storages 区显示 "No alerts" | C |
| TC-F-012 | Temp Zones 显示4个温区卡片和利用率进度条 | D |
| TC-F-013 | 温区利用率 ≥90% 进度条为红色 | D |
| TC-F-014 | Storage 搜索框过滤库区结果 | D |
| TC-F-015 | 点击排序按钮可切换升降序 | D |
| TC-F-016 | FZ-01 (100%) 显示 "Full"，RT-02 (95%) 显示 "Near Full" | D |
| TC-E-005 | 搜索不存在的库区编码时列表为空 | D |
| TC-E-006 | 清空搜索框后恢复显示全部库区 | D |
| TC-F-017 | 楼层切换 Tab 显示 Floor 1 和 Floor 2 | E |
| TC-F-018 | 默认选中 Floor 1 且信息栏显示 Floor 1 数据 | E |
| TC-F-019 | 点击 Floor 2 后信息栏切换为 Floor 2 数据 | E |
| TC-F-020 | 信息栏展示 Capacity 和 Utilization 进度条 | E |
| TC-F-021 | 缩放控制器默认 100%，点击 + 后变为 110% | E |
| TC-F-022 | 点击 - 按钮后缩放比率减小 | E |
| TC-F-023 | CAD 无数据时展示 "No CAD floor data" 空状态 | E |
| TC-F-024 | 信息区展示 "Racks: N" 货架数量标签 | E |
| TC-E-007 | 多次点击 + 后缩放比率不超过最大值 | E |
| TC-E-008 | 多次点击 - 后缩放比率不低于最小值 | E |
| TC-F-025 | 切换 Volume 后三个接口同时被调用 | F |
| TC-F-026 | Volume 维度下待激活仓容显示 "--" | F |
| TC-F-027 | 切换至 Volume 后 Volume Tab 高亮 | F |
| TC-ERR-001 | 利用率接口 500 时页面不崩溃 | G |
| TC-ERR-002 | 利用率接口返回空数据时展示默认占位符 | G |
| TC-ERR-003 | 预警接口返回空时 Alerts 模块不崩溃 | G |
| TC-ST-001 | 温区利用率 80% 恰好为橙色预警 | G |
| TC-ST-002 | 温区利用率 89% 未达红色阈值 | G |
| TC-ST-003 | Storage 卡片利用率 100% 环形图为红色 | G |
| TC-ST-004 | 容量数据缺失时展示 "--" 占位符 | G |
| TC-SVC-001 | window.__dashboardService 在开发环境可访问 | H |
| TC-SVC-002 | 初始维度为 PCS（currentDimension = "1"） | H |
| TC-SVC-003 | 点击 Volume 后 currentDimension 变为 "2" | H |
| TC-SVC-004 | 切回 PCS 后 currentDimension 恢复为 "1" | H |
| TC-SVC-005 | currentPoiId 初始值为 "default" | H |
| TC-SVC-006 | refreshAll() 完成后 utilizationData 已填充 | H |
| TC-HDR-001 | 初始加载后 Header 显示 overviewData.poiName | I |
| TC-HDR-002 | 打开门店选择器触发 poiSearch API | I |
| TC-HDR-003 | poiSearch 接口数据正确渲染门店列表 | I |
| TC-HDR-004 | 门店切换触发全量数据刷新（refreshAll 联动） | I |
| TC-HDR-005 | poiSearch 返回空列表时页面不崩溃 | I |

### 8.2 AI 视觉测试（`dashboard-midscene.spec.ts`）— 40 个

| ID | 用例名 | Suite |
|----|--------|-------|
| TC-AI-F-001 | Dashboard 页面包含标题和四大核心模块 | A |
| TC-AI-F-002 | 页面数据加载完成无转圈 spinner | A |
| TC-AI-F-003 | Utilization 模块显示 Total/Used/Available 三个容量指标 | B |
| TC-AI-F-004 | PCS 维度下 Unactivated 标签显示具体数值 | B |
| TC-AI-F-005 | 7D Trend 趋势图区域可见 | B |
| TC-AI-F-006 | 点击 Volume 按钮后维度切换，Unactivated 变为 "--" | B |
| TC-AI-F-007 | 若有满载天数数据则显示 "Full in Xd" 预警标签 | B |
| TC-AI-F-008 | Alerts 模块包含 Temp zones/Floors/Storages 三个子区域 | C |
| TC-AI-F-009 | 温区预警展示各温区图标和利用率百分比 | C |
| TC-AI-F-010 | Floors 预警区域显示楼层编号和利用率 | C |
| TC-AI-F-011 | Storages 标题旁显示预警计数徽标 | C |
| TC-AI-F-012 | Storages 预警列表展示库区编码和对应利用率 | C |
| TC-AI-F-013 | Temp Zones 区域展示多个温区卡片含利用率进度条 | D |
| TC-AI-F-014 | Storage 搜索框可输入文字过滤库区卡片 | D |
| TC-AI-F-015 | 点击排序按钮切换升降序，库区卡片顺序变化 | D |
| TC-AI-F-016 | 高利用率库区卡片显示 "Full" 或 "Near Full" 状态标签 | D |
| TC-AI-F-017 | 楼层切换 Tab 显示可用楼层列表，默认选中第一层 | E |
| TC-AI-F-018 | 楼层信息栏显示 Capacity 数值和 Utilization 进度条 | E |
| TC-AI-F-019 | 点击 Floor 2 后信息栏切换为 Floor 2 数据 | E |
| TC-AI-F-020 | 缩放控制器显示百分比，点击 + 后数值增大 | E |
| TC-AI-F-021 | 楼层可视化区域显示 "Racks: N" 货架数量 | E |
| TC-AI-F-022 | 切换至 Volume 后 Utilization/温区/楼层数据同步更新 | F |
| TC-AI-F-023 | 从 Volume 切回 PCS 后单位恢复为 pcs | F |
| TC-AI-VIS-001 | 预警颜色编码：高利用率区域颜色更深/更红 | G |
| TC-AI-VIS-002 | Storage 卡片环形图颜色根据利用率分级 | G |
| TC-AI-VIS-003 | Alerts 模块右上角有橙色背景装饰图案 | G |
| TC-AI-VIS-004 | 页面整体布局完整无截断或错位 | G |
| TC-AI-VIS-005 | 楼层可视化 Canvas 区域显示 CAD 图或空状态提示 | G |
| TC-AI-D-001 | 提取 Dashboard 全量核心指标快照并验证数据完整性 | H |
| TC-AI-D-002 | 容量详情中高利用率温区与预警模块数据一致 | H |
| TC-AI-HDR-001 | 页面顶部显示当前门店名称和切换器入口 | I |
| TC-AI-HDR-002 | 点击门店选择器展开下拉，显示门店列表 | I |
| TC-AI-HDR-003 | 门店选择器支持搜索过滤 | I |
| TC-AI-HDR-004 | 选择不同门店后 Dashboard 数据重新加载 | I |
| TC-AI-EMPTY-001 | Alerts 无预警时各区域显示 "No alerts" | J |
| TC-AI-EMPTY-002 | 楼层可视化无 CAD 数据时显示空状态提示 | J |
| TC-AI-EMPTY-003 | Storages 区域无预警时徽标数字为 0 或隐藏 | J |
| TC-AI-FL-001 | Floor 2 切换后信息栏和可视化区域同步更新 | K |
| TC-AI-FL-002 | Volume 维度切换后楼层信息栏单位变为体积单位 | K |
| TC-AI-FL-003 | 切换楼层后 Utilization 和 Alerts 模块数据不变 | K |

---

## 九、Agent 中断恢复规范

当前 Agent 会话因上下文耗尽、超时、网络中断或用户关闭窗口而停止时，新建的 Chat 窗口可以按以下规范**无缝接续**，无需人工梳理进度。

### 9.1 进度持久化机制

Harness 循环依赖以下两类持久化信息，**存储在文件系统**中，不随 Agent 会话消失：

| 信息 | 持久化位置 | 说明 |
|------|-----------|------|
| 执行指南与规范 | `docs/harness-e2e-guide.md` | 本文档，包含全量流程、命令、用例清单 |
| 当前轮次进度 | `docs/harness-progress.json` | 每轮 Harness 循环结束后由 Agent 写入，记录已通过/失败/跳过的用例 |
| 测试报告 | `playwright-report/` | Playwright HTML 报告，含每个用例的最终状态 |
| AI 视觉截图报告 | `midscene_run/report/` | Midscene 报告，含每步截图和 AI 判断依据 |
| 代码变更记录 | `git log --oneline` | 每轮修复后 commit，记录改动范围 |
| 认证状态 | `tests/.auth/user.json` | seed 产出，无需重跑（除非过期） |

### 9.2 进度快照文件格式

Agent 在**每轮 Harness 循环结束**后，必须将进度写入 `docs/harness-progress.json`：

```json
{
  "lastUpdated": "2026-06-01T12:34:56+08:00",
  "currentRound": 3,
  "harnessCycle": "layer1",
  "contextSwitchCount": 1,
  "summary": {
    "layer1Total": 53,
    "layer1Passed": 47,
    "layer1Failed": 6,
    "layer3Total": 40,
    "layer3Passed": 0,
    "layer3Failed": 0,
    "layer3Skipped": 40
  },
  "failedCases": [
    {
      "id": "TC-SVC-003",
      "suite": "H",
      "file": "tests/dashboard.spec.ts",
      "error": "Expected '2' but received '1'",
      "rootCause": "switchDimension() 未更新 currentDimension",
      "fixedInFile": "src/pages/dashboard/dashboard.service.ts",
      "status": "in_progress"
    }
  ],
  "knownSkipped": [
    "TC-AI-HDR-004（单门店环境，storeCount <= 1，已条件保护）"
  ],
  "nextAction": "修复 TC-SVC-003 后重跑 pnpm test:e2e:offline"
}
```

**字段说明**：

| 字段 | 说明 |
|------|------|
| `currentRound` | 当前是第几轮 Harness 循环 |
| `harnessCycle` | 当前所在阶段：`layer1` / `layer2` / `layer3` / `complete` |
| `contextSwitchCount` | 历史上主 Agent 切换次数（累计），新 Agent 接续时可了解历史 |
| `failedCases[].status` | `pending`（待修复）/ `in_progress`（修复中）/ `fixed`（已修复待验证）|
| `failedCases[].roundFailed` | 连续失败轮次，达到 3 时触发深度修复模式 |
| `nextAction` | 下一步具体命令，越具体越好（如 `pnpm test:e2e:offline -- --grep TC-SVC-003`） |

### 9.3 新 Chat 窗口恢复流程

当需要在新窗口继续 Harness 执行时，使用以下**标准启动 Prompt**：

---

**标准恢复 Prompt（复制粘贴到新 Chat）**：

```
继续执行 osg-fe-store-web 项目的 Harness E2E 循环。

执行指南：docs/harness-e2e-guide.md
当前进度：docs/harness-progress.json（如不存在则从头开始）

关键资源（全局记忆）：
- PRD：https://km.sankuai.com/collabpage/2749938366
- 技术文档：https://km.sankuai.com/collabpage/2761200038
- 测试页面：http://localhost:8418/store-alias/dashboard.html（所有层次统一走 8418，需要 SSO Cookie）
- 源码目录：src/pages/dashboard

执行规范：
1. 先读取 docs/harness-progress.json 了解上轮进度和 nextAction
2. 如进度文件不存在，从 Layer 1 开始（pnpm test:e2e:offline）
3. 根据 nextAction 继续执行，不重复已通过的轮次
4. 每轮结束后更新 docs/harness-progress.json
5. 浏览器规范：Layer 1/2 默认 headless: true（可用 HEADLESS=false 切换），Layer 3 固定 headless: false，均使用 Chrome Beta，SSO Cookie 必须有效
6. 主 Agent 上下文压缩达到 3 次时，执行 WIP commit + 更新进度文件后主动切换新 Agent
7. 跨模块失败时按模块拆分 SubAgent 并行修复，SubAgent 完成后向主 Agent 汇报修改文件 + 门禁结果
8. 所有用例通过（layer3 complete）后在进度文件标记 harnessCycle: "complete"
```

---

### 9.4 Agent 恢复后的首轮操作

新 Agent 接收到恢复 Prompt 后，**必须按顺序执行以下步骤**：

```
Step R1: 读取进度文件
  → 文件存在：读取 harnessCycle / nextAction / failedCases
  → 文件不存在：视为首次运行，从 Layer 1 开始

Step R2: 检查环境
  → curl http://localhost:8418/store-alias/dashboard.html
  → 不可访问：提示用户执行 pnpm run dev 后等待

Step R3: 检查已通过轮次
  → layer1Passed = 53 且无 failed：跳过 Layer 1，直接从 nextAction 继续
  → 有 failedCases[].status = "in_progress"：接续上次修复工作

Step R4: 执行 nextAction 中的命令
  → 严格按进度文件的 nextAction 字段执行，不重复已完成工作

Step R5: 每轮完成后更新进度文件
  → 用实际测试结果覆盖 docs/harness-progress.json
```

### 9.5 上下文压缩次数达到上限时的主动切换规范

**触发条件**：主 Agent 当前会话已发生 **3 次 Context Compaction**，下一次压缩即将来临。

> ⚠️ **关键区别**：切换触发条件是「压缩次数达到 3 次」，而非「上下文使用率超过 80%」。压缩次数是可观察的离散事件，更准确可靠。

**触发后必须按顺序执行**：

```
1. 完成当前最小可完成任务（不允许在文件修改中途切换，必须达到可编译状态）
   → 如正在修复某个用例，至少完成 lint + type-check 通过

2. 将当前修改 commit（即使用例还未全绿，标注 WIP）：
   git add src/pages/dashboard/
   git commit -m "wip(harness): round-N 进行中，已修复 TC-XXX，上下文压缩 3 次主动切换"

3. 更新 docs/harness-progress.json，填写：
   - 当前所有用例的最新状态
   - nextAction：下一步精确命令（越具体越好，如 "pnpm test:e2e:offline -- --grep TC-SVC-003"）
   - failedCases 中未完成项的 status 改为 "pending"
   - contextSwitchCount：本次是第几次主动切换（累计计数）

4. 在 Chat 中输出标准恢复 Prompt（见 9.3），提示用户：
   「当前主 Agent 已完成 3 次上下文压缩，请开启新的主 Agent + SubAgent 继续执行。」
```

**新 Agent 接续时的首要动作**：
- 读取 `docs/harness-progress.json`，恢复进度
- 执行 `git log --oneline -5` 查看上次 WIP commit 内容
- 执行 `pnpm run lint && pnpm run type-check` 确认代码状态可用

### 9.6 进度文件的维护时机

| 时机 | 操作 |
|------|------|
| Layer 1 全轮测试完成 | 更新 `summary.layer1*`，设置 `harnessCycle: "layer2"` |
| Layer 2 全轮测试完成 | 更新 `summary.layer2*`，设置 `harnessCycle: "layer3"` |
| Layer 3 全轮测试完成 | 更新 `summary.layer3*`，设置 `harnessCycle: "complete"` |
| 修复单个用例后 | 更新 `failedCases` 中对应项的 `status: "fixed"` |
| 上下文切换前 | 必须更新 `nextAction` 为精确的下一步命令 |
| 发现新 Bug | 追加到 `failedCases` 数组 |

---

## 十、已知限制与后续改进

| 限制 | 影响 | 改进方向 |
|------|------|---------|
| Midscene 依赖 Friday API Key | 无 Key 时 AI 视觉测试无法运行 | 在 CI 中通过环境变量注入，本地依赖 `.env` |
| CAD 相关测试依赖真实数据 | 无 CAD 数据时 TC-AI-VIS-005 走空态分支 | 已在用例中做 `aiQuery<boolean>` 条件判断，两路均有断言 |
| 门店切换需要真实门店数据 | TC-AI-HDR-004 在只有一个门店时跳过 | 已做 `storeCount > 1` 条件保护 |
| `midscene` project 依赖 `chromium` | `pnpm test:e2e:ai` 单独执行时也会先跑离线测试 | 已知，这是有意设计；如需跳过，直接用 `pnpm test:midscene` |
| Web MCP 效率低于 evaluate_script | 模型倾向于视觉操作而非状态查询 | 通过 `window.__dashboardService` 全局暴露引导 Agent 优先用 evaluate |

---

## 十一、Agent 可靠执行约束与自动发现错误规范

> **本节定位**：这是让 Agent 能够**无人工干预**地自动发现错误、定位根因并执行修复的操作手册。所有流程均基于可观测的信号（命令输出、文件、Service 状态），不依赖主观判断。

### 11.1 错误自动发现链路

每次运行测试命令后，Agent 必须按以下优先级顺序收集和分析失败信号：

```
测试运行（pnpm test:e2e:offline）
  │
  ├── Step A：解析命令行输出
  │     → 抓取失败用例名（"● Suite X > TC-XXX"）
  │     → 抓取错误类型（TimeoutError / expect.toBeVisible / expect.toBe）
  │     → 抓取期望值 vs 实际值（Expected: "xxx" / Received: "yyy"）
  │
  ├── Step B：读取 Playwright HTML 报告（playwright-report/）
  │     → 查看每个失败 test 的截图（screenshot attachment）
  │     → 查看 trace（如已开启）定位精确失败行
  │
  ├── Step B'：接口响应层诊断（数据驱动类失败必须先执行此步）
  │     → 通过 page.evaluate() 检查所有 Service 数据字段是否已正确填充：
  │       const health = await page.evaluate(() => ({
  │         util:    Boolean(window.__dashboardService?.utilizationData),
  │         alerts:  Boolean(window.__dashboardService?.alertsData),
  │         detail:  Boolean(window.__dashboardService?.capacityDetailData),
  │         floor:   Boolean(window.__dashboardService?.floorInfoData),
  │         overview:Boolean(window.__dashboardService?.overviewData),
  │       }));
  │     → 若某字段为 false/null：
  │       - 先确认是接口故障（401/500）还是真实空数据
  │       - 接口故障 → 修复环境（SSO、泳道服务），不算代码问题
  │       - 真实空数据 → 验证空态 UI 是否正确降级（显示 "--" / "No alerts"）
  │     → 参考 § 5.2 接口响应与数据健全性验证
  │
  ├── Step C：状态层诊断（window.__dashboardService）
  │     → 针对 TC-SVC-* 失败，直接用 page.evaluate() 检查 Service 状态
  │     → 针对数值渲染类失败，验证 Service 中对应数据字段是否存在且值正确
  │     → **接口字段均为 null 时，停止代码修复，先排查接口环境问题（§ 5.2.2）**
  │
  └── Step D：Layer 3 失败时额外查看 Midscene 报告
        → 路径：midscene_run/report/
        → 每步 AI 操作均有截图和 AI 判断依据
        → 优先从截图定位 UI 问题，而非靠直觉猜测
```

**失败信号优先级**（诊断顺序）：

| 优先级 | 信号来源 | 适用场景 |
|--------|---------|---------|
| **0** | **接口响应状态**：`page.evaluate()` 检查所有 Service 数据字段是否非 null | **任何数据相关失败的第一步**，先判断是环境问题还是代码问题 |
| 1 | `page.evaluate(() => window.__dashboardService)` + 字段值 | TC-SVC-* 和所有数据驱动类失败（接口返回正常后再诊断） |
| 2 | Playwright 命令行输出 + Expected/Received | DOM 断言、文本匹配类失败 |
| 3 | playwright-report/ 截图 | 渲染/布局类失败 |
| 4 | midscene_run/report/ 截图 + AI 判断 | Layer 3 视觉断言类失败 |

---

### 11.2 修复决策树（Agent 自动执行）

```
失败用例接收到后，按以下决策树定位根因：

┌─ 【前置检查】接口是否正确响应并返回数据？（所有数据相关用例必须先执行此步）
│   → 执行：page.evaluate(() => ({
│       util:    Boolean(window.__dashboardService?.utilizationData),
│       alerts:  Boolean(window.__dashboardService?.alertsData),
│       detail:  Boolean(window.__dashboardService?.capacityDetailData),
│     }))
│   → 有字段为 false/null：
│       ┌─ 接口 401/500/超时？ → 环境问题，修复 SSO/泳道，不算代码 Bug
│       └─ 泳道真实无数据？   → 验证空态 UI 正确降级（--/No alerts），属正常通过
│   → 所有字段均非 null → 继续向下诊断代码问题
│
│   ⚠️ 禁止：在接口无数据的情况下直接 skip 用例或删除断言
│   ⚠️ 禁止：把接口未返回数据的失败当作代码 Bug 反复修改 TSX
│
├─ window.__dashboardService 为 undefined？
│   YES → 检查 src/pages/dashboard/index.tsx
│          - useEffect 是否调用了 dashboardService.mountToWindow()
│          - useEffect 依赖数组是否为 []
│          → 修复 index.tsx，重跑 TC-SVC-001 验证
│
├─ Service 状态字段存在但值错误？
│   例：currentDimension 应为 '2' 但仍为 '1'
│   YES → 检查 src/pages/dashboard/dashboard.service.ts
│          - 对应方法（如 switchDimension()）是否正确更新了 observable 属性
│          - 是否触发了正确的 API 调用（通过 requestGet/requestPost）
│          → 修复 dashboard.service.ts
│
├─ Service 状态正确，但 DOM 渲染值错误？
│   例：window.__dashboardService.utilizationData.utilRate = 75，
│        但页面显示 "0%"
│   YES → 检查对应 module TSX 文件
│          - 数据绑定是否正确（observer 是否包裹了组件？）
│          - 条件渲染是否有边界情况遗漏
│          → 修复对应 module TSX
│
├─ DOM 可见性断言失败（toBeVisible / toBeAttached）？
│   YES → 检查 CSS 类名条件 / 条件渲染 / z-index / overflow
│          → 修复样式或条件渲染逻辑
│
├─ 颜色/类名断言失败（style threshold）？
│   例：利用率 90% 应有 text-red，但实际是 text-orange
│   YES → 检查 TailwindCSS 条件类名和阈值边界
│          - 颜色阈值定义位置（通常在 module TSX 中的 getColorClass 函数）
│          → 修复阈值条件
│
└─ aiAssert 失败（Layer 3）？
    YES → 先查 midscene_run/report/ 对应截图
          - 截图显示 UI 异常 → 追溯到 Service 状态或 TSX 渲染问题
          - 截图显示 UI 正常 → 可能是 aiAssert 描述歧义，检查断言措辞
          → 根据截图中的实际 UI 问题修复对应模块
          → 不允许仅修改断言措辞来规避失败（违反准则一）
```

---

### 11.3 每轮修复后强制执行的验证门禁

```bash
# 步骤 1：静态检查（修改任何文件后立即执行）
pnpm run lint          # ESLint — 不允许有 error 级别问题
pnpm run type-check    # TypeScript — 不允许有类型错误

# 步骤 2：针对性重跑（门禁通过后执行，节省时间）
# 只重跑失败的 Suite，而非全量
pnpm exec playwright test tests/dashboard.spec.ts --grep "Suite H"

# 步骤 3：全量重跑（套件级修复完成后确认）
pnpm test:e2e:offline

# 步骤 4：Layer 3（Layer 1 全绿后）
pnpm test:e2e:ai
```

> **效率原则**：优先针对性重跑（指定 Suite / grep），避免每次修改都全量运行，节省 Harness 循环时间。
> **正确性原则**：在提交前必须全量运行一次，确保修复没有引入新的失败。

---

### 11.4 Harness 轮次退出条件（精确定义）

| 退出条件 | 检查方式 | 下一步 |
|---------|---------|--------|
| Layer 1 全部 53 个用例通过，`skipped: 0` | Playwright 报告 `passed: 53, failed: 0, skipped: 0` | 进入 Layer 2（SVC 断言） |
| Layer 2 全部 6 个 SVC 用例通过 | Playwright 报告 Suite H 全绿 | 进入 Layer 3（Midscene AI） |
| Layer 3 全部 40 个用例通过，`skipped: 0` | Playwright + Midscene 报告全绿 | 进入 **Layer 4（设计稿 UI 对比）** |
| Layer 4 无 🔴/🟡 级别样式偏差 | 截图与设计稿比对通过，`uiReview` 字段所有模块状态为 `pass` 或 `minor` | 标记 `harnessCycle: "complete"` |
| **同一用例连续 3 轮失败** | `failedCases[x].roundFailed >= 3` | 升级为「深度修复」模式（见下） |
| **环境问题** | dev server 无响应 / API key 失效 | 停止循环，输出诊断，等待人工介入 |

**深度修复模式**（连续 3 轮仍失败时触发）：

```
1. 重新阅读 PRD（https://km.sankuai.com/collabpage/2749938366）中对应功能的描述
2. 重新阅读技术文档（https://km.sankuai.com/collabpage/2761200038）中对应模块的实现规范
3. 阅读 src/pages/dashboard/dashboard.service.ts 中相关方法的完整实现
4. 对比设计稿（imd.sankuai.com）与当前 UI 的差异（参考二节 Layer 4 对比维度表）
5. 基于以上四个来源，重新判断根因，采用不同于前 3 次的修复策略
6. 修复后先通过 lint + type-check，再全量运行 Layer 1
```

---

### 11.5 SubAgent 并行化策略（跨模块失败时）

当 Layer 1 同时出现**多个不同模块**的失败时，主 Agent 可拆分 SubAgent 并行处理：

| SubAgent | 负责模块 | 核心文件 | 触发条件 |
|----------|---------|---------|---------|
| **SubAgent-Service** | RSJS 状态层 | `dashboard.service.ts` | TC-SVC-* 有失败 |
| **SubAgent-Utilization** | 利用率看板 | `utilization-module.tsx` | Suite B (TC-F-003~009) 有失败 |
| **SubAgent-Alerts** | 预警模块 | `alerts-module.tsx` | Suite C (TC-F-010~016) 有失败 |
| **SubAgent-Capacity** | 容量详情 | `capacity-detail-module.tsx` | Suite D (TC-F-017~022) 有失败 |
| **SubAgent-Floor** | 楼层可视化 | `floor-visualization-module.tsx` | Suite E (TC-E-001~008) 有失败 |
| **SubAgent-Header** | 门店切换 | `header-module.tsx` | Suite I (TC-HDR-001~005) 有失败 |

**SubAgent 工作规范**：

```
每个 SubAgent 接收到的任务格式：
  目标：修复 [Suite X] 中的 [TC-xxx, TC-yyy] 用例
  失败错误：[粘贴 Playwright 输出的 Expected/Received]
  相关文件：[module TSX 文件路径]
  参考文档：docs/harness-e2e-guide.md 第 5、6 节

SubAgent 完成后必须向主 Agent 汇报：
  1. 修改了哪些文件（精确到文件名）
  2. 修改的具体内容摘要（一句话）
  3. pnpm run lint 结果（通过/失败）
  4. pnpm run type-check 结果（通过/失败）
  5. 针对性重跑该 Suite 的测试结果（通过/失败）
```

**主 Agent 汇总流程**：
1. 等待所有 SubAgent 完成汇报
2. 若所有 SubAgent 均汇报通过 → 全量运行 `pnpm test:e2e:offline` 确认
3. 若某 SubAgent 汇报失败 → 主 Agent 接管该 SubAgent 的修复工作，不再重新拆分
4. 全量测试通过后更新 `docs/harness-progress.json`

---

### 11.6 禁止行为清单（完整版）

以下行为在任何情况下均**严格禁止**：

| 禁止行为 | 说明 | 典型表现 |
|---------|------|---------|
| 删除或注释 `expect()` 断言 | 违反准则一（禁止虚假通过） | `// expect(x).toBe(y)` |
| 降低断言精度 | 违反准则一 | 把 `toBe('75%')` 改为 `toBeTruthy()` |
| 使用 `test.skip()` / `test.fixme()` | 违反准则二（覆盖率 100%） | 任何形式的跳过 |
| 修改 `config/config.ts` 代理配置 | 可能切断 Layer 3 真实数据 | 更换泳道、删除代理规则 |
| 将 `midscene` project 的 `headless: false` 改为 `true` | 导致 AI 视觉无法正常渲染 | 修改 playwright.config.ts midscene project（Layer 1/2 的 headless 可通过 `HEADLESS=false` 控制，不属于禁止范围）|
| 绕过真实 SSO 登录（如使用 `DISABLE_SSO=true`）运行 Harness 正式测试 | Harness 主流程必须使用真实 Cookie，接口才能返回真实数据 | 设置 `DISABLE_SSO=true`、修改 `global-setup.ts` 跳过登录 |
| 在 `dashboard.spec.ts` 中使用 `page.route()` Mock API | Layer 1 必须调用真实接口，Mock 会掩盖真实问题 | 在 `dashboard.spec.ts` 中添加 `page.route()` 拦截 |
| 为测试另起 8418/8420 以外的端口 | 违反端口约束（0.1 节） | 启动 8419 或其他端口 |
| 在 Layer 3 中使用 `page.route()` Mock API | 违反 SSO/真实数据约束（0.2 节） | Midscene 测试中添加 route 拦截 |
| 修改测试文件以降低测试标准 | 违反 Harness 核心原则 | 修改 dashboard.spec.ts 的 expect 条件 |
| 第 4 次上下文压缩时继续当前 Agent | 违反上下文轮换约束（0.4 节） | 在第 3 次压缩后未切换就继续工作 |
| 接口无数据时直接 skip 或删除断言 | 违反 § 5.2.2 接口数据规则；接口无数据 ≠ 用例通过 | 以"后端没返回数据"为由 skip 或注释断言 |
| 把接口故障（401/500）当作代码 Bug 反复改 TSX | 误判根因，浪费修复轮次 | 接口全部返回 null 时仍不检查 SSO/泳道，直接修改渲染逻辑 |
| 接口环境问题未在进度文件标记就继续跑测试 | 掩盖真实阻塞原因，导致多轮无效循环 | 连续 3 轮同一用例失败未写 `failureType: "env_dependency"` |

---

### 11.7 Harness 结束验收标准

当以下所有条件同时满足时，Harness 循环正式完成：

```
✅ Playwright 报告：passed = 93, failed = 0, skipped = 0
   （Layer 1: 53 个 + Layer 3: 40 个）

✅ docs/harness-progress.json 中 harnessCycle = "complete"

✅ pnpm run lint 无 error 级别问题

✅ pnpm run type-check 无类型错误

✅ git log 中有完整的修复记录（每轮修复均有 commit）

✅ Layer 4 UI 对比通过：harness-progress.json 的 uiReview 字段
   所有模块状态为 "pass" 或 "minor"（无 🔴/🟡 级别偏差）
   各模块截图已保存在 midscene_run/report/ 中供人工复核
```

> **注意**：Layer 2（SVC 状态断言，Suite H）的 6 个用例包含在 Layer 1 的 53 个用例总数中，不单独计入总数。真实总用例数为：Layer 1（含 SVC + Header）53 个 + Layer 3（Midscene）40 个 = **93 个**。

---

### 11.8 harness-progress.json 的 uiReview 字段规范

Layer 4 完成后，必须将结果写入 `docs/harness-progress.json` 的 `uiReview` 字段：

```json
{
  "uiReview": {
    "executedAt": "2026-06-03T10:00:00Z",
    "designUrl": "https://imd.sankuai.com/file/189270363591816",
    "modules": {
      "layout": {
        "status": "pass",
        "issues": []
      },
      "utilization": {
        "status": "pass",
        "issues": ["进度条高度偏差 2px，可接受"]
      },
      "alerts": {
        "status": "minor",
        "issues": ["Temp zones 图标与文字间距差 4px"]
      },
      "capacityDetail": {
        "status": "pass",
        "issues": []
      },
      "floorVisualization": {
        "status": "pass",
        "issues": []
      },
      "header": {
        "status": "pass",
        "issues": []
      }
    },
    "screenshotDir": "midscene_run/report/"
  }
}
```

**状态枚举说明**：
- `"pass"`：与设计稿对齐，无明显偏差（或偏差 ≤ 4px 的轻微差异）
- `"minor"`：存在轻微偏差（间距偏差 ≤ 8px、颜色接近），记录在案但不阻塞 Harness 完成
- `"fail"`：存在严重偏差（明显色差、布局错乱、字号差 2 级以上），**必须修复后才能退出**

---

---

## 十二、仓容线上化 Harness 执行指南

### 12.1 功能概述

**仓容线上化**功能覆盖以下核心业务链路：

```
货架库位规格极限空间利用率配置
  → 仓容预警规则（6 类使用率预警：服务站/温区/库区/货架过载/货架低利用/楼层）
  → 预警颜色联动生成
  → 仓容概览看板五大模块（服务站利用率 / 预警模块 / 温区容量 / 库区容量 / 楼层可视化）
  → 各指标容量和使用率计算
```

E2E 自动化测试由两个 Playwright spec 文件组成，均基于 Dashboard 看板页面（`src/pages/dashboard`）：

- **Layer 3**：`tests/warehouse-capacity-layer3.spec.ts`（Midscene AI 视觉，28 个用例）
- **Layer 4**：`tests/warehouse-capacity-layer4.spec.ts`（设计稿 UI 对比，16 个用例）

> ⚠️ **说明**：货架规格配置页、预警规则配置列表页等管理后台页面（非 Dashboard）的 Layer 1/2 精确测试用例，需要等对应页面入口实现后单独补充 spec 文件。当前 Layer 3/4 测试聚焦于 Dashboard 看板页面，通过预警配置的最终呈现（颜色联动）间接验证预警配置功能。

### 12.2 Playwright Project 配置

| Project 名 | 测试文件 | 依赖 | headless | 说明 |
|-----------|---------|------|----------|------|
| `warehouse-midscene` | `warehouse-capacity-layer3.spec.ts` | `chromium` | false | 仓容 Layer 3 AI 视觉测试 |
| `warehouse-layer4` | `warehouse-capacity-layer4.spec.ts` | `warehouse-midscene` | false | 仓容 Layer 4 UI 对比 |

> 两个 project 均已在 `playwright.config.ts` 中配置，使用 `storageState: 'tests/.auth/user.json'` 复用 SSO Cookie，接对应泳道真实后端数据。

### 12.3 运行命令

```bash
# 仅运行 Layer 3 AI 视觉测试（28 个用例）
pnpm test:wh:layer3

# 仅运行 Layer 4 UI 对比测试（16 个用例）
pnpm test:wh:layer4

# Layer 3 + Layer 4 完整循环（按依赖顺序自动执行）
pnpm test:wh:harness
```

### 12.4 测试用例分布

#### Layer 3（`tests/warehouse-capacity-layer3.spec.ts`，28 个）

| Suite | 核心验证点 |
|-------|----------|
| Suite A · 整体结构 | 五大模块可见性、页面正常加载 |
| Suite B · 容量使用率指标 | PCS/Volume 维度切换、Unactivated 字段、7D Trend、满载天数 |
| Suite C · 预警模块 | Alerts 三分区（温区/楼层/库区）、徽标计数 |
| Suite D · 容量详情 | 温区卡片、库区搜索、Full/Near Full 状态标签、排序 |
| Suite E · 楼层可视化 | Tab 切换、信息栏容量/利用率、缩放控制器、CAD 渲染 |
| Suite F · 维度切换联动 | Volume 切换后多模块数据同步、楼层切换不影响全局 |
| Suite G · 预警颜色联动 | 进度环颜色分级（绿/橙/红）、圆环颜色与预警等级一致 |
| Suite H · 数据完整性 | 预警配置数据 Dashboard 反映、全量指标快照 |

#### Layer 4（`tests/warehouse-capacity-layer4.spec.ts`，16 个）

| Suite | 检查项 | 偏差级别 |
|-------|--------|----------|
| Step1 · 逐模块截图存档 | 整体/Header/Utilization/Alerts/温区卡片/库区卡片/楼层可视化 | — |
| Step2 · 颜色规范精确验证 | 页面背景 #F7F8FA、绿色 #00B080、橙色 #EE7F00、红色 #F0390E | 🔴/🟡 |
| Step3 · 字号规范验证 | 主标题 24px（text-2xl）、模块标题 20px（text-xl） | 🟡 |
| Step4 · 间距与布局 | 双列布局、圆角 12px（rounded-xl）、无文字截断 | 🟡/⚪ |
| Step5 · 预警颜色联动 UI | 温区/库区卡片颜色与预警等级、维度切换后颜色不变 | 🔴/🟡 |
| Step6 · 综合质量评估 | 整体视觉风格、UI 最终快照 | — |

### 12.5 进度记录格式

Layer 3/4 完成后，在 `docs/harness-progress.json` 中维护 `warehouseHarness` 字段：

```json
{
  "warehouseHarness": {
    "lastUpdated": "2026-06-04T12:00:00+08:00",
    "layer3": {
      "specFile": "tests/warehouse-capacity-layer3.spec.ts",
      "project": "warehouse-midscene",
      "total": 28,
      "passed": 0,
      "failed": 0,
      "status": "PENDING"
    },
    "layer4": {
      "specFile": "tests/warehouse-capacity-layer4.spec.ts",
      "project": "warehouse-layer4",
      "total": 16,
      "passed": 0,
      "failed": 0,
      "status": "PENDING",
      "uiReview": {
        "overallStatus": "pending",
        "modules": []
      }
    }
  }
}
```

### 12.6 验收标准

```
✅ pnpm test:wh:layer3 全部 28 个用例通过（warehouse-midscene project）

✅ pnpm test:wh:layer4 全部 16 个用例通过（warehouse-layer4 project），
   无 🔴/🟡 级别 UI 偏差

✅ docs/harness-progress.json 的 warehouseHarness.layer3.status = "PASS"
   且 warehouseHarness.layer4.uiReview.overallStatus = "pass" 或 "minor"

✅ Midscene 截图已保存在 midscene_run/report/ 中供人工复核
```

---

## 十三、兜底泳道环境测试（本地接口异常时的备用方案）

> **触发场景**：本地 `pnpm run dev:sso`（8418 端口）启动正常，但某些接口返回异常（如代理配置不生效、响应与测试环境不一致、数据与预期不符），导致 Layer 1/2/3 频繁误报。此时可跳过本地 dev server，直接将 Playwright 指向真实测试环境 URL 作为兜底验证。

### 13.1 兜底泳道环境地址

```
https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store/dashboard.html
```

- **泳道**：`baihaoxiang-eutnc`（与本地代理目标一致）
- **SSO**：需要美团内网 Cookie（通过 global-setup 完成 SSO 登录后复用）
- **数据**：与本地 8418 代理后端相同，接口路径一致

### 13.2 切换至泳道环境运行测试

所有测试文件的 `DASHBOARD_URL` 均通过 `process.env.BASE_URL` 注入，**无需修改任何测试代码**，只需在命令前覆盖该环境变量即可。

#### Layer 1 精确断言（泳道兜底）

```bash
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:e2e:offline
```

#### Layer 2 Service 状态断言（泳道兜底）

```bash
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:svc
```

#### Layer 3 Midscene AI 视觉（泳道兜底）

```bash
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:e2e:ai
```

#### 完整 Harness 循环（泳道兜底）

```bash
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:harness
```

> ⚠️ **注意**：泳道 URL 末尾不带 `/dashboard.html`，各测试文件会自动拼接 `/dashboard.html`（与本地模式行为一致）。

### 13.3 SSO Cookie 在泳道环境的适配

泳道环境同样需要美团 SSO 登录，**Cookie 存储机制与本地模式相同**（`tests/.auth/user.json`）。

#### 首次使用泳道兜底环境（Cookie 不存在或已过期）

```bash
# Step 1：用泳道 URL 初始化 SSO Cookie
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:seed

# Step 2：确认 Cookie 已生成
ls -la tests/.auth/user.json

# Step 3：运行测试
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:e2e:offline
```

#### Cookie 已有效（本地 8418 登录过，Cookie 未过期）

由于 Cookie 中包含美团 SSO 的 Token，**同一套 Cookie 通常对本地和泳道环境都有效**（Cookie 域名绑定 `.sankuai.com`），可直接复用：

```bash
# 直接用现有 Cookie 跑泳道环境测试（无需重新登录）
BASE_URL=https://baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com/overseas/store \
  pnpm test:e2e:offline
```

> 若遇到 401/302 跳转登录页，说明 Cookie 对泳道域名无效，执行 `pnpm test:seed`（加 `BASE_URL`）重新登录。

### 13.4 泳道环境 vs 本地环境差异说明

| 对比维度 | 本地 8418（`dev:sso`） | 泳道环境（直连） |
|---------|----------------------|---------------|
| 接口路由 | webpack-dev-server 代理到 `baihaoxiang-eutnc-sl-manager.osg.test.sankuai.com` | 直接请求泳道服务 |
| 数据来源 | 相同（同一个泳道后端） | 相同 |
| SSO | `global-setup` 弹出浏览器登录，Cookie 保存至 `tests/.auth/user.json` | 相同机制 |
| 稳定性 | 依赖本地代理配置、`o dev` 启动状态 | **不依赖本地服务**，适合作为兜底验证 |
| 使用时机 | 日常开发调试、Harness 主流程 | 本地接口异常时的备用验证 |
| `webServer` 启动 | `playwright.config.ts` 自动执行 `pnpm dev:sso` | 不需要启动本地 dev server（`reuseExistingServer` 探测失败后不会启动） |

### 13.5 常见问题排查

**Q：泳道 URL 访问 403 / 302 跳转登录页**  
→ Cookie 已过期或对泳道域名无效。执行 `BASE_URL=... pnpm test:seed` 重新完成 SSO 登录。

**Q：`global-setup` 提示找不到服务**  
→ 泳道 URL 不受 `webServer` 管理，`global-setup.ts` 会直接访问 `BASE_URL`，无需本地服务启动。若报错，检查网络是否可以访问内网（需美团 VPN 或内网环境）。

**Q：测试用例断言失败但手动打开泳道页面显示正常**  
→ 检查 `tests/.auth/user.json` 中的 Cookie 是否与当前泳道绑定，必要时重新执行 `pnpm test:seed`（加 `BASE_URL`）。

**Q：泳道测试通过但本地 8418 测试失败**  
→ 说明本地代理配置有问题（`config/config.ts` 或 `o-builder.config.mts`），排查方向：
1. 检查 `config/config.ts` 中代理路径是否完整覆盖所有接口
2. 检查 `pnpm run dev:sso` 启动日志中是否有代理报错
3. 用 Chrome DevTools Network 面板对比本地 vs 泳道的接口请求/响应

---

*本文档由 CatPaw Agent 根据项目实际代码结构、TMS Harness 实践经验（https://km.sankuai.com/collabpage/2764956067）自动生成并持续维护。*
