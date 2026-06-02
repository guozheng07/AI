# 商品建档 Agent（SKU-Agent）技术架构与设计文档

> **适用读者**：前端开发、架构评审、技术汇报
> **最后更新**：2026-06-02

---

## 一、项目定位

SKU-Agent 是 Keemart 芥末系统内嵌的 AI 商品建档助手，运行于 `osg-fe-ai-iframe` 应用中。用户通过自然语言或上传 Excel/图片，由 AI Agent 自动完成商品批量建档的结构化数据填写，最终由商家确认后提交到芥末平台。

---

## 二、整体架构

### 2.1 三层嵌套架构

```
pc-w（osg-fe-jiemo-pc-w，外层主应用壳）
  └── ai-iframe（osg-fe-ai-iframe，AI Agent 容器层）
        └── sku-agent（商品建档 Agent 业务层）
              ├── shared/    跨 Agent 公共能力（ChatArea、TaskPanel、EmbeddedBrowser 等）
              └── sku-agent/ 商品建档专属实现（Dialogue、EditableTable、ResultCard 等）
```

**关键设计原则**：`GlobalEventService`（全局 SSE 连接 `/api/events`）生命周期绑定在 `agent/index.tsx` 这一层，Agent 切换时连接不断开；各 Agent 专属的 `ChatService`、`SessionService`、`TaskService` 通过 `bindServices` 随组件挂载/卸载自动创建/销毁。

### 2.2 技术栈

| 层级 | 技术 |
|------|------|
| UI 框架 | React 18 + TypeScript |
| 组件库 | `@sailor/keeta-design-pc`（Keeta Design PC） |
| 状态管理 | `@osgfe/rs-react`（RSJS，响应式 Service 模式） |
| 实时通信 | SSE（Server-Sent Events），基于 Fetch ReadableStream |
| 工具库 | `@osgfe/tools`（含 lodash-es、hGet/hPost 等） |
| 国际化 | `@sailor/i18n-web`（支持中/英/阿拉伯文） |
| 样式 | TailwindCSS + CSS Module（复杂样式兜底） |
| 构建 | OBuilder（`pnpm run dev:HK / dev:EU / build`） |

---

## 三、UI 布局架构

页面采用**三栏可拖拽布局**，各区域宽度以百分比定义，窗口缩放后比例不变：

```
┌─────────────────────────────────────────────────────────────┐
│  AppLayout（顶部导航占位，100dvh - TOP_NAVBAR_HEIGHT_PX）    │
├──────────────┬──────────────────────────┬────────────────────┤
│  TaskPanel   │       ChatArea           │     Dialogue       │
│  （左侧）    │  （中间，flex:1 自适应）  │  （右侧，可拖拽   │
│  任务列表    │  历史消息 + 流式消息      │  展开/收起）       │
│  会话列表    │  QueuePanel（排队浮层）   │  可编辑表格        │
│  flex:0 0    │  输入框（含上传按钮）     │  flex:0 0          │
│  taskWidthPct│                          │  dialogueWidthPct  │
│  (10%~30%)   │                          │  (20%~50%)         │
└──────────────┴──────────────────────────┴────────────────────┘
```

- **ChatArea 与 EmbeddedBrowser 互斥**：任务"查看芥末链接"时主内容区切换为内嵌浏览器视图，对话区暂时隐藏
- **拖拽实现**：基于 `pointerdown/pointermove/pointerup` 事件，以百分比换算位移，并对各区域设置最小像素约束（TaskPanel ≥ 250px，ChatArea ≥ 400px）

---

## 四、业务功能清单

### 4.1 商品批量建档主流程

| 功能 | 实现组件 | 说明 |
|------|----------|------|
| 自然语言建档 | `ChatArea` 输入框 | 用户输入商品描述，AI 自动识别并结构化 |
| 文件上传建档 | `McS3Upload`（S3 上传封装） | 支持 Excel / 图片，拖拽或点击上传 |
| AI 进度可视化 | `ProgressCard` | 展示各阶段步骤名称、耗时、状态（running/done/error） |
| 折叠进度摘要 | `FoldProcessCard` | 可折叠的执行过程概览，避免冗余信息占用空间 |
| 结果卡片 | `ResultCard` | 多 block 编排容器：折叠进度 + Markdown 说明 + 芥末链接 + 可编辑表格 |
| 批量确认提交 | `Dialogue` + `EditableTable` | 用户在侧栏审核/修改 AI 填写的表格字段后提交 |
| 任务跳转 | `TaskPanel` 菜单 | 支持"查看会话"、"查看待办"、"查看我的申请"直接跳转芥末页面 |

### 4.2 对话会话管理

| 功能 | 说明 |
|------|------|
| 新建/切换会话 | 左侧会话列表，点击切换，自动加载对应历史消息 |
| 历史消息分页 | 切换会话后加载最近 N 条，上拉触底自动加载更早消息 |
| 会话状态 Tag | 标题右侧展示 processing / queued / completed / error 状态 |
| 消息取消 | 对当前正在生成的 AI 消息点击停止，调用 `/api/cancel` |
| 消息撤销 | 从排队队列删除尚未处理的消息（撤销当前流时先 `clearCurrentStream`） |
| 中断并发送 | 打断当前回复，携带 `queueQid` 直接发送新消息 |
| 异地踢出重连 | 同账号其他窗口激活时弹出提示，一键重连恢复所有数据 |

### 4.3 任务管理

| 功能 | 说明 |
|------|------|
| 任务列表 | 左侧 TaskPanel 展示全部建档任务，高亮待处理任务 |
| 任务分页 | 滚动到底部自动加载更多历史任务 |
| 任务操作 | 下拉菜单：查看会话 / 查看待办 / 查看我的申请（跳转芥末系统） |

### 4.4 内嵌浏览器

| 功能 | 说明 |
|------|------|
| 多标签浏览 | 最多 20 个标签页，超出时替换最早标签 |
| 地址栏导航 | 输入 URL 回车跳转，相对路径自动解析 |
| 前进/后退 | 独立维护每个标签页内部历史栈，不污染外层浏览器 history |
| 刷新 | URL 追加时间戳参数强制 iframe 重载 |
| 芥末链接直开 | AI 回复中的 `jiemo_link` block 点击后切换到内嵌浏览器并打开 |
| 关闭回到对话 | 全部标签关闭后自动切回聊天视图 |

### 4.5 可编辑表格（Dialogue 侧栏）

| 功能 | 说明 |
|------|------|
| 行内编辑 | 单元格点击即可编辑，文本/图片两种编辑器 |
| 低置信度高亮 | `cellStatus: lowConfidence` → 黄色背景 |
| 无效值高亮 | `cellStatus: invalid` → 红色背景 + Tooltip 错误提示 |
| 编辑清除状态 | 修改后自动将 cellStatus 置为 `confirmed`，清除高亮 |
| 统计展示 | 展示总行数 / 待确认数 / 待填写数 |
| 导出提交 | 表格数据 POST 生成 Excel → S3 链接 → 通过聊天发给 AI 形成闭环 |
| 侧栏宽度控制 | 拖拽分隔条（20%~50%）；点击摘要卡片触发自动展开 |

---

## 五、核心技术设计

### 5.1 SSE 实时流通信 + 消息块渲染引擎

AI 回复通过 SSE 流式推送，前端消费多种异构消息块并实时渲染。

**事件类型**：

```
message.start    → 初始化消息流，清空 loading block（sessionId 守卫，过期事件直接丢弃）
block            → 新增 block（整体替换 currentMessageStream$ 对象，触发 observer 重渲染）
block.append     → 向已有 block 追加内容（流式文本增量）
block.delta      → Markdown 流式增量更新
message.replace  → Server 端重算结果，整体替换当前消息流
message.done     → 消息结束，loading → 完成态
message.resume   → 断点续传，将历史 blocks 注入 currentMessageStream$
kicked           → 账号被其他设备抢占
```

**Block 渲染器插件化**：通用 `ChatArea` 不内置业务逻辑，各 Agent 通过 `blockRenderers` 属性注入专属卡片渲染函数，key 对应后端 SSE block 的 `type` 字段：

```tsx
const blockRenderers = useMemo<Record<string, BlockRenderer>>(() => ({
  process_card:      (block, key) => <ProgressCard ... />,
  fold_process_card: (block, key) => <FoldProcessCard ... />,
  result_card:       (block, key) => <ResultCard ... />,
  jiemo_link:        (block, key) => <JiemoLink ... />,
  modify_excel_card: (block, key) => <ModifyExcelCardSummary ... />,  // 摘要卡片
}), [...]);
// ChatArea 内置处理（无需注册）：loading / thinking / markdown / text / error / image
```

> 详细设计见 [`tech/SSE_IMPLEMENTATION.md`](./tech/SSE_IMPLEMENTATION.md)

---

### 5.2 内嵌浏览器 UI Bridge 三层透传通信

**背景**：芥末系统（product-w）运行在 ai-iframe 内嵌的 iframe 中，需要调用 confirm / toast / alert 等 UI 原语，但不能直接访问外层 pc-w 的 UI 组件库。

**消息协议** (`source: 'jiemo-iframe-ui-bridge'`)：

| 消息类型 | 方向 | 是否需要回写 | 说明 |
|----------|------|-------------|------|
| `ui:confirm` | product-w → ai-iframe → pc-w | ✅（`ui:confirm:result`） | 确认/取消弹窗 |
| `ui:alert` | product-w → ai-iframe → pc-w | ✅（`ui:alert:result`） | 单确认弹窗 |
| `ui:toast` | product-w → ai-iframe → pc-w | ❌ | 轻提示，无需回调 |

**通信链路**：

```
product-w（最内层 iframe）
  ──postMessage(ui:confirm)──▶
    ai-iframe（EmbeddedBrowser 中继层）
      ──window.parent.postMessage(原样转发)──▶
        pc-w（最外层，用 Element UI 渲染 Modal）
          ──postMessage(ui:confirm:result, requestId)──▶
        ai-iframe（收到后中继给 product-w）
          ──activeFrame.contentWindow.postMessage──▶
        product-w（resolve 内部 Promise，继续执行）
```

**导航劫持**：product-w 内 `navigateTo`/`linkTo`（`@osgfe/keemart-api`）通过 `xx-message-event` source 向父窗口发消息，ai-iframe 拦截后根据 `options.target` 决定：
- `_blank` / `isNewTab: true` → 新开内嵌浏览器标签页
- `_top` → 跳出内嵌浏览器到顶层系统窗口
- 默认 → 当前标签页导航

> 详细实现见 [`shared/components/embedded-browser/iframe-navigation-message.ts`](../shared/components/embedded-browser/iframe-navigation-message.ts)

---

### 5.3 跨会话隔离三层防御体系

**背景**：用户快速切换会话时，旧会话 SSE 消息可能延迟到达并污染新会话的 UI。

| 防御层 | 触发时机 | 机制 |
|--------|----------|------|
| **第 1 层：主动中断** | `clearSession()` | 调用 `_currentFetchAbortController.abort()` 立即切断旧 SSE fetch 连接 |
| **第 2 层：事件 Guard** | `message.start` 收到时 | 校验 `sessionId` 是否与 `currentSessionId$` 匹配，不匹配则丢弃整条消息流 |
| **第 3 层：状态清零** | `clearSession()` | 重置 10+ 项跨会话敏感状态（消息流、排队链、重连计数、终结事件标志等） |

QueuePanel 的「中断并发送」/「撤销」操作还在渲染时**快照 sessionId** 到闭包中，防止切换会话竞态时操作作用于错误会话。

> 详细设计见 [`tech/SSE_IMPLEMENTATION.md`](./tech/SSE_IMPLEMENTATION.md)「八点五、会话隔离三层防御体系」

---

### 5.4 断点续传机制（message.resume）

当用户刷新页面或网络中断后，AI 已生成到一半的回复可以无缝续接。

**流程**：

```
1. fetchChatHistoryWithResume(sessionId)
   └── 后端返回历史消息 + 未完成消息携带的 resumeToken

2. ChatService 检测到 resumeToken
   └── 调用 resumeMessageFromQueue(sessionId, resumeQid, userParts)

3. SSE 推送 message.resume 事件
   └── 将历史 blocks 注入 currentMessageStream$，重建渲染状态

4. 后续 block 事件追加到已恢复的流中
   └── 用户看到无缝续接的消息内容
```

---

### 5.5 可编辑表格双向数据链路

完整闭环：AI 推送表格数据 → 前端展示 → 用户编辑 → 提交 → AI 继续建档。

```
SSE block (modify_excel_card)
  → blockRenderer 渲染摘要卡片（ModifyExcelCardSummary）
  → 点击"展开"触发 pushResultToDialogue()
  → setDialogueTableData() + setDialogueOpenSignal++ → Dialogue 侧栏展开
  → 用户在 EditableTable 编辑字段
  → 点击提交 → POST /api/table/export（后端生成 Excel，返回 S3 URL）
  → chatService.sendMessage(submitPrompt, sessionId, [{ url: s3Url }])
  → ChatArea 展示用户消息（含 Excel 附件）
  → AI 收到 Excel 后继续建档流程（闭环）
```

**切换会话时自动清理**：`currentSessionId$` 变化时，页面自动触发 `dialogueCloseSignal++`，收起侧栏并清空 `dialogueTableData`/`dialogueStatistics`，防止旧会话数据残留。

---

### 5.6 RSJS 状态管理架构

项目统一使用 `@osgfe/rs-react`（响应式 Service 模式），以 `$` 结尾的属性为可观测属性，`observer` 包裹的组件自动追踪依赖并精确重渲染。

**Service 职责划分**：

| Service | 注册层级 | 职责 |
|---------|----------|------|
| `GlobalEventService` | `agent/index.tsx`（顶层，跨 Agent 共享） | 全局 SSE 连接（`/api/events`），下发 task/session/queue/kicked/message.add 事件 |
| `ChatService` | `bindServices`（随 Agent 组件） | 消息流管理、SSE 发送/取消/排队、历史加载、断点续传 |
| `SessionService` | `bindServices`（随 Agent 组件） | 会话列表增删查，会话状态/标题同步 |
| `TaskService` | `bindServices`（随 Agent 组件） | 任务列表分页加载，任务数据更新 |

**页面双层组件模式**（规避 `bindServices` 类型推断限制）：

```tsx
// 外层：注册 Service（bindServices 完成依赖注入容器初始化）
const SkuAgentPage = bindServices(SkuAgentContent, [ChatService, SessionService, TaskService]);

// 内层：observer 包裹，直接读取 Service 响应式属性
const SkuAgentContent = observer(({ agentType, globalEventService }) => {
  const chatService = useService(ChatService);
  // chatService.currentSessionId$ 变化时组件自动重渲染
});
```

> 详细设计见 [`tech/STATE_MANAGEMENT.md`](./tech/STATE_MANAGEMENT.md)

---

### 5.7 消息排队管理（QueuePanel）

用户可在 AI 还未回复时连续发送消息，多条消息有序排队等待处理。

**核心交互**：

- **取消**：调用 `cancelMessage(qid)`；若取消的是当前响应中的消息，需先 `clearCurrentStream()` 将停止按钮切回发送状态
- **中断并发送**：绕过 ChatArea 输入框流程，需在 `chatService.outboundPendingUserMessage$` 中预存用户消息（保证聊天区立即显示用户气泡），再调用 `sendMessage` 携带 `queueQid` 让后端识别插队意图
- **撤销排队消息**：删除队列中尚未开始处理的消息，不影响当前正在响应的消息

---

### 5.8 文件上传（McS3Upload 封装）

```
用户拖拽 / 点击上传（支持 Excel、图片）
  → McS3Upload 组件 → 调用 uploadFileToS3()
  → 返回 UploadedFileInfo { url, name, mimeType, size }
  → 与消息文本一起通过 chatService.sendMessage(text, sessionId, files) 发送给 AI
```

表格提交时单独走 `downloadTableToS3` 接口（POST 表格数据 → 后端生成 Excel → 返回 S3 URL），不经过 McS3Upload 上传流程。

---

## 六、完整数据流转图

```
用户输入文字 / 拖拽上传文件
  │
  ▼
ChatArea 输入框 → chatService.sendMessage(content, sessionId, files?)
  │                  │
  │                  └── POST /api/send → 返回 queueItem(qid)
  │
  ▼
SSE /api/events（GlobalEventService 常驻连接，agent 切换不断开）
  ├── task.updated       → TaskService 更新任务列表
  ├── session.updated    → SessionService + ChatService 更新会话状态/标题
  ├── queue.updated      → ChatService.queue$ 更新 QueuePanel 显示
  ├── kicked             → 弹出异地登录提示弹窗
  └── message.add        → 触发加载对应消息的 SSE 流

SSE /api/message/{messageId}（ChatService 按消息建立独立连接）
  ├── message.start      → 初始化消息流（sessionId 守卫）
  ├── block              → 新增 block（整体替换 stream 对象触发重渲染）
  ├── block.append       → 追加内容到已有 block
  ├── block.delta        → Markdown 增量更新
  ├── message.replace    → 整体替换消息流
  ├── message.resume     → 断点续传，恢复历史 blocks
  └── message.done       → 消息结束

消息块 → blockRenderers 映射 → 各专属卡片组件渲染
  ├── markdown           → MarkdownRenderer（支持 GFM）
  ├── thinking           → ThinkingBlock（可折叠）
  ├── process_card       → ProgressCard（进度步骤可视化）
  ├── fold_process_card  → FoldProcessCard（折叠过程摘要）
  ├── result_card        → ResultCard（多 block 编排容器）
  ├── jiemo_link         → JiemoLink（点击后切换到内嵌浏览器打开）
  └── modify_excel_card  → ModifyExcelCardSummary（摘要）→ 展开 Dialogue 侧栏

Dialogue 侧栏（EditableTable 可编辑表格）
  → 用户修改 + 点击提交
  → POST /api/table/export → 生成 Excel → 返回 S3 URL
  → chatService.sendMessage(submitPrompt, sessionId, [{ url: s3Url }])
  → AI 收到 Excel 继续建档流程（完整闭环）
```

---

## 七、详细技术文档索引

| 文档 | 内容 |
|------|------|
| [`tech/MODULES.md`](./tech/MODULES.md) | 目录结构与模块职责划分 |
| [`tech/COMPONENTS.md`](./tech/COMPONENTS.md) | 组件分类、Props 规范、Keeta Design PC 使用规范 |
| [`tech/SSE_IMPLEMENTATION.md`](./tech/SSE_IMPLEMENTATION.md) | SSE 连接管理、事件分发、断点续传、会话隔离防御体系 |
| [`tech/STATE_MANAGEMENT.md`](./tech/STATE_MANAGEMENT.md) | RSJS Service 设计模式、响应式属性、依赖注入 |
| [`tech/API_INTERFACE.md`](./tech/API_INTERFACE.md) | HTTP 接口与 SSE 事件流详细规范 |
| [`../NEW_AGENT_SOP.md`](../NEW_AGENT_SOP.md) | 新 Agent 开发标准操作流程（复用矩阵、步骤、骨架代码） |
