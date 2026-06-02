# Keemart AI 助手技术架构设计文档

> 最后更新：2026-06-02

---

## 一、系统整体架构

整个 AI 助手体系采用 **「外壳 + 子应用 iframe」的微前端架构**，核心通信手段是 `window.postMessage`。

```
┌────────────────────────── osg-fe-jiemo-pc-w（外壳） ──────────────────────────────┐
│  Header (main.vue)                                                                  │
│  ├── AIToolsContainer   （AI 工具集：一键美化 / 生成场景图 / 语言专家）              │
│  └── Translation        （旧版 ChatGPT 翻译工具，独立浮层，已逐步被语言专家替代）   │
│                                                                                      │
│  utils/aiToolsCommunication.js  ← 外壳侧通信协议定义                                │
└────────────────────────────────────────────────────────────────────────────────────┘
                    ↕ postMessage（source=jiemo-product-subapp / jiemo-pc-shell）
┌────────────────── osg-fe-jiemo-product-w（子应用 iframe） ─────────────────────────┐
│  global.js                                                                           │
│  └── Vue.use(AITextSelector)   ← 全局文本选中唤醒插件（语言专家）                   │
│                                                                                      │
│  src/utils/aiToolsCommunication.js       ← 子应用侧通信协议定义（镜像实现）          │
│  src/components/AITextSelector/index.js  ← 翻译工具全局唤醒插件                     │
│                                                                                      │
│  src/pages/operateContentManage/components/detail.vue                                │
│  └── AI 图片工具唤醒：showAIContextMenu / handleAIContextMenuSelect / handleAIFillBack│
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、AIToolsContainer 组件架构（外壳侧）

### 2.1 文件结构

```
osg-fe-jiemo-pc-w/src/components/Menu/Header/AITools/
├── AIToolsContainer.vue     ← 主容器（工具集入口、模块切换、拖拽、通信）
├── ImageEditor.vue          ← 一键美化（图片编辑）模块
├── ImageCreator.vue         ← 生成场景图（图片生成）模块
├── LanguageSpec.vue         ← 语言专家（快捷翻译）模块
├── ImageDetailModal.vue     ← 图片详情弹窗
├── PicDetailParamsModal.vue ← 图片参数详情弹窗
└── constant.js              ← 公共常量（状态枚举、模型列表、参数字段映射、下载工具）
```

### 2.2 入口挂载

`AIToolsContainer` 在 `main.vue` 的 Header 右侧导航区挂载，无权限门控（任何人均可见）：

```html
<!-- main.vue Header 右侧 -->
<AIToolsContainer ref="aiToolsContainer" />
```

关闭其他 Header 浮层时，通过 `ref` 调用 `close()` 方法：

```js
// closeHeaderOverlays()
if (this.$refs.aiToolsContainer) {
    this.$refs.aiToolsContainer.close();
}
```

### 2.3 核心状态

| 状态字段 | 类型 | 说明 |
|---|---|---|
| `visible` | Boolean | 工具集容器是否展开 |
| `activeModule` | String/null | 当前激活工具：`imageEditorList` / `imageCreatorMain` / `languageSpec` / `null`（首页） |
| `imageEditorState` | Object | 图片编辑子状态：`{ originalPic, isFullscreen }` |
| `imageCreatorState` | Object | 图片生成子状态：`{ originalPic, isEditing, isFullscreen }` |
| `historyActive` | Boolean | 历史面板 icon 是否激活 |
| `sharedHistoryVisible` | Boolean | 两个图片工具共享的历史面板状态 |
| `runningPicNums` | Number | 生成中任务数量（首页角标） |
| `donePicNums` | Number | 已完成/未读图片数量（首页角标） |
| `currentSkuId` | String | 跨工具共享的 skuId |
| `currentWakeData` | Object/null | 子应用唤醒时携带的完整上下文 |
| `subAppContentWindow` | Window/null | 唤醒消息来源的 contentWindow（用于精准回填） |
| `containerX/Y` | Number | 拖拽定位坐标，`-1` 表示未初始化 |
| `hasBeenDragged` | Boolean | 用户是否已手动拖动过（关闭后重置） |

### 2.4 容器宽度状态

| 显示状态 | 宽度 |
|---|---|
| 首页 / 语言专家 | `430px` |
| 图片模块（进入即展示左右双栏） | `705px`（275px 左侧面板 + 430px 展示区） |
| 历史面板展开 + 图片模块 | `813px`（705 + 108） |
| 全屏 | `100vw × 100vh` |

### 2.5 模块加载策略

子模块全部使用**异步组件**懒加载，避免首屏加载全部工具；模块切换使用 `v-show`，保证 `$refs` 始终可用：

```js
components: {
    ImageEditor:  () => import('./ImageEditor'),
    ImageCreator: () => import('./ImageCreator'),
    LanguageSpec: () => import('./LanguageSpec'),
},
```

### 2.6 父子通信模式

```
AIToolsContainer（父）
  ↓ ref 调用子组件方法
  - fillData(data)         → 填充数据（唤醒时调用）
  - toggleHistory()        → 切换历史面板
  - queryHistory()         → 请求历史数据
  - updateSkuId(id)        → 更新 skuId
  - _stopRunningPoll()     → 停止生成轮询

  ↑ emit 通知父容器
  - @state-change          → 子状态变更（originalPic, isFullscreen 等）
  - @history-change        → 历史面板展开/收起
  - @mark-read             → 标记已读，触发角标刷新
```

---

## 三、三大工具业务功能与技术实现

### 3.1 一键美化（ImageEditor）

**文件：** `AITools/ImageEditor.vue`（约 2596 行）

#### 业务功能

一键美化用于对已有商品图片进行 AI 增强处理，核心是通过一套标准化参数将原图提交给后端 AI 服务，等待处理完成后展示效果并支持回填到商品详情。

**处理参数（8 种开关 + 1 个数值）：**

| 参数字段 | 含义 | 默认值 |
|---|---|---|
| `needRemoveWaterMark` | 去除水印 | `true` |
| `needImproveClarity` | 增强清晰度 | `true` |
| `needCutout` | 抠图 | `true` |
| `needBrighten` | 调色 | `false` |
| `needRemoveDecorativeElements` | 去除装饰性元素 | `false` |
| `needRemoveWords` | 去除文字 | `false` |
| `needSubjectCompletion` | 主体补全 | `false` |
| `needRemoveFrame` | 去除黑白边框 | `false` |
| `mainImagePercent` | 主图占比（缩放比例） | `90%` |
| `prompt` | 图像自然语言指令 | `''` |

**生成结果：** 单张图片 + 图片质量诊断报告（宽高比 / 原图相似度 / 清晰度）

**历史记录：** 展示所有历史生成记录，分进行中（置顶）、完成未读、完成已读、失败、取消五种状态。

#### API 接口

| 接口 | 说明 | 关键参数 |
|---|---|---|
| `POST .../AIToolTService/editPicture` | 提交编辑任务 | `{ skuId, picUrl, needRemoveWaterMark, needImproveClarity, needCutout, ... }` → `taskId` |
| `POST .../AIToolTService/queryHandleResult` | 轮询生成结果 | `{ taskId }` → `[{ picId, picUrl }]` |
| `POST .../AIToolTService/queryDiagnoseResult` | 轮询图片质量诊断 | `{ picId }` → `{ picQuality, diagnoseInfoList }` |
| `POST .../api/AIToolTService/queryPicDetail` | 查询图片详情参数 | `{ picId }` → `{ taskId, exeParams }` |
| `POST .../api/AIToolTService/cancelTask` | 取消任务 | `{ taskId }` |
| `POST .../AIToolTService/queryPicHistoryPage` | 历史记录分页 | `{}` → `[{ picId, picUrl, status }]` |
| `POST .../api/AIToolTService/queryRunningPicInfo` | 查进行中任务 | `{}` → `[{ picId, originalPicUrl }]` |
| `POST .../api/AIToolTService/markHasRead` | 标记已读 | `{ picId }` |

#### 核心业务流程

```
子应用唤醒 / 用户上传图片
        ↓
fillData(data) → 参数重置 → queryEditPicture()
        ↓
POST editPicture { skuId, picUrl, 8种参数, prompt }
        ↓
并行启动：
  ① _startRunningPoll()：每 3s 轮询 queryRunningPicInfo（更新左侧历史进行中列表）
  ② queryAIResult()：每 3s 轮询 queryHandleResult，最多 40 次（≈2分钟超时）
        ↓
轮询返回 [{ picId, picUrl }] 非空 → 停止轮询
        ↓
aiPic = picUrl，显示生成结果
queryHistory() 刷新历史面板
_startDiagnosis() → queryDiagnosisResult() 每 6s 轮询，最多 20 次
        ↓
诊断完成 → aiQualityList（指标展示）、picQuality（整体质量星级）
```

#### 技术难点

**① 三路轮询协调**

同时运行三条独立轮询链：生成任务轮询、诊断结果轮询、进行中任务轮询。需要保证：
- 取消（`handleCancel`）时立即停止生成和诊断轮询
- 新一轮生成时先调用 `_stopDiagnosis()`，再重新发起
- 组件销毁时（`beforeDestroy`）全部清除

```js
// 非响应式中止 flag，避免触发不必要的视图更新
this._diagnosisAborted = false;
this._cancelled = false;
this._runningPollTimer = null;
```

**② fetchPicDetail 竞态保护**

用户快速切换历史记录时，多个异步请求可能乱序返回，后发先至会导致详情数据错乱：

```js
this._fetchPicDetailSeq = (this._fetchPicDetailSeq || 0) + 1;
const seq = this._fetchPicDetailSeq;
const res = await hPost(...);
if (seq !== this._fetchPicDetailSeq) return;  // 已被新请求覆盖，丢弃
this.picDetailParams = res.data;
```

**③ 参数标签二行截断算法（_measureParamTags）**

生成的参数标签需显示前两行，第二行末尾显示 `+N`。由于标签宽度不均等，无法用字数截断，必须用 DOM 测量：

```
1. 遍历标签，找到"溢出第三行的第一个"位置 cutAt
2. 在 cutAt 插入 +N 占位节点（probe）
3. 检测 probe.offsetTop 是否仍在第三行
4. 若在 → cutAt--，重新插入 probe 直到 +N 落入第二行
5. 清理 probe，更新 visibleTagCount
（双 $nextTick 等待 DOM 渲染完毕后才执行）
```

**④ 进行中任务占位卡片**

历史面板需要展示正在进行的任务（后端未返回前），通过在前端维护 `runningPicList`，在 `mergedHistoryList` computed 属性中将其置顶：

```js
computed: {
    mergedHistoryList() {
        // 进行中：置顶，展示原图 + loading 蒙层
        const running = this.runningPicList.map(item => ({
            ...item, _isRunning: true
        }));
        // 历史：status=20(完成未读) / 40(已读) / 30/50(失败)
        return [...running, ...this.historyList];
    }
}
```

---

### 3.2 生成场景图（ImageCreator）

**文件：** `AITools/ImageCreator.vue`（约 2684 行）

#### 业务功能

生成场景图基于商品原图，通过选择 AI 模型、场景类型和图片数量，一次生成多张商品主图。用于快速生成各种场景下的商品展示图，减少拍摄成本。

**生成参数：**

| 参数 | 选项 | 说明 |
|---|---|---|
| `model` | Gemini-2.5 / Chatgpt-image-1.5 | AI 生成模型 |
| `scene` | 自动推荐 / 电商场景 / 餐桌场景 / 卖点突出 / 自定义 | 场景类型 |
| `picNum` | 1 ~ 6 张 | 一次生成图片数量，默认 4 张 |
| `prompt` | 自然语言 | 自定义场景描述（scene=自定义时） |

**生成结果：** 多张图片（最多 6 张），支持切换大图预览、单张回填、单张下载。

**与 ImageEditor 的核心差异：**

| 维度 | ImageEditor | ImageCreator |
|---|---|---|
| 生成结果数 | 单张 | 多张（最多 6 张） |
| 结果存储 | `aiPic: String` | `aiPicList: Array` + `selectedPicIndex` |
| 参数类型 | 8 种布尔开关 + 数值 + prompt | 模型选择 + 场景选择 + 数量 |
| 初始化快照 | `initialAiPic`（单图 URL） | `initialPicList`（`[{picUrl, picId}]`） |
| 诊断轮询次数 | 最多 20 次（6s间隔） | 最多 30 次（6s间隔） |

#### API 接口

| 接口 | 说明 | 关键参数 |
|---|---|---|
| `POST .../AIToolTService/createPicture` | 提交生成任务 | `{ picUrl, skuId, model, prompt, scene, picNum }` → `taskId` |
| `POST .../AIToolTService/queryHandleResult` | 轮询生成结果 | `{ taskId }` → `[{ picId, picUrl }]`（多张） |
| （其余接口与 ImageEditor 相同） | | |

#### 核心业务流程

```
fillData() → 清空旧结果 → queryAICreate()
        ↓
POST createPicture { picUrl, skuId, model, scene, picNum, prompt }
        ↓
轮询 queryHandleResult → 返回 aiPicList（多张）
        ↓
selectedPicIndex = 0（默认展示第一张）
initialPicList = aiPicList.map(url => ({ picUrl, picId: '' }))  ← 历史面板快照
        ↓
queryHistory() + _startDiagnosis()
        ↓
点击缩略图 → selectedPicIndex 切换大图
```

#### 技术难点

**① 多图历史快照问题**

ImageEditor 结果为单图，失去当前 `aiPic` 后可从历史接口恢复。但 ImageCreator 返回多图，历史接口只存储每个任务的代表图（非全部），需要在生成完成时立即保存完整快照：

```js
// 保存完整 { picUrl, picId } 快照，用于历史面板展示缩略图
this.initialPicList = aiPicList.map(url => ({
    picUrl: url, picId: ''
}));
// 真正的 picId 在 queryDiagnosisResult 成功后才能填入
```

**② 并行历史数据聚合（Promise.allSettled）**

历史面板同时请求历史列表和进行中任务两个接口，需要两个都完成才能合并展示：

```js
const [historyRes, runningRes] = await Promise.allSettled([
    queryPicHistoryPage(),
    queryRunningPicInfo()
]);
// allSettled 保证即使一个失败，另一个仍然展示
if (historyRes.status === 'fulfilled') { ... }
if (runningRes.status === 'fulfilled') { ... }
```

**③ 场景分组 UI 与缩略图映射**

`sceneGroups` 将场景列表分组为「基础 / 更多」两组，并通过缩略图 URL 映射渲染预览图，场景切换时高亮选中样式。

**④ 全局点击监听关闭下拉**

数量选择和模型选择均为自定义下拉，需要在 `mounted` 中注册全局 `click` 监听器，在 `beforeDestroy` 中移除，并在 `handleGlobalClick` 中统一关闭两个下拉面板：

```js
mounted() {
    document.addEventListener('click', this.handleGlobalClick);
},
beforeDestroy() {
    document.removeEventListener('click', this.handleGlobalClick);
}
```

---

### 3.3 语言专家（LanguageSpec）

**文件：** `AITools/LanguageSpec.vue`（约 982 行）

#### 业务功能

语言专家是一个多语言机器翻译工具，支持中文、英文、阿拉伯文三种语言互译。核心特性是"零操作翻译"——用户选中页面上的任意文本后，翻译结果会自动填充进来并触发翻译，无需手动复制粘贴。

**功能点：**
- 自动检测源语言（中/英/阿，基于字符集统计）
- 支持多个目标语言同时翻译（默认英文 + 阿拉伯文）
- 翻译结果以卡片形式叠加展示（每次翻译追加，不覆盖历史）
- 结果卡片内支持切换显示的目标语言
- 一键复制翻译结果（支持 Clipboard API + execCommand 降级）
- RTL（从右到左）支持：阿拉伯文卡片自动切换文字方向
- 支持互换源/目标语言（单目标语言时有效）
- 支持中途停止翻译（`abortFlag` 控制）

#### API 接口

| 接口 | 方法 | 参数 | 返回 |
|---|---|---|---|
| `POST /api/m/osgi18n/translator/machine_translation` | hPost | `{ source: "待译文本", target: ["en", "ar"] }` | `[{ lang: "en", res: "..." }, { lang: "ar", res: "..." }]` |

#### 核心业务流程

```
AITextSelector（全局插件）选中文本 → autoWakeAITools('languageSpec', { text })
        ↓
外壳 handleTextSelect → visible=true, activeModule='languageSpec'
        ↓
waitForRefAndFill → LanguageSpec.fillData({ text })
        ↓
autoDetectLang(text)
  阿拉伯字符最多 → sourceLang='ar', targetLangList=['zh', 'en']
  英文字符最多   → sourceLang='en', targetLangList=['zh', 'ar']
  中文字符最多   → sourceLang='zh', targetLangList=['en', 'ar']（默认）
        ↓
queryTranslate()
  → POST machine_translation { source: text, target: ['en', 'ar'] }
  → buildResultCards(sourceText, sourceLang, res)
        ↓
resultCards 追加新卡片（每张卡片保存全量 _fullTranslations）
        ↓
滚动到最新卡片
```

#### 语言自动检测算法

基于字符集正则匹配，统计各语种字符数量，以最多的作为检测结果：

```js
autoDetectLang(text) {
    const arCount = (text.match(/[\u0600-\u06FF]/g) || []).length;
    const enCount = (text.match(/[a-zA-Z]/g) || []).length;
    const zhCount = (text.match(/[\u4E00-\u9FFF]/g) || []).length;

    if (zhCount >= enCount && zhCount >= arCount) return 'zh';
    if (arCount >= enCount && arCount >= zhCount) return 'ar';
    return 'en';  // 默认英文
}
```

检测结果会同步更新 `targetLangList`（排除源语言，默认选中其余两种）。

#### 技术难点

**① Vue 2 响应式陷阱：卡片语言切换**

`resultCards` 是对象数组，切换卡片内显示的语言时需要替换整个 card 对象，否则 Vue 2 不会触发视图更新：

```js
handleToggleCardLang(cardIdx, langVal) {
    const card = { ...this.resultCards[cardIdx] };
    // 切换 selectedLangs（至少保留一种）
    if (card.selectedLangs.includes(langVal)) {
        if (card.selectedLangs.length === 1) return;
        card.selectedLangs = card.selectedLangs.filter(l => l !== langVal);
    } else {
        card.selectedLangs = [...card.selectedLangs, langVal];
    }
    // 从全量翻译中筛选当前显示的语种
    card.translations = card._fullTranslations.filter(
        t => card.selectedLangs.includes(t.lang)
    );
    this.$set(this.resultCards, cardIdx, card);  // 必须用 $set
}
```

**② RTL 文本渲染**

阿拉伯语是从右到左书写，结果卡片需要根据语言类型动态切换 CSS 方向：

```css
.translation-card[data-lang="ar"] .card-text {
    direction: rtl;
    text-align: right;
}
```

**③ 剪贴板 API 降级处理**

现代浏览器支持 `navigator.clipboard.writeText()`，但部分环境（HTTPS 限制、iframe 内嵌等）会失败，需要降级到 `execCommand('copy')`：

```js
handleCopyTranslation(text) {
    navigator.clipboard.writeText(text)
        .then(() => { /* 显示复制成功提示 */ })
        .catch(() => this._fallbackCopy(text));
},
_fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
}
```

**④ 全量翻译快照（_fullTranslations）**

每次翻译后，在卡片上同时保存"全量结果" `_fullTranslations` 和"当前显示" `translations`。这样切换卡片内显示的语言时无需重新请求接口，直接从全量中筛选：

```js
buildResultCards(sourceText, sourceLang, res) {
    return {
        sourceText,
        sourceLang,
        selectedLangs: this.targetLangList,
        _fullTranslations: res,           // 全量（接口返回的全部语言）
        translations: res.filter(        // 当前显示（按选中语言过滤）
            t => this.targetLangList.includes(t.lang)
        )
    };
}
```

---

### 3.4 三工具通用技术模式

#### 轮询参数对比

| 工具 | 生成轮询（次×间隔） | 诊断轮询（次×间隔） | 最大等待时间 |
|---|---|---|---|
| ImageEditor | 40 × 3s | 20 × 6s | ~2分钟 |
| ImageCreator | 40 × 3s | 30 × 6s | ~3分钟 |
| LanguageSpec | 无轮询 | 无轮询 | 接口同步返回 |

#### 图片下载技术方案

直接用 `<a download>` 无法跨域触发下载，需 XHR 将图片转为 Blob 再创建临时链接：

```js
// constant.js downloadImage 核心实现
const xhr = new XMLHttpRequest();
xhr.open('GET', url, true);
xhr.responseType = 'blob';
xhr.onload = function() {
    const objectUrl = URL.createObjectURL(xhr.response);
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = filename || `ai-image-${Date.now()}.jpg`;
    a.click();
    URL.revokeObjectURL(objectUrl);  // 及时释放内存
};
xhr.onerror = () => window.open(url, '_blank');  // 降级：新标签页打开
// 1s 防抖，避免重复点击多次下载
const debouncedDownload = debounce(_doDownload, 1000);
```

#### 历史状态枚举（共用 constant.js）

```js
PIC_HISTORY_STATUS = {
    IN_PROGRESS: 10,   // 进行中（前端维护，后端未返回前占位）
    DONE_UNREAD: 20,   // 已完成/未读（右上角橙色角标）
    FAILED: 30,        // 失败（灰色占位图）
    DONE_READ: 40,     // 已完成/已读
    CANCELLED: 50,     // 任务取消
}
```

---

## 四、通信协议层（双侧镜像实现）

### 4.1 事件类型

两个工程各自维护一份 `aiToolsCommunication.js`，事件常量完全一致：

| 事件常量 | 方向 | 说明 |
|---|---|---|
| `ai-tools:auto-wake` | 子应用 → 外壳 | 唤醒指定工具并填充图片/文本数据 |
| `ai-tools:text-select` | 子应用 → 外壳 | 用户点击"AI翻译"气泡后唤醒语言专家 |
| `ai-tools:text-copy` | 子应用 → 外壳 | **已弃用**，保留供回滚 |
| `ai-tools:fill-back-image` | 外壳 → 子应用 | 图片处理完成后回填到子应用指定字段 |
| `ai-tools:toggle` | 外壳 → 子应用 | 显示/隐藏工具（备用） |
| `ai-tools:ready` | 外壳 → 子应用 | 外壳准备就绪 |

### 4.2 消息来源标识

| 方向 | `source` 字段值 |
|---|---|
| 子应用 → 外壳 | `jiemo-product-subapp` |
| 外壳 → 子应用 | `jiemo-pc-shell` |

### 4.3 子应用侧主要工具函数

```js
// 唤醒外壳指定 AI 工具
autoWakeAITools(type, data)
  → postMessage({ type: 'ai-tools:auto-wake', payload: { type, data }, source: 'jiemo-product-subapp' })

// 监听外壳回填图片
listenFillBack(callback)
  → 监听 source='jiemo-pc-shell' + type='ai-tools:fill-back-image' 的消息

// 发送文本选中事件（用于唤醒语言专家）
sendTextSelectEvent(text)
  → postMessage({ type: 'ai-tools:text-select', payload: { text }, source: 'jiemo-product-subapp' })
```

---

## 五、翻译工具（语言专家）全局唤醒机制

### 4.1 旧版 Translation 组件（保留中）

`Translation.vue` 是早期方案，直接在 Header 渲染图标，点击展开 `el-popover` 浮层，手动输入文案后调用翻译接口：

- 接口：`POST /api/m/osgi18n/translator/machine_translation`
- 参数：`{ source: string, target: string[] }`
- 返回：`Array<{ lang, res }>`

### 4.2 新版全局翻译唤醒：AITextSelector 插件

在 `global.js` 全局注册，**无需改动任何业务页面**：

```js
// global.js
import AITextSelector from 'components/AITextSelector';
Vue.use(AITextSelector);
```

#### 工作流程

```
用户在页面中选中文本（≥2 个字符）
        ↓
全局 mouseup 监听器触发
        ↓
检测输入框锚点（el-input / mc-input / mc-l10n-tiled-input）→ 菜单定位策略
        ↓
显示 .ai-text-selector-popup 菜单（position: fixed，贴选区/输入框底部）
        ↓
用户点击"语言专家"
        ↓
autoWakeAITools('languageSpec', { text, isEdit: true })
        ↓
postMessage AUTO_WAKE → 外壳 handleTextSelect
        ↓
visible=true, activeModule='languageSpec'
waitForRefAndFill('languageSpec', { text })
        ↓
LanguageSpec.fillData({ text }) → autoDetectLang → queryTranslate()
```

#### 三种选中场景

| 场景 | 检测方式 | 菜单定位 |
|---|---|---|
| 普通文本选区 | `window.getSelection()` | 跟随 `range.getBoundingClientRect()` 右下角 |
| 输入框内选中 | `detectInputAnchor()` 向上识别组件容器 | 贴输入框左下角 |
| Radio/Checkbox label 单击 | `detectRadioLabel()` 检测 label 文本 | 贴 label 右下角 |

#### 已弃用方案对比

| 方案 | 状态 | 说明 |
|---|---|---|
| 文本复制唤醒（`TEXT_COPY`） | 已弃用 | 监听 `copy` 事件，已注释关闭 |
| `handleDocumentCopy` | 已弃用 | 外壳层 copy 事件监听，已注释关闭 |
| 当前方案（`TEXT_SELECT` + 气泡） | **线上** | 选中后点击气泡按钮主动触发 |

---

## 六、图片工具唤醒机制（detail.vue）

### 5.1 支持 AI 工具的图片字段

`operateContentManage/components/detail.vue` 中共 5 种图片字段挂载了 AI 工具入口：

| 图片字段 | 上传组件 | picType 标识 |
|---|---|---|
| 列表用图 | `ImageUpload` | `lbPicture` |
| 商品主图 | `MainPicUpload` | `skuPictureList` |
| 实拍图 | `MainPicUpload` | `detailPics` |
| 细节图-营养成分图 | `DetailPicUpload` | `nutritionDetailPic` |
| 细节图-配料图 | `DetailPicUpload` | `ingredientDetailPic` |

每个组件均绑定：
- `@ai-tool-click="(url, idx, e) => showAIContextMenu(e, url, 'picType', idx)"`
- `@ai-tool-leave="scheduleCloseMenu"`

### 5.2 AI 右键浮动菜单

菜单 DOM 挂载在根 `<div>` 下（`xx-layout` 外层），避免被内层 `overflow: hidden` 裁剪：

```html
<div
  v-if="aiContextMenu.visible"
  class="ai-context-menu-popup"
  :style="{ left: aiContextMenu.x + 'px', top: aiContextMenu.y + 'px' }"
  @mouseenter="clearCloseTimer()"
  @mouseleave="closeAIContextMenu()"
>
  <div @click="handleAIContextMenuSelect('imageEditorList')">一键美化</div>
  <div @click="handleAIContextMenuSelect('imageCreatorMain')">生成场景图</div>
</div>
```

菜单状态：

```js
aiContextMenu: {
  visible: false,
  x: 0, y: 0,
  picUrl: '',      // 当前图片 URL
  picType: '',     // 图片字段类型
  picIndex: -1,    // 在列表中的下标（-1 表示追加）
}
```

### 5.3 唤醒 AI 工具完整流程

```
用户 hover 图片 → 图片组件 emit @ai-tool-click
        ↓
showAIContextMenu(e, picUrl, picType, picIndex)
  计算位置：右对齐图片，超出视口则显示在图片上方
        ↓
用户点击菜单项（一键美化 / 生成场景图）
        ↓
handleAIContextMenuSelect(toolType)
  autoWakeAITools(toolType, {
      originalPic: picUrl,
      skuId: ...,
      fillContext: { picType, picIndex },
      isEdit: this.mode !== 'show'
  })
        ↓
postMessage → source: 'jiemo-product-subapp', type: 'ai-tools:auto-wake'
        ↓
外壳 AIToolsContainer.handleAutoWake({ type, data })
  visible=true, activeModule=type
  waitForRefAndFill(type, data)  // 轮询等待异步组件 $refs 就绪（100ms × 30次）
        ↓
ImageEditor/ImageCreator.fillData({ originalPic, skuId, fillContext, isEdit })
```

### 5.4 图片回填完整流程

```
用户在 AI 工具中点击"使用此图"
        ↓
外壳 fillBackImageToIframe(subAppContentWindow, picType, picUrl, picIndex)
  优先用缓存的 subAppContentWindow（来自 AUTO_WAKE 的 event.source）
  降级：DOM 查找第一个 iframe
        ↓
postMessage → source: 'jiemo-pc-shell', type: 'ai-tools:fill-back-image'
payload: { picType, picUrl, picIndex }
        ↓
子应用 listenFillBack → handleAIFillBack({ picType, picUrl, picIndex })
  ① loadImageAsync(picUrl) 获取真实宽高（失败降级 800×800）
  ② 按 picType 分类回填：
     lbPicture       → updateField('lbPicture', ...)            覆盖 1 张
     skuPictureList  → 追加或替换指定下标（最多 6 张）
     detailPics      → 追加或替换指定下标
     nutritionDetailPic / ingredientDetailPic → 追加到 ztPictureList 对应类型
```

### 5.5 菜单关闭机制（防止 hover 间隙闪烁）

```
图片组件 @ai-tool-leave
    → scheduleCloseMenu()  → setTimeout 150ms → visible=false
菜单 @mouseenter
    → clearCloseTimer()    → 取消关闭定时器
document click
    → closeAIContextMenu() → 立即关闭
```

---

## 七、生命周期与事件注册

### detail.vue 生命周期

| 钩子 | 操作 |
|---|---|
| `mounted` | 注册 `listenFillBack` 回调、注册 `document click` 关闭菜单 |
| `beforeDestroy` | 取消 `listenFillBack` 监听、移除 `document click` 监听、清除关闭定时器 |

### AIToolsContainer 生命周期

| 钩子 | 操作 |
|---|---|
| `created` | 注册 `mousemove/mouseup`（拖拽）、`window message`（跨 iframe 通信）、`document click`（关闭下拉） |
| `beforeDestroy` | 移除所有以上监听 |

---

## 八、角标与历史记录

### 首页角标逻辑

每次打开工具首页 / `@mark-read` 事件触发时调用接口刷新角标：

```
POST /shepherd/merchandise/management/api/AIToolTService/queryTargetPicStatusInfo
→ { runningPicNums, donePicNums }
```

图片任务状态枚举：

| 值 | 含义 |
|---|---|
| `10` | 进行中 |
| `20` | 已完成/未读 |
| `30` | 失败 |
| `40` | 已完成/已读 |
| `50` | 任务取消 |

### 历史面板共享状态

两个图片工具（ImageEditor / ImageCreator）通过 `sharedHistoryVisible` 共享历史面板展开状态。切换工具时，新工具同步到相同状态并重新请求数据。

---

## 九、关键设计决策

| 决策点 | 选择方案 | 原因 |
|---|---|---|
| 跨 iframe 通信 | `postMessage` | 外壳与子应用独立 JS 上下文 |
| 异步组件等待 | 轮询重试（100ms × 30次） | `$nextTick` 不足以等待异步组件首次渲染完成 |
| 子组件引用 | `v-show` + `$refs`（非 `v-if`） | 避免切换时 `$refs` 为 null |
| 图片回填 iframe 定位 | 优先缓存 `event.source`，降级 DOM 查找 | 页面可能有多个 iframe，精准定位避免误发 |
| 翻译工具唤醒 | 全局 `AITextSelector` 插件 | 无需修改各业务页面，低侵入 |
| 菜单 DOM 挂载位置 | 根 `<div>`（`xx-layout` 外层） | 避免被 `xx-layout` 内层 `overflow:hidden` 裁剪 |
| AI 图片浮层 z-index | `$z-header-ai-overlay`（独立变量） | 低于 Element UI 默认 2000+，不遮盖语言/地区下拉 |
