![image](https://github.com/user-attachments/assets/808ee9ce-72dd-4511-ad6a-8ce8bec0ceb4)# 开篇词 (1讲) 开篇词｜开发工程师如何进阶为AI应用型人才？
## 转型为 AI 应用型工程师的难点
![image](https://github.com/user-attachments/assets/94da3a81-a88c-49e3-b2c1-d967711c3729)

## 开发工程师入局 AI 的最佳路径
![image](https://github.com/user-attachments/assets/e2cdb21d-36ed-433a-8d32-e488d9686907)

## 这门课程是如何设计的？
根据这条最佳的学习路径，由浅入深地把课程分成了 5 个章节。

![image](https://github.com/user-attachments/assets/a8f77474-f072-4699-99c4-73e44c19bf6b)

# 第一章：小试牛刀，理解基础概念 (3讲) 01｜洞察本质：从工程学角度看ChatGPT为什么会崛起
ChatGPT 具体是如何赢得这场胜利的呢？我们一一来看。

## NLP 技术突破：强势整合技术资源
![image](https://github.com/user-attachments/assets/b53e236f-ea06-477a-92a3-7041c7cb47c2)

## 基于自回归的无监督训练
![image](https://github.com/user-attachments/assets/dba9fc35-117d-42e0-a170-6b6e11464c9b)

## 与人类意识对齐（Alignment）
![image](https://github.com/user-attachments/assets/b7c0808b-e5c5-4071-9b98-e57f007a50e4)

![image](https://github.com/user-attachments/assets/800caa4d-a5d7-4fe2-ad11-76009fbf412f)

![image](https://github.com/user-attachments/assets/f0312037-dd20-44fb-900a-fa1844925798)

## 突现能力（Emergent Ability）
![image](https://github.com/user-attachments/assets/457e0e37-76e6-4712-9277-88431d1e5c67)

![image](https://github.com/user-attachments/assets/b8d87364-159c-4b27-b3a2-a50bdf0fbd0a)

## 超大规模数据集：超过 40T 的文本数据
![image](https://github.com/user-attachments/assets/fbeb4fa4-9b34-4b86-81aa-caa616e9ce4d)

![image](https://github.com/user-attachments/assets/2e859aa0-d546-47e1-8788-59187f88a60c)

## 找对了金主爸爸
![image](https://github.com/user-attachments/assets/f1ba684b-3f2b-4dec-a244-233e560cbe33)

## 产品化开放：让大家随便玩
![image](https://github.com/user-attachments/assets/49617884-3b97-4901-b984-dbf9b2c76798)

## 便捷使用
![image](https://github.com/user-attachments/assets/b06b3906-f538-4589-9948-fd7936ef2080)

## 适用场景多
![image](https://github.com/user-attachments/assets/c224a392-c88f-4bad-960e-2ccc73571280)

## 使用效果好
![image](https://github.com/user-attachments/assets/f26ea80c-08bd-4113-a287-905c30eb54fb)

## 工程化应用
![image](https://github.com/user-attachments/assets/4273890d-934d-4fde-9f47-11e3b2425e96)

## 小结
![image](https://github.com/user-attachments/assets/a055a795-744a-4f72-bbd0-f5172f413ca1)

# 第一章：小试牛刀，理解基础概念 (3讲) 02｜学好提示工程，轻松驾驭大模型
**能否充分使用好 AI 大模型，提示是关键**。

## 什么是提示？
![image](https://github.com/user-attachments/assets/c6277e4d-a968-4c68-8687-f498c09f4bd3)

## 什么是提示工程？
![image](https://github.com/user-attachments/assets/d97ed688-f273-434d-95ae-e850c8cb4e9d)

## 什么是 AI 领导力？
![image](https://github.com/user-attachments/assets/1d15492c-63f2-4f68-991a-949f3b3490ce)

## 如何构造好的提示？
![image](https://github.com/user-attachments/assets/d6fef219-4c20-4133-aae4-8f824a8a87bc)

![image](https://github.com/user-attachments/assets/4ed235b8-2d62-46d6-9391-f569fb4963e0)

```
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class LoginModule {
    private Map<String, String> userDatabase;

    public LoginModule() {
        // 初始化用户数据库，实际应用中通常会连接到数据库
        userDatabase = new HashMap<>();
        userDatabase.put("user1", "password1");
        userDatabase.put("user2", "password2");
        // 添加更多用户...
    }

    public boolean authenticateUser(String username, String password) {
        // 在实际应用中，这里通常会连接到数据库，验证用户名和密码是否匹配
        String storedPassword = userDatabase.get(username);
        return storedPassword != null && storedPassword.equals(password);
    }

    public static void main(String[] args) {
        LoginModule loginModule = new LoginModule();
        Scanner scanner = new Scanner(System.in);

        System.out.print("请输入用户名: ");
        String username = scanner.nextLine();

        System.out.print("请输入密码: ");
        String password = scanner.nextLine();

        if (loginModule.authenticateUser(username, password)) {
            System.out.println("登录成功！");
        } else {
            System.out.println("登录失败，用户名或密码错误！");
        }

        scanner.close();
    }
}

```

![image](https://github.com/user-attachments/assets/b24af86c-ef54-495b-8070-fc4b39ad017d)

```
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class UserController {

    @PostMapping("/login")
    public String login(@RequestBody UserLoginRequest request) {
        // 第二步：非空验证
        if (request.getUsername() == null || request.getPassword() == null) {
            return "用户名和密码不能为空";
        }

        // 第三步：数据库验证
        if (isValidUser(request.getUsername(), request.getPassword())) {
            return "登录成功";
        } else {
            return "用户名或密码错误";
        }
    }

    private boolean isValidUser(String username, String password) {
        // 这里应该调用数据库查询用户信息的方法，进行用户名和密码的验证
        // 在实际项目中，不应该将密码存储为明文，而是使用哈希算法进行存储和比对
        // 这里仅为演示，实际中应该使用更安全的方式
        return "admin".equals(username) && "21232f297a57a5a743894a0e4a801fc3".equals(md5(password));
    }

    private String md5(String input) {
        // 这里是一个简单的MD5加密示例，实际项目中应该使用更安全的加密方式
        // 注意：MD5不是安全的加密算法，仅作为演示使用
        // 实际项目中应该使用更安全的哈希算法，如BCrypt
        // 可以使用Spring Security等库来进行密码的安全处理
        return org.apache.commons.codec.digest.DigestUtils.md5Hex(input);
    }
}

```

![image](https://github.com/user-attachments/assets/0f2a0ae0-76e5-4565-a41a-1facc321e022)

## 万能工具：生成提示的提示
![image](https://github.com/user-attachments/assets/e5bca8a7-67e0-4695-a3d1-32d70c6ea1a1)

## 写给软件工程师的话
![image](https://github.com/user-attachments/assets/f6a44b6e-ad40-4518-92a3-bb36c7dde8d7)

## 小结
![image](https://github.com/user-attachments/assets/678c489b-d2ef-4a79-a2bd-c57126cee709)

## 思考题
请你思考一下，我们为什么要为大模型指定角色呢？欢迎你把你思考后的结果分享到评论区，也欢迎你把这节课的内容分享给其他朋友，我们下节课再见！

# 第一章：小试牛刀，理解基础概念 (3讲) 03｜探索智能体世界：LangChain与RAG检索增强生成
![image](https://github.com/user-attachments/assets/eecca5d0-068a-40be-80c3-7dc68112c164)

## AI Agent
AI Agent 就是以大语言模型为核心控制器的一套代理系统。

举个形象的例子：如果把人的大脑比作大模型的话，眼睛、耳朵、鼻子、嘴巴、四肢等联合起来叫做 Agent，眼睛、耳朵、鼻子感知外界信号作为大脑的输入；嘴巴、四肢等根据大脑处理结果形成反馈进行输出，形成以大脑为核心控制器的一套系统。

![image](https://github.com/user-attachments/assets/c1807e0e-6408-41de-8363-517a7d205c9b)

控制端处于核心地位，承担记忆、思考以及决策制定等基础工作，感知模块则负责接收和处理来自外部环境的多样化信息，如声音、文字、图像、位置等，最后行动模块通过生成文本、API 调用、使用工具等方式来执行任务以及改变环境。

相信通过这样的介绍，你就明白智能体的概念和结构了，接下来我给你介绍一些常见的智能体技术，下面我们统称为 Agent。目前比较流行的 Agent 技术有 AutoGen、LangChain 等，因为 LangChain 既是开源的，又提供了一整套围绕大模型的 Agent 工具，可以说使用起来非常方便，而且从设计到构建、部署、运维等多方面都提供支持，所以下面我们主要介绍一下 LangChain 的应用场景。

## LangChain 介绍
起初，LangChain 只是一个技术框架，使用这个框架可以快速开发 AI 应用程序。这可能是软件开发工程师最容易和 AI 接触的一个点，因为我们不需要储备太多算法层面的知识，只需要知道如何和模型进行交互，也就是熟练掌握模型暴露的 API 接口和参数，就可以利用 LangChain 进行应用开发了。

LangChain 发展到今天，已经不再是一个纯粹的 AI 应用开发框架，而是成为了一个 AI 应用程序开发平台，它包含 4 大组件。
- LangChain：大模型应用开发框架。
- LangSmith：统一的 DevOps 平台，用于开发、协作、测试、部署和监控大模型应用程序，同时，LangSmith 是一套 Agent DevOps 规范，不仅可以用于 LangChain 应用程序，还可以用在其他框架下的应用程序中。
- LangServe：部署 LangChain 应用程序，并提供 API 管理能力，包含版本回退、升级、数据处理等。
- LangGraph：一个用于使用大模型构建有状态、多参与者应用程序的库，是 2024 年 1 月份推出的。

![image](https://github.com/user-attachments/assets/34737872-6b71-43f4-a624-812fab8812d5)

![image](https://github.com/user-attachments/assets/84ed0015-de22-4a0e-b290-5c128057e45d)

## LangChain 技术架构
接下来我们看一下目前 LangChain 整个平台技术体系，不包含 LangGraph，LangChain 框架本身包含三大模块。
- LangChain-Core：基础抽象和 LangChain 表达式语言。
- LangChain-Community：第三方集成。
- LangChain：构成应用程序认知架构的链、代理和检索策略。

![image](https://github.com/user-attachments/assets/eb667546-60fb-4ceb-8cbf-bbc51b6508b7)

下面我们介绍一下其中的重要模块。

### 模型 I/O（Model I/O）
模型 I/O 模块主要由三部分组成：格式化（Format）、预测（Predict）、解析（Parse）。顾名思议，模型 I/O 主要是和大模型打交道，前面我们提到过，大模型其实可以理解为只接受文本输入和文本输出的模型 **（ps：这句话过时了，langchain 已经支持多模态输入）**。

在把数据输入到 AI 大模型之前，不论它来源于搜索引擎、向量数据库还是第三方系统接口，都必须先对数据进行格式化，转化成大模型能理解的格式。这就是格式化部分做的事情。

![image](https://github.com/user-attachments/assets/1ab557b6-2fc8-4b6b-b118-3c8135eca2f5)

预测是指 LangChain 原生支持的丰富的 API，可以实现对各个大模型的调用。解析主要是指对大模型返回的文本内容的解析，随着多模态模型的日益成熟，相信很快就会实现对多模态模型输出结果的解析。

### Retrieval
Retrieval 可以翻译成检索、抽取，就是从各种数据源中将数据抓取过来，进行词向量化 Embedding（Word Embedding）、向量数据存储、向量数据检索的过程。你可以结合图片来理解整个过程。

![image](https://github.com/user-attachments/assets/6063a4be-fdb7-4982-b812-444a19a51aa3)

### Agents
Agents（代理）就是指实现具体任务的模块，比如从某个第三方接口获取数据，用来作为大模型的输入，那么获取数据这个模块就可以称为 XXX 代理，LangChain 本身支持多个类型的代理，当然也可以根据实际需要进行自定义。

### Chains
链条就是各种各样的顺序调用，类似于 Linux 命令里的管道。可以是文件处理链条、SQL 查询链条、搜索链条等等。LangChain 技术体系里链条主要通过 LCEL（LangChain 表达式）实现。既然是主要使用 LCEL 实现，那说明还有一部分不是使用 LCEL 实现的链条，也就是 LegacyChain，一些底层的链条，没有通过 LCEL 实现。

### Memory
内存是指模型的一些输入和输出，包含历史对话信息，可以放入缓存，提升性能，使用流程和我们软件开发里缓存的使用一样，在执行核心逻辑之前先查询缓存，如果查询到就可以直接使用，在执行完核心逻辑，返回给用户前，将内容写入缓存，方便后面使用。

![image](https://github.com/user-attachments/assets/9993c17e-90fc-4ef9-ab6d-e484f27ce752)

### Callbacks
LangChain 针对各个组件提供回调机制，包括链、模型、代理、工具等。回调的原理和普通开发语言里的回调差不多，就是在某些事件执行后唤起提前设定好的调用。LangChain 回调有两种：构造函数回调和请求回调。构造函数回调只适用于对象本身，而请求回调适用于对象本身及其所有子对象。

### LCEL
LangChain 表达式，前面我们介绍 Chains（链）的时候讲过，LCEL 是用来构建 Chains 的，我们看官方的一个例子。

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})
```

就像前面讲到的一样，这里的链和 Linux 里的管道很像，通过特殊字符 | 来连接不同组件，构成复杂链条，以实现特定的功能。

```
chain = prompt | model | output_parser
```

**每个组件的输出会作为下一个组件的输入**，直到最后一个组件执行完。当然，我们也可以通过 LCEL 将多个链关联在一起。

```
chain1 = prompt1 | model | StrOutputParser()
chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)
chain2.invoke({"person": "obama", "language": "spanish"})
```

以上就是 LangChain 技术体系里比较重要的几个核心概念，整个设计思想还是比较简单的，你只要记住两个核心思想。
- **大模型是核心控制器**，所有的操作都是围绕大模型的输入和输出在进行。
- **链**的概念，可以将一系列组件串起来进行功能叠加，这对于逻辑的抽象和组件复用是非常关键的。

理解这两个思想，玩转 LangChain 就不难了，下面举一个 RAG 的例子来说明 Agent 的使用流程。

## Agent 使用案例——RAG
就像前面讲的，大模型是基于预训练的，一般大模型训练周期 1～3 个月，因为成本过高，所以**大模型注定不可能频繁更新知识**。正是这个训练周期的问题，导致大模型掌握的知识基本上都是滞后的，GPT-4 的知识更新时间是 2023 年 12 月份，**如果我们想要大模型能够理解实时数据，或者把企业内部数据喂给大模型进行推理，我们必须进行检索增强，也就是常说的 RAG，检索增强生成**。就拿下面这个案例来说吧，我们可以通过 RAG 技术让大模型支持最新的知识索引。我们先来看一下技术流程图。

![image](https://github.com/user-attachments/assets/b0096fc1-9c56-4130-a7b5-728c32171a83)

任务一：先通过网络爬虫，爬取大量的信息，这个和搜索引擎数据爬取过程一样，当然这里不涉及 PR（Page Rank），只是纯粹的知识爬取，并向量化存储，为了保障我们有最新的数据。

任务二：用户提问时，先把问题向量化，然后在向量库里检索，将检索到的信息构建成提示，喂给大模型，大模型处理完进行输出。

整个过程涉及两个新的概念，一个叫**向量化**，一个叫**向量存储**，你先简单理解下，向量化就是将语言通过数学的方式进行表达，比如男人这个词，通过某种模型向量化后就变成了类似于下面这样的向量数据：

```
// 注意：此处只是举例，实际使用过程中男人这个词生成的向量数据取决于我们使用的 Embedding 模型。
[0.5,−1.2,0.3,−0.4,0.1,−0.8,1.7,0.6,−1.1,0.9]
```

向量存储就是将向量化后的数据存储在向量数据库里，常见的向量数据库有 Faiss、Milvus，我们会在后面的实战部分用到 Faiss。

通过任务一、二的结合，大模型就可以使用最新的知识进行推理了。当然不一定只有这一种思路，比如我们不采取预先爬取最新的数据的方式，而是实时调用搜索引擎的接口，获取最新数据，然后向量化后喂给大模型，同样也可以获取结果。在实际的项目中，要综合成本、性能等多方面因素去选择最合适的技术方案。

## 小结
这节课我们详细学习了智能体以及 LangChain 技术体系。目前看来，智能体很有可能在未来一段时间内成为 AI 发展的一个重要方向。因为大模型实际上是大厂商的游戏（除非未来开发出能够低成本训练和推理的大模型），而智能体不一样，普通玩家一样可以入局，而且现在基本上是一片蓝海。

![image](https://github.com/user-attachments/assets/cbf1fa72-8044-49d1-a4e1-83a9a08f544f)

## 思考题
从软件开发及架构的思路去看，LangChain 未来还有可能增加什么组件？你可以对比 Java 技术体系来思考一下。欢迎你把你思考后的结果分享到评论区，也欢迎你把这节课的内容分享给需要的朋友，我们下节课再见。

# 第二章：超燃实战，深度玩转 AI 模型 (4讲) 04｜本地部署：如何本地化部署开源大模型ChatGLM3-6B？
前面听我讲了这么多，相信你也很想上手试一试了。从这节课开始，我们进入一个新的章节，这部分我们会学习如何部署开源大模型 ChatGLM3-6B，本地搭建向量库并结合 LangChain 做检索增强（RAG），并且我会带你做一次微调，从头学习大模型的**部署、微调、推理**等过程。

这节课我们就来讲一下如何本地化部署 ChatGLM3-6B（后面我们简称为 6B）。讲 6B 之前我们先整体看一下目前国内外大模型的发展状况，以便我们进行技术选型。

## 大模型的选择
![image](https://github.com/user-attachments/assets/99f7d8b7-0bd5-4768-b4d8-7a78ca69234d)

![image](https://github.com/user-attachments/assets/e9e25873-1fd0-4e33-9135-ea0bdaf71892)

## 如何搞定显卡资源？
![image](https://github.com/user-attachments/assets/d62ae030-8d1f-4355-8e5b-828bc01285f2)

## ChatGLM3-6B 部署
ChatGLM-6B 目前已经发展到第 3 代 ChatGLM3-6B，除了中英文推理，还增强了数学、代码等推理能力，我记得一年前的 6B 在代码或者数学方面是比较弱的。根据目前的官方信息，在语义、数学、推理、代码、知识等不同角度的数据集上测评显示，ChatGLM3-6B-Base 在 10B 以下的基础模型中性能是最强的，除此之外，还具有 8K、32K、128K 等多个长文理解能力版本。下面我们就一步一步来安装部署 ChatGLM3-6B，你也可以在[官方文档](https://github.com/THUDM/ChatGLM3)里找到安装教程。

### 准备环境
![image](https://github.com/user-attachments/assets/5a95eafd-91da-4deb-8e06-72f7b407ff8c)

### 克隆代码
```
git clone https://github.com/THUDM/ChatGLM3
```

### 安装依赖
注意：要切换成国内 pip 源，比如阿里云，下载会快很多。
```
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
cd ChatGLM3
pip install -r requirements.txt
```

### 下载模型
```
git clone https://huggingface.co/THUDM/chatglm3-6b
```

如果 Huggingface 下载比较慢的话，也可以选择 ModelScope 进行下载。下载完将 chatglm3-6b 文件夹重新命名成 model 并放在 ChatGLM3 文件夹下，这一步非必需，只要放在一个路径下，在下一步提示的文件里，指定好模型文件路径即可。

### 命令行模式启动
打开文件 basic_demo/cli_demo.py，修改模型加载路径。
```
MODEL_PATH = os.environ.get('MODEL_PATH', '../model')
```

执行 python cli_demo.py。

![image](https://github.com/user-attachments/assets/d01d8404-8535-42df-a07a-53191b4b3d2b)

### Web 控制台模式启动
打开文件 basic_demo/web_demo_gradio.py，修改模型加载路径。
```
MODEL_PATH = os.environ.get('MODEL_PATH', '../model')
```

同时修改最后一行：
```
demo.launch(server_name="127.0.0.1", server_port=7870, inbrowser=True, share=False)
```

server_name 修改为本地 IP，并指定端口 server_port 即可。也可以设置 share=True，使用 gradio 提供的链接进行访问。

执行 python web_demo_gradio.py。

![image](https://github.com/user-attachments/assets/3b7732e3-2074-4344-bce9-07b6880c7351)

默认情况下，模型以 FP16 精度加载，大概需要 13GB 显存。如果你的电脑没有 GPU，只能通过 CPU 启动，6B 也是支持的，需要大概 32G 的内存。我们修改一下模型加载脚本。
```
model = AutoModel.from_pretrained(MODEL_PATH trust_remote_code=True).float()
```

如果你的电脑有 GPU，但是显存不够，也可以通过修改模型加载脚本，在 4-bit 量化下运行，只需要 6GB 左右的显存就可以进行流程推理。
```
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, ).quantize(4).cuda()
```

同时，官方也提供了一个全新的 web demo，支持 Chat、Tool、Code Interpreter，就在我们克隆下来的代码里，在文件夹 composite_demo 下。
```
cd composite_demo
pip install -r requirements.txt
export MODEL_PATH=../model
streamlit run main.py 或者 python -m streamlit run main.py
```

页面确实上了一个档次。

![image](https://github.com/user-attachments/assets/21be55b7-fb3d-4193-937d-7030b0b94211)

![image](https://github.com/user-attachments/assets/a2ce9e80-30e8-4551-90e8-85a6159757a7)

## 超参数介绍
ChatGLM3-6B 有 3 个参数可以设置。
1. max_length：模型的总 token 限制，包括输入和输出的 tokens。
2. temperature：模型的温度。温度只是调整单词的概率分布。它最终的宏观效果是，在较低的温度下，我们的模型更具确定性，而在较高的温度下，则不那么确定。数字越小，给出的答案越精确。
3. top_p：模型采样策略参数。每一步只从累积概率超过某个阈值 p 的最小单词集合中进行随机采样，而不考虑其他低概率的词。只关注概率分布的核心部分，忽略了尾部。

对于以下场景，官方推荐使用这样的参数进行设置：

![image](https://github.com/user-attachments/assets/9c8338db-6bcc-4de3-8078-88b308e69512)

系统设置好，我们基本就可以开始进行问答了，ChatGLM3-6B 采用了一种新的 Prompt 格式，看上去应该是模仿的 ChatGPT。下面我们介绍下这种提问格式。

## 新的 Prompt 格式
新的提示格式，主要是增加了几个角色，在对话场景中，有且仅有以下三种角色。
- system：系统信息，出现在消息的最前面，可以指定回答问题的角色。
- user：我们提的问题。
- assistant：大模型给出的回复。

在代码场景中，有且仅有 user、assistant、system、observation 四种角色。observation 是外部返回的结果，比如调用外部 API，代码执行逻辑等返回的结果，都通过 observation 返回。observation 必须放在 assistant 之后。

下面这个是官方提供的例子，基本把以上 4 种角色都解释清楚了。
```
<|system|>
Answer the following questions as best as you can. You have access to the following tools:
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]
<|user|>
今天北京的天气怎么样？
<|assistant|>
好的，让我们来查看今天的天气
<|assistant|>get_current_weather


```python
tool_call(location="beijing", unit="celsius")

<|observation|>
{"temperature": 22}
<|assistant|>
根据查询结果，今天北京的气温为 22 摄氏度。
```

![image](https://github.com/user-attachments/assets/acfea71e-0a22-4f7b-bc91-344c531ba879)

## 小结
这节课我们学习了如何部署 6B。从模型的选择到环境配置再到模型启动、推理，整体来说已经比较全面了，如果你在实际操作的过程中遇到环境问题，可以自己 Google 一下尝试去解决。毕竟每个人的环境不一样，可能会遇到各种各样的问题，主要还是 Python 相关的多一些。如果这一节课有些内容你没有看懂也不用急，先把模型部署及推理这一块熟悉一下，后面我们会逐渐深入地讲解。

![image](https://github.com/user-attachments/assets/db3b0745-8d9f-4867-a5cc-3bf134e9a76b)

## 思考题
我们知道 ChatGLM3-6B 是一个具有 62 亿参数规模的大语言模型，那你知道大模型的参数是什么意思吗？62 亿表示什么？欢迎你把你的观点分享到评论区，我们一起讨论，如果你觉得这节课的内容对你有帮助的话，也欢迎你分享给其他朋友，我们下节课再见！

# 第二章：超燃实战，深度玩转 AI 模型 (4讲) 05｜大模型微调：如何基于ChatGLM3-6B+Lora构建基本法律常识大模型？
![image](https://github.com/user-attachments/assets/09a2b866-e324-40bc-b020-41850a33dcf5)

## 如何增强模型能力？
微调是其中的一个方法，当然还有其他方式，比如外挂知识库或者通过 Agent 调用其他 API 数据源，下面我们详细介绍下这几种方式的区别。
- **微调**是一种让预先训练好的模型适应特定任务或数据集的方案，成本相对较低，这种情况下，模型会学习训练者提供的微调数据，并且具备一定的理解能力。
- **知识库**使用向量数据库或者其他数据库存储数据，为大语言模型提供信息来源外挂。
- **API** 和知识库类似，为大语言模型提供信息来源外挂。

简单理解，**微调相当于让大模型去学习一门新的学科，在回答的时候进行闭卷考试**，**知识库和 API 相当于为大模型提供了新学科的课本，回答的时候进行开卷考试**。几种模式并不冲突，我们可以同时使用几种方案来优化模型，提升内容输出能力，下面我简单介绍下几种模式的优缺点。

![image](https://github.com/user-attachments/assets/6ea7fb50-c2ad-4446-9cd4-25ec96207ffe)

注意，大模型领域所谓的性能，英文原词是 Performance，指推理效果，并非我们软件开发里所说的接口性能，比如响应时间、吞吐量等。

了解这几种模型的区别，有助于我们进行技术方案选型。在大模型实际落地过程中，我们需要先分析需求，然后确定落地方式。
1. **微调：准备数据、微调、验证、提供服务**。
2. **知识库：准备数据、构建向量库、构建智能体、提供服务**。
3. **API：准备数据、开发接口、构建智能体、提供服务**。

接下来我会通过一个真实的案例，把整个过程串起来。

## 企业真实案例
![image](https://github.com/user-attachments/assets/b356b3ad-e223-4160-89bd-a86dfe0452a7)

### 需求分析
法律小助手用来帮助员工解决日常生活中遇到的法律问题，以问答的方式进行，这种场景可以使用知识库模式，也可以使用微调模式。使用知识库模式的话，需要将数据集拆分成一条一条的知识，先放到向量库，然后通过 Agent 从向量库检索，再输入给大模型，这种方式的好处是万一我们发现数据集不足，可以随时补充，即时生效。

还有一种方式就是进行微调，因为法律知识有的时候需要一定的逻辑能力，不是纯文本检索，而微调就是这样的，通过在一定量的数据集上的训练，增加大模型法律相关的常识及思维，从而进行推理。经过分析，我们确定下来，使用微调的方式进行。接下来就是准备数据了。

### 准备数据
准备数据有很多种，可以从公共数据集下载，然后进行调整并加入私有化的知识，也可以完全自己整理，为了便于展示，我从 [GitHub](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/lawzhidao/intro.ipynb) 上面找了一个公开的数据集，下载下来的是一个 CSV 格式的文件。官方给的微调数据格式如下：
```
{"conversations": [{"role": "user", "content": "类型#裙*裙长#半身裙"}, {"role": "assistant", "content": "这款百搭时尚的仙女半身裙，整体设计非常的飘逸随性，穿上之后每个女孩子都能瞬间变成小仙女啦。料子非常的轻盈，透气性也很好，穿到夏天也很舒适。"}]}
```

conversations 是对话（提供训练示例）；user 对应的 content 是大模型的输入；assistant 对应的 content 是大模型的输出。

现在我们需要把 CSV 里的内容，转成官方要求的格式。转化代码我直接让 ChatGPT 帮我生成，用了如下提示：
```
原始数据是CSV格式，包含4列：title、question、reply、is_best，需要通过Python语言处理该CSV文件，来构建大语言模型的微调数据集，目标数据集格式是JSON的，单条数据格式为：{"conversations":[{"role":"user","content":"value1"},{"role":"assistant","content":"value2"}]}，需要将原始CSV文件里的title列填充到目标JSON文件里的value1处，原始CSV文件里的reply填充到目标JSON文件里的value1处，请注意：最终生成的不是JSON数组，而是每个JSON对象生成一行，出示示例代码。
```

ChatGPT 生成的 Python 代码：
```
import csv
import json

# CSV文件的路径
csv_file_path = 'your_csv_file.csv'
# 输出JSON文件的路径
json_file_path = 'output_json_file.json'

# 打开CSV文件，并进行处理
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
     open(json_file_path, mode='w', encoding='utf-8') as json_file:
    
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # 根据CSV文件的列名获取数据
        title = row['title']
        reply = row['reply']
        # 构造单条对话的JSON结构
        conversation_entry = {
            "conversations": [
                {"role": "user", "content": title},
                {"role": "assistant", "content": reply}
            ]
        }
        # 将单条记录以JSON格式写入文件，每条记录一行
        json_line = json.dumps(conversation_entry, ensure_ascii=False)
        json_file.write(json_line + '\n')

```

通过上面的代码，我们就可以成功将下载下来的 CSV 格式的数据集，格式化成微调所需的数据集。接下来就可以准备微调了。

### 微调
1. **安装依赖**
   我们先进入到 finetune_demo 文件夹。执行 pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple，过程中报了一个错。

   ![image](https://github.com/user-attachments/assets/87c39f01-21b5-4990-aac5-a2870fd53de2)

   原因是缺少 libopenmpi-dev 库，所以我们需要安装这个库。
   ```
   sudo apt update
   sudo apt-get install libopenmpi-dev
   ```

   紧接着安装 nltk 库。
   ```
   pip3 install nltk
   ```

   到这里，依赖包基本就安装完了。你在操作的过程中，如果遇到其他库的缺失，有可能是操作系统库也有可能是 Python 库，按照提示完成安装即可。
2. **准备数据**
   训练需要至少准备两个数据集，一个用来训练，一个用来验证。我们把“准备数据”环节格式化好的文件命名为 train.json，再准备一个相同格式的测试数据集 dev.json，在里面添加一些测试数据，几十条即可，当大模型在训练的过程中，会自动进行测试验证，输出微调效果。然后在 finetune_demo 文件夹下建立一个 data 文件夹，将这两个文件放入 data 文件夹中。
3. **修改配置**
   修改 finetune_demo/configs 下的 lora.yaml 文件，将 train_file、val_file、test_file、output_dir 定义好，记得写全路径，其他参数也可以按需修改，比如：
   - max_steps：最大训练轮数，我们填 3000。
   - save_steps：每训练多少轮保存权重，填 500。
   
   其他参数可以参考下面这张表格。

   ![image](https://github.com/user-attachments/assets/6299cffd-7784-4bf0-a133-3248a53e054c)
4. **开始微调**
   如果数据量比较少的话，比如少于 50 行，注意这一行，会报数组越界，修改小一点即可。
   ```
   eval_dataset=val_dataset.select(list(range(50))),
   ```

   微调脚本用的是 finetune_hf.py。
   - 第一个参数是训练数据集所在目录，此处值是 data。
   - 第二个参数是模型所在目录，此处值是 ./model。
   - 第三个参数是微调配置，此处值是 configs/lora.yaml。
   
   执行微调命令，记得 Python 命令使用全路径。
   ```
   /usr/bin/python3 finetune_hf.py data ../model configs/lora.yaml
   ```

   如果控制台输出下面这些内容，则说明微调开始了。

   ![image](https://github.com/user-attachments/assets/578b2f66-3a97-42ee-a245-c8d8b01e044e)

   trainable params 指的是在模型训练过程中可以被优化或更新的参数数量。在深度学习模型中，这些参数通常是网络的权重和偏置。它们是可训练的，因为在训练过程中，通过反向传播算法这些参数会根据损失函数的梯度不断更新，以减小模型输出与真实标签之间的差异。通过调整 lora.yaml 配置文件里 peft_config 下面的参数 r 来改变可训练参数的数量，r 值越大，trainable params 越大。

   我们这次微调 trainable params 为 1.9M（190 万），整个参数量是 6B（62 亿），训练比为 3%。

   ![image](https://github.com/user-attachments/assets/c9a34560-97db-4a71-b34d-eba087daabab)

   ![image](https://github.com/user-attachments/assets/ea4a02f6-ffee-4a82-bc7c-407e1759b939)
5. **验证**
   等待微调结束，就可以进行验证了，官方 demo 提供了验证脚本，执行如下命令：
   ```
   /usr/bin/python3 inference_hf.py output/checkpoint-3000/ --prompt "xxxxxxxxxxxx"
   ```
   output/checkpoint-3000 是指新生成的权重，模型启动的时候会将原模型和新权重全部加载，然后进行推理。–prompt 是输入的提示。

   下面是一组微调前后的对比问答，我们对比着来看一下。

   ![image](https://github.com/user-attachments/assets/b3af5822-473b-4733-a04a-d3b55416786d)

   我们的微调数据集中有下面这条内容：
   ```
   {"conversations": [{"role": "user", "content": "不交房电费多出由谁承担？法律法规第几条？"}, {"role": "assistant", "content": "按照约定处理，协商不成可以委托律师处理。"}]}
   ```

   部分回答效果是比较明显的。

   当然，这里只是通过快速搭建一个 demo 向你展示 Lora 微调的细节。实际生产过程中，需要考虑的事情比较多，比如训练轮数、并行数、微调效果比对等一系列问题，需要我们根据实际情况进行调整。
6. **提供服务**
   当微调完成，我们验证后得知整体效果满足一定的百分比，那我们就认为这个微调是有效的，可以对外服务，接下来就可以通过 API 组件将模型的输入输出封装成接口对外提供服务了。实际生产环境中，我们还需要考虑几件事情：
   - 模型的推理性能（效果）；
   - 模型的推理吞吐量；
   - 服务的限流，适当保护大模型集群；
   - 服务降级，当大模型服务不可用的时候，可以考虑通过修改开关，将 AI 小助手隐藏暂停使用。
   
   具体架构思路，后面第四章我会进行专门的讲解，到那时我们再详细学习具体内容。

## 小结
![image](https://github.com/user-attachments/assets/120d2d9b-9a0f-43a0-b5bd-877cc7bb3cf1)

![image](https://github.com/user-attachments/assets/72c088b3-7341-4e11-b34b-1fb0dab62836)

## 思考题
在实际微调过程中我们发现微调不是轮数越多越好，有时轮数低产生的权重会比轮数高产生的权重效果更好，你可以想想这是为什么？

# 第二章：超燃实战，深度玩转 AI 模型 (4讲) 06｜RAG实战：基于ChatGLM3-6B+LangChain+Faiss搭建企业内部知识库
![image](https://github.com/user-attachments/assets/7976c1e3-8746-4597-8fb4-7bcb8ff44232)

## Langchain-Chatchat 架构
Langchain-Chatchat 主要包含几个模块：大语言模型、Embedding 模型、分词器、向量数据库、Agent Tools、API、WebUI。

![image](https://github.com/user-attachments/assets/a082c49d-fb81-47a6-a9e2-79b979f79913)

知识库大概流程图包含**分词、向量化、存储、查询**等过程，图中不包括其他 Tools。

![image](https://github.com/user-attachments/assets/08169d97-c726-49b8-b9bb-5e1429b18656)

1～7 是文档完成向量化存储的过程，8～15 是知识库检索的过程。下面我们完整地跑一下这个 demo。

## 系统部署
1. **安装依赖**
   首先，从 GitHub 上克隆代码 https://github.com/chatchat-space/Langchain-Chatchat.git。然后安装依赖，如果之前已经安装过的，可以排除掉，通过 pip3 list 查看已经安装过的包和版本，当然如果网速允许，建议直接安装，否则可能会有依赖库冲突的问题。

   三个依赖全部安装：requirements.txt、requirements_api.txt、requirements_webui.txt。

   指定国内源速度更快：
   ```
   pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
   ```

   使用 Faiss 向量库，需要安装 faiss-gpu 依赖。
   ```
   pip3 install faiss-gpu
   ```
2. **下载模型**
   之前我们已经安装过 ChatGLM3-6B 了，本次无需再次安装，只要下载 Embedding 模型即可，我们选择 bge-large-zh-v1.5 进行下载，可以从 HuggingFace 下载，也可以从 ModelScope 下载，后者速度更快。
   ```
   git clone https://www.modelscope.cn/AI-ModelScope/bge-large-zh-v1.5.git
   git lfs pull
   ```

   git lfs 是专门用来拉取 git 上存储的大文件的，通过 git lfs pull 将模型权重拉取到本地。
3. **参数配置**
   执行以下命令，将配置文件复制一份到 config 目录，方便修改。
   ```
   python copy_config_example.py
   ```

   主要有以下几个配置文件：

   ![image](https://github.com/user-attachments/assets/3cbe057a-806a-4d62-81ca-6e77da135490)

   大部分不需要修改，我们只需要改一下 model_config.py 指定大模型和 Embedding 模型的路径即可。找到 MODEL_PATH 配置，分别修改 embed_model 里 bge-large-zh-v1.5 模型的本地目录，以及 llm_model 里 chatglm3-6b 的本地目录即可。
4. **初始化向量库数据**
   这一步主要是确认向量数据库和 Embedding 模型有没有部署好，实际生产环境中，我们可以自己创建业务向量数据库，这里通过初始化放一些示例数据进去。执行命令：
   ```
   python3 init_database.py --recreate-vs
   ```

   如果看到不断地加载示例数据，直到进度 100% 完成，就说明成功了，接下来就是一键启动。
   ```
   $ python3 startup.py -a
   ```

   当命令行出现如下提示，说明启动成功了。

   ![image](https://github.com/user-attachments/assets/769eb696-6405-4c3f-85b1-83372c016355)

   ![image](https://github.com/user-attachments/assets/63f4111c-16de-4812-87a2-027ae62512a7)

## 知识管理
打开页面，点击左侧的知识库管理，可以进行知识库的上传、删除、更新操作。

![image](https://github.com/user-attachments/assets/60f86725-5817-4406-a229-6300bee83834)

![image](https://github.com/user-attachments/assets/96e99693-0024-4b8c-9f21-7982e248d4b2)

通过上面的操作可以将文档存入所选择的向量库内，供对话时检索。我们也可以通过 API 进行知识库管理、对话及查看服务器信息等操作。

![image](https://github.com/user-attachments/assets/04ff4aa2-3715-4fdf-b46a-00d3be72b46a)

下面我们看一下知识库的效果，先对大模型进行提问。

![image](https://github.com/user-attachments/assets/115bae80-1bfd-483a-8169-8fe1f23c16b4)

选择知识库模式，并选择我们上传好的知识，再次提问。

![image](https://github.com/user-attachments/assets/7ef5f443-a9b1-466e-8090-82180bee7911)

![image](https://github.com/user-attachments/assets/cd1320a2-f426-4055-8b46-8502bcf7b876)

## Tools 使用
![image](https://github.com/user-attachments/assets/571c3153-595a-4674-b007-0488706677f9)

## 向量数据库
在聊向量数据库之前，让我们先简单了解一下“向量”这个概念。在计算机科学和数学中，向量是由一系列数字组成的数组，这些数字可以表示任何东西，从物理空间的方向和大小到商品的特性和用户偏好等。

### 相似度计算
![image](https://github.com/user-attachments/assets/b84f1ca2-fdee-4df2-9788-d9975dd100e6)

```
import numpy as np
# 定义计算余弦相似度的函数
def calculate_similarity(vector1, vector2):
    # 使用numpy库来计算余弦相似度
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
# 假设我们有一个向量代表用户偏好
user_preference = np.array([10, 5, 10, 4])
# 我们有一系列电影向量
movie_vectors = np.array([
    [8, 7, 9, 6],   # 电影A
    [9, 6, 8, 7],   # 电影B
    [10, 5, 7, 8]   # 电影C
])
# 计算并打印每部电影与用户偏好之间的相似度
for i, movie_vector in enumerate(movie_vectors):
    similarity = calculate_similarity(user_preference, movie_vector)
    print(f"电影{chr(65+i)}与用户偏好的相似度为: {similarity:.2f}")

```

我们这个示例里使用余弦相似度算法，执行这个代码块得出如下结果：
```
电影A与用户偏好的相似度为: 0.97
电影B与用户偏好的相似度为: 0.97
电影C与用户偏好的相似度为: 0.95
```

### 文本向量
![image](https://github.com/user-attachments/assets/88355ad9-346f-4872-bddf-4d1ae2e564ec)

```
import numpy as np
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
def cosine_similarity(vec_a, vec_b):
    # 计算两个向量的点积
    dot_product = np.dot(vec_a, vec_b)
    # 计算每个向量的欧几里得长度
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    # 计算余弦相似度
    return dot_product / (norm_a * norm_b)
# 获取man和boy两个词的向量
man_vector = model['man']
boy_vector = model['boy']
# 打印出这两个向量的前10个元素
print(man_vector[:10])
print(boy_vector[:10])
similarity_man_boy = cosine_similarity(man_vector, boy_vector)
print(f"男人和男孩的相似度: {similarity_man_boy}")
```

程序输出如下：
```
[ 0.32617188  0.13085938  0.03466797 -0.08300781  0.08984375 -0.04125977
 -0.19824219  0.00689697  0.14355469  0.0019455 ]
[ 0.23535156  0.16503906  0.09326172 -0.12890625  0.01599121  0.03613281
 -0.11669922 -0.07324219  0.13867188  0.01153564]
男人和男孩的相似度: 0.6824870705604553
```

![image](https://github.com/user-attachments/assets/a6fb76f7-344a-4c7c-b696-800b507fe094)

### 向量存储
![image](https://github.com/user-attachments/assets/deff6857-ed46-4d58-8eeb-510e0d71bfb1)

```
import numpy as np
import faiss

# 假设我们有一些词向量，每个向量的维度为100
dimension = 100  # 向量维度
word_vectors = np.array([
    [0.1, 0.2, ...],  # 'man'的向量
    [0.01, 0.2, ...],  # 'boy'的向量
    ...  # 更多向量
]).astype('float32')  # Faiss要求使用float32

# 创建一个用于存储向量的索引
# 这里使用的是L2距离（欧氏距离），如果你需要使用余弦相似度，需要先规范化向量
index = faiss.IndexFlatL2(dimension)

# 添加向量到索引
index.add(word_vectors)

# 假设我们想找到与'new_man'（新向量）最相似的5个向量
new_man = np.array([[0.1, 0.21, ...]]).astype('float32')  # 新的查询向量
k = 5  # 返回最相似的5个向量
D, I = index.search(new_man, k)  # D是距离的数组，I是索引的数组

# 打印出最相似的向量的索引
print(I)
```

记得先安装一下 Faiss 依赖。
```
pip install faiss-cpu  # 对于没有GPU的系统
# 或者
pip install faiss-gpu  # 对于有GPU的系统
```

![image](https://github.com/user-attachments/assets/b6b414b9-a5df-4924-b0a9-69f2b73c7eee)

## 应用场景
**知识库模式可以用在相对固定的场景做推理**，比如企业内部使用的员工小助手，包含考勤制度、薪酬制度、报销制度、法律帮助，以及产品操作手册、使用帮助等，**这类场景不需要太多的逻辑推理**，**使用知识库模式检索精确度高，并且可以随时更新**。企业实际应用过程中，除了使用大语言模型本身的基础能力外，其他的也就是在不同场景下，把各种各样的 Agent 进行堆叠，产生智能化的效果。

## 小结
![image](https://github.com/user-attachments/assets/4f683137-6b79-4756-ad0f-7ce08640da37)

本节课我们基于 6B、LangChain、Faiss 搭建了企业内部知识库系统，了解了知识库、向量库、智能体等相关的知识和应用场景，完整地体验了 RAG 场景。这节课还有很多可以动手操作的小实验，比如自定义 Tool、切换向量数据库、自定义 API，时间充足的话你可以每一个都试一下，学习效果会更好。

## 思考题
在前面知识库的例子中，我们将大模型的输出和知识库检索的全部结果进行了展示，实际应用过程中，一般不会把这两个输出的内容全部返回给用户，你可以思考一下如何设计可以更加人性化的返回结果。

# 第二章：超燃实战，深度玩转 AI 模型 (4讲) 07｜大模型API封装：自建大模型如何对外服务？
![image](https://github.com/user-attachments/assets/4e30e456-cb49-4b4e-8205-b0fc8d948e0b)

## 接口封装
提供 Web API 服务需要两个技术组件：Uvicorn 和 FastAPI。

Uvicorn 作为 Web 服务器，类似 Tomcat，但是比 Tomcat 轻很多。允许异步处理 HTTP 请求，所以非常适合处理并发请求。基于 uvloop 和 httptools，所以具备非常高的性能，适合高并发请求的现代 Web 应用。

FastAPI 作为 API 框架，和 SpringBoot 差不多，同样比 SpringBoot 轻很多，只是形式上类似于 SpringBoot 的角色。结合使用 Uvicorn 和 FastAPI，你可以构建一个高性能、易于扩展的异步 Web 应用程序或 API。Uvicorn 作为服务器运行你的 FastAPI 应用，可以提供优异的并发处理能力，而 FastAPI 则让你的应用开发得更快、更简单、更安全。

接下来我们一步一步讲解。首先，安装所需要的依赖包。

### 安装依赖
```
pip install fastapi
pip install uvicorn
```

### 代码分层
简单来看，创建 api.py，写入以下代码，就可以定义一个接口。
```
import uvicorn
from fastapi import FastAPI

# 创建API应用
app = FastAPI()

@app.get("/")
async def root():
  return {"message": "Hello World"}

if __name__ == '__main__':
  # 启动服务
  uvicorn.run(app, host='0.0.0.0', port=6006, log_level="info", workers=1)
```

执行：
```
python api.py
```

结果：

![image](https://github.com/user-attachments/assets/b48c9ad2-f2bc-4813-86f6-c538c63e4d3d)


实际开发过程中，接口输入可能是多个字段，和 Java 接口一样，需要定义一个 Request 实体类来承接 HTTP 请求参数，Python 里使用 Pydantic 模型来定义数据结构，Pydantic 是一个数据验证和设置管理的库，它利用 Python 类型提示来进行数据验证。类似 Java 里的 Validation，下面这段代码你应该并不陌生。
```
import javax.validation.constraints.Min;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;

public class Product {

    @NotNull
    @Size(min = 2, max = 30)
    private String name;

    @NotNull
    @Min(0)
    private Float price;

    // 构造器、getter 和 setter 省略
}
```

对应的 Python 实现就是这样的：
```
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatMessage(BaseModel):
    history: List[Message]
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float = Field(default=1.0)

@app.post("/v1/chat/completions")
async def create_chat_response(message: ChatMessage):
    return {"message": "Hello World"}

if __name__ == '__main__':
  uvicorn.run(app, host='0.0.0.0', port=6006, log_level="info", workers=1)
```

这里引入了一个 BaseModel 类，类似于 Java 里的 Object 类，但是又不完全是 Object，Object 是所有 Java 类的基类，Java 中所有类会默认集成 Object 类的公共方法，比如 toString()、equals()、hashcode() 等，而 BaseModel 是为了数据验证和管理而设计的。当你创建一个继承自 BaseModel 的类时，比如上面的 ChatSession 和 Message 类，将自动获得数据验证、序列化和反序列化的功能。

另外，我们实际开发过程中，也不可能把所有 API 的定义和 Pydantic 类放在最外层，按照 Java 工程化的最佳实践，Web 应用我们一般会进行分层，比如 controller、service、model、tool 等，Python 工程化的时候，为了方便管理代码，也会进行分层，一个典型的代码结构如下：
```
project_name/
│
├── app/                         # 主应用目录
│   ├── main.py                  # FastAPI 应用入口
│   └── controller/              # API 特定逻辑
│       └── chat.py
│   └── common/                  # 通用API组件
│       └── errors.py            # 错误处理和自定义异常
│
├── services/                    # 服务层目录
│   ├── chat_service.py          # 聊天服务相关逻辑
│
├── schemas/                     # Pydantic 模型（请求和响应模式）
│   ├── chat_schema.py           # 聊天数据模式
│
├── database/                    # 数据库连接和会话管理
│   ├── session.py               # 数据库会话配置
│   └── engine.py                # 数据库引擎配置
│
├── tools/                       # 工具和实用程序目录
│   ├── data_migration.py        # 数据迁移工具
│
├── tests/                       # 测试目录
│   ├── conftest.py              # 测试配置和夹具
│   ├── test_services/           # 服务层测试
│   │   ├── test_chat_service.py
│   └── test_controller/                
│       ├── test_chat_controller.py
│
├── requirements.txt             # 项目依赖文件
└── setup.py                     # 安装、打包、分发配置文件
```

FastAPI 的 include_router 方法就是用来将不同的路由集成到主应用中的，有助于组织和分离代码，特别是在构建大型工程化应用时，非常好用。你可以看一下修改后的代码。
```
import uvicorn as uvicorn
from fastapi import FastAPI
from controller.chat_controller import chat_router as chat_router
app = FastAPI()
app.include_router(chat_router, prefix="/chat", tags=["chat"])
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=6006, log_level="info", workers=1)
```

chat_controller.py：
```
from fastapi import APIRouter
from service.chat_service import ChatService
from schema.chat_schema import ChatMessage, MessageDisplay
chat_router = APIRouter()
chat_service = ChatService()

@chat_router.post("/new/message/")
def post_message(message: ChatMessage):
    return chat_service.post_message(message)

@chat_router.get("/get/messages/")
def get_messages():
    return chat_service.get_messages()
```

chat_service.py：
```
from schema.chat_schema import ChatMessage

class ChatService:
    def post_message(self, message: ChatMessage) :
        print(message.prompt)
        return {"message": "post message"}
    def get_messages(self):
        return {"message": "get message"}
```

参数类定义如下：
```
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str

class ChatMessage(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
```

我们可以在 chat_service 里进行详细地业务逻辑处理，到这里基本就和 Java 里一样了。下面是一段简单的测试代码：
```
import json
import requests

url = 'http://localhost:6006/chat/new/message/'
data = {
    'prompt': 'hello',
    'max_tokens': 1000
}

response = requests.post(url, data=json.dumps(data))
print(response.text)

url2 = 'http://localhost:6006/chat/get/messages/'
response = requests.get(url2)
print(response.text)
```
```
{"message":"post message"}
{"message":"get message"}
```

关于 FastAPI 的使用，你可以参考这个[教程](https://fastapi.tiangolo.com/zh/tutorial/)。工程化代码结构搞定，我们就可以封装大模型的接口了。

### 大模型接口封装
不同的大模型对应的对话接口不一样，下面的示例代码基于 ChatGLM3-6B。我们在 service 层进行模型对话的封装。你可以看一下示例代码。
```
from datetime import datetime
import model_manager
from schema.chat_schema import ChatMessage

class ChatService:
    def post_message(self, message: ChatMessage):
        print(message.prompt)
        model = model_manager.ModelManager.get_model()
        tokenizer = model_manager.ModelManager.get_tokenizer()
        response, history = model.chat(
            tokenizer,
            message.prompt,
            history=message.histroy,
            max_length=message.max_tokens,
            top_p=message.top_p,
            temperature=message.temperature
        )
        now = datetime.datetime.now()  # 获取当前时间
        time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
        answer = {
            "response": response,
            "history": history,
            "status": 200,
            "time": time
        }
        log = "[" + time + "] " + '", prompt:"' + message.prompt + '", response:"' + repr(response) + '"'
        print(log)
        return answer
    def get_messages(self):
        return {"message": "get message"}
```

定义一个 ModelManager 类进行大模型的懒加载。
```
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelManager:
    _model = None
    _tokenizer = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            _model = AutoModelForCausalLM.from_pretrained("chatglm3-6b", trust_remote_code=True).half().cuda().eval()
        return _model

        @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b", trust_remote_code=True)
        return _tokenizer
```

model.chat() 是 6B 暴露的对话接口，通过对 model.chat() 的封装就可以实现基本的对话接口了，这个接口一次性输出大模型返回的内容，而我们在使用大模型产品的时候，比如 ChatGPT 或者文心一言，会发现大模型是一个字一个字返回的，那是什么原因呢？那种模式叫**流式输出**。

### 流式输出
流式输出使用另一个接口：model.stream_chat，有几种模式，像一个字一个字输出，比如：
```
我
是
中
国
人
```

或者每次输出当前已经输出的全部，比如：
```
我
我是
我是中
我是中国
我是中国人
```

当然也有每次吐出 2 个字的，实际生产过程中可以根据产品交互设计自行修改逻辑。我们看一个简单的代码片段，通过 stream 变量来控制是否是流式输出。
```
if stream:
    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield json.dumps(
            {"text": token, "message_id": message_id},
            ensure_ascii=False)
else:
    answer = ""
    async for token in callback.aiter():
        answer += token
    yield json.dumps(
        {"text": answer, "message_id": message_id},
        ensure_ascii=False)
await task
```

我们输入“你好”，当 stream=true 时，接口输出是这样的：

![image](https://github.com/user-attachments/assets/06fb7b74-73ea-4399-87a1-7668feabac6c)

当 stream=false 时，接口返回如下：
```
data: {"text": "你好！我是人工智能助手，很高兴为您服务。请问有什么问题我可以帮您解答吗？", "message_id": "741a630ac3d64fd5b1832cc0bae6bb68"}
```

到这里，大模型的 API 基本就封装好了，接下来我们看下如何调用。

## 接口调用
在实际工程化过程中，我们一般会把 AI 相关的逻辑，包括大模型 API 的封装放在 Python 应用中，上层应用一般通过其他语言实现，比如 Java、C#、Go 等，这里我简单举一个 Java 版本的调用例子。非流式输出就是普通的 HTTP 请求，我们就不展示了，重点看下流式输出怎么进行调用，主要分两步，都是流式的。

1. Java 调用 Python 接口：主要用到了 okhttp3 框架，需要组装参数、发起流式请求，事件监听处理三步。
   ```  
    @ApiOperation(value = "流式发送对话消息")
    @PostMapping(value = "sendMessage")
    public void sendMessage(@RequestBody ChatRequest request, HttpServletResponse response) {
      try {
        JSONObject body = new JSONObject();
        body.put("model", request.getModel());
        body.put("stream", true);
        JSONArray messages = new JSONArray();
        JSONObject query = new JSONObject();
        query.put("role", "user");
        query.put("content", request.getQuery());
        messages.add(query);
        body.put("messages", messages);
        EsListener eventSourceListener = new EsListener(request, response);
    
    
    
        RequestBody formBody = RequestBody.create(body, MediaType.parse("application/json"));
        Request.Builder requestBuilder = new Request.Builder();
    
        Request request2 = requestBuilder.url(URL).post(formBody).build();
        EventSource.Factory factory = EventSources.createFactory(OkHttpUtil.getInstance());

        factory.newEventSource(request2, eventSourceListener);
        eventSourceListener.getCountDownLatch().await();
      } catch (Exception e) {
        log.error("流式调用异常", e);
      }
    }
   ```
   
   EsListener 继承自 EventSourceListener，在 Request 请求的过程中不断触发 EsListener 的 onEvent 方法，然后将数据写回前端。
   ```
   @Override
    public void onEvent(EventSource eventSource, String id, String type, String data) {
      try {
        output.append(data);
        if ("finish".equals(type)) {
        }
        if ("error".equals(type)) {
        }
    
        // 开始处理data，此处只展示基本操作
        // 开发过程中具体逻辑可自行扩展
        if (response != null) {
          response.getWriter().write(data);
          response.getWriter().flush();
        }
      } catch (Exception e) {
        log.error("事件处理异常", e);
      }
    }
   ```
2. 前端调用 Java 接口：使用 JS 原生 EventSource 的 API 就可以。
   ```
   <script>
    let eventData = '';
    const eventSource = new EventSource('http://localhost:8888/sendMessage');
    eventSource.onmessage = function(event) {
        // 累加接收到的事件数据
        eventData += event.data;
    };
   </script>
   ```

到这一步，大模型 API 从封装到调用就基本完成了，你可以把整个链路都串起来跑一跑，体验下效果。实际工程化的过程中，还会遇到其他问题，比如 API 的鉴权（指 Java->Python）、跨域问题、API 限流问题（大模型的吞吐量有限），我们会在后面的课程中讲解。

## 小结
我们这节课学的内容是自建大模型服务不可缺少的一步，整体来说不算难，唯一可能难一点的就是要使用 Python 语言，因为在使用 FastAPI 的过程中，会有大量的异步操作，和 Java 的处理方式有点差异，需要注意下。

这节课学完，我们基本上把企业内部构建大模型的过程全部讲完了，你自己构建的大模型基本可以对外提供服务了。**如果在生产环境使用，一定要注意做好降级准备**，因为有很多不确定性，比如模型的吞吐量（TPS）评估是否准确，模型会不会出现意想不到的输出等等，一旦出现问题随时降级。

## 思考题
前面我们提到，大模型相关的 API 封装在 Python 应用中，对用户提供服务的时候，会再套一层 Java 应用，你可以想一下为什么要这么设计？

![image](https://github.com/user-attachments/assets/f6e28bbe-9ff0-49cb-8ecd-6862dd373adb)

# 第三章：打入核心，挑战底层技术原理 (8讲) 08｜关于机器学习，你需要了解的基本概念（一）
![image](https://github.com/user-attachments/assets/7a4cc8a3-ce11-4959-b191-035994db071d)

## 机器学习
![image](https://github.com/user-attachments/assets/acbbd5c4-04dd-400f-b9ff-b6265fb8cea6)

我们使用 Python 和一个流行的机器学习库，如 scikit-learn，来实现这一目标，代码可能会是这样的：
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# 加载数据集
data = pd.read_csv("housing_data.csv")  # 假设这是我们的房屋数据

# 准备数据
X = data[['面积', '卧室数量', '地理位置']]  # 特征
y = data['售价']  # 目标变量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 使用模型进行预测
predictions = model.predict(X_test)

# 现在，`predictions`包含了我们模型预测的房价
```

![image](https://github.com/user-attachments/assets/a6f30663-fd15-403b-9615-3e2d598c36ab)

![image](https://github.com/user-attachments/assets/ad391d1b-6012-495c-a3b5-b181eadecefe)

## 深度学习
![image](https://github.com/user-attachments/assets/1616a9b7-5546-4e28-aeb3-4a9d68f0cdc2)

![image](https://github.com/user-attachments/assets/4234dc94-a7b6-49d9-a462-4311eb03f704)

## 机器学习过程
了解了机器学习的基本概念后，我们来看看机器学习一般有哪些过程，从工程化角度，我梳理了 10 个步骤，从问题提出到模型上线及运维，算是比较全面的了。

![image](https://github.com/user-attachments/assets/9beccdd1-03bd-48f4-90bd-a537dbda2e77)

![image](https://github.com/user-attachments/assets/353a9908-9570-4e97-80fe-78eef0f9acb3)

![image](https://github.com/user-attachments/assets/9bd07de6-fe55-4bd9-acd9-caa3d13e1b46)

![image](https://github.com/user-attachments/assets/7dc9f6a2-5d74-46b1-8304-e1247f84ab22)

## 经典算法
### 线性回归
![image](https://github.com/user-attachments/assets/0a7bd3ea-8e81-41c7-909e-79deed4d4d2d)

下面是一个使用 sklearn 库进行线性回归的简单例子。假设我们有以下面积和价格的数据：
```
面积（平方米）: [35, 45, 40, 60, 65]
价格（万元）: [30, 40, 35, 60, 65]
```

我们将使用这些数据来拟合一个线性回归模型，并预测面积为 50 平方米的房屋的价格。
```
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 定义数据
X = np.array([35, 45, 40, 60, 65]).reshape(-1, 1) # 面积
y = np.array([30, 40, 35, 60, 65]) # 价格

# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)

# 预测面积为50平方米的房屋价格
predict_area = np.array([50]).reshape(-1, 1)
predicted_price = model.predict(predict_area)

print(f"预测的房价为：{predicted_price[0]:.2f}万美元")

# 绘制数据点和拟合直线
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('房价预测')
plt.xlabel('面积（平方米）')
plt.ylabel('价格（万元）')
plt.show()
```

![image](https://github.com/user-attachments/assets/454140cd-11eb-4652-827e-e45e323862b5)

### 逻辑回归
![image](https://github.com/user-attachments/assets/99d1e8b1-9a7a-468c-9ac2-0751f9fd66d8)

```
# 导入库
from sklearn.linear_model import LogisticRegression
import numpy as np
# 准备数据
X = np.array([[10], [20], [30], [40], [50]]) # 学习时间
y = np.array([0, 0, 1, 1, 1]) # 通过考试与否
# 创建逻辑回归模型并训练
model = LogisticRegression()
model.fit(X, y)
# 预测学习时间为25小时的学生通过考试的概率
prediction_probability = model.predict_proba([[25]])
prediction = model.predict([[25]])
print(f"通过考试的概率为：{prediction_probability[0][1]:.2f}")
print(f"预测分类：{'通过' if prediction[0] == 1 else '未通过'}")
```

![image](https://github.com/user-attachments/assets/58eb9ed3-9d11-466d-8b63-83b19afd4778)

## 小结
![image](https://github.com/user-attachments/assets/5a167ded-397d-496b-bf5c-949861270f99)

## 思考题
这节课我没有给出各个算法的缺点，你可以思考一下，线性回归的局限是什么，逻辑回归的局限又是什么？

# 第三章：打入核心，挑战底层技术原理 (8讲) 09｜关于机器学习，你需要了解的基本概念（二）
上一节课我们了解了机器学习的基本概念，学习了线性回归和逻辑回归，相信你对机器学习有了初步理解，这节课我们继续讲解机器学习的经典算法，先从决策树开始。

## 经典算法
### 决策树
![image](https://github.com/user-attachments/assets/80958f28-c9c5-423b-b440-6caf051fab04)

我们看一下使用 sklearn 库提供的决策树算法和模型的示例代码。
```
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# 创建数据集
X = np.array([
    [0, 2, 0],  # 晴天，高温，无风
    [1, 1, 1],  # 阴天，中温，微风
    [2, 0, 2],  # 雨天，低温，强风
    # ... 添加更多样本以增加模型的准确性
])
y = np.array([0, 1, 2])  # 分别对应去野餐、去博物馆、在家看书

# 初始化决策树模型，设置最大深度为5
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# 训练模型
clf.fit(X, y)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=["天气状况", "温度", "风速"], class_names=["去野餐", "去博物馆", "在家看书"], rounded=True, fontsize=12)
plt.show()
```

程序运行结果如下：

![image](https://github.com/user-attachments/assets/847ad9a5-9ff5-4b30-92a3-f6374c7720b1)

![image](https://github.com/user-attachments/assets/d9d23227-c370-445c-b178-74f1bb95dcc4)

### 随机森林
![image](https://github.com/user-attachments/assets/572dbbb2-3011-4a69-9eb0-0da45cdeff5a)

![image](https://github.com/user-attachments/assets/8e97c4aa-1364-4d1d-8944-842c170c5247)

简易代码如下：
```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=3, random_state=42) # 使用3棵树以便于可视化
rf.fit(X, y)

# 绘制随机森林中的决策树
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5), dpi=100)
for index in range(0, 3):
    plot_tree(rf.estimators_[index], 
              feature_names=iris.feature_names, 
              class_names=iris.target_names, 
              filled=True, 
              ax=axes[index])

    axes[index].set_title(f'Tree {index + 1}')

plt.tight_layout()
plt.show()
```

程序运行结果如下：

![image](https://github.com/user-attachments/assets/710c339d-010d-4430-b5fc-115aeb3010ee)

![image](https://github.com/user-attachments/assets/d503c6d6-be06-44f7-8932-c78a53a6ea30)

### 支持向量机
![image](https://github.com/user-attachments/assets/3a4ca6b6-d1ad-4a74-9e43-48e3501f56c0)

我们简单看下示例代码：
```
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 生成模拟数据
X, y = datasets.make_blobs(n_samples=50, centers=2, random_state=6)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建 SVM 模型
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 绘制数据点和分类边界
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格点
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和间隔
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("支持向量机分类示例")
plt.xlabel("特征1")
plt.ylabel("特征2")
plt.show()
```

程序运行结果如下：

![image](https://github.com/user-attachments/assets/898d6467-edb9-4c22-98b1-f4e3a3f34253)

### 神经网络
![image](https://github.com/user-attachments/assets/fc3107e4-350a-48fa-a124-5575fa68acfa)

我们来看一个示例。
```
import matplotlib.pyplot as plt

# 创建一个简单的神经网络图，并调整文字标签的位置
def plot_neural_network_adjusted():
    fig, ax = plt.subplots(figsize=(10, 6))  # 创建绘图对象

    # 输入层、隐藏层、输出层的神经元数量
    input_neurons = 3
    hidden_neurons = 4
    output_neurons = 2

    # 绘制神经元
    layer_names = ['输入层', '隐藏层', '输出层']
    for layer, neurons in enumerate([input_neurons, hidden_neurons, output_neurons]):
        for neuron in range(neurons):
            circle = plt.Circle((layer*2, neuron*1.5 - neurons*0.75 + 0.75), 0.5, color='skyblue', ec='black', lw=1.5, zorder=4)
            ax.add_artist(circle)

    # 绘制连接线
    for input_neuron in range(input_neurons):
        for hidden_neuron in range(hidden_neurons):
            line = plt.Line2D([0*2, 1*2], [input_neuron*1.5 - input_neurons*0.75 + 0.75, hidden_neuron*1.5 - hidden_neurons*0.75 + 0.75], c='gray', lw=1, zorder=1)
            ax.add_artist(line)
    for hidden_neuron in range(hidden_neurons):
        for output_neuron in range(output_neurons):
            line = plt.Line2D([1*2, 2*2], [hidden_neuron*1.5 - hidden_neurons*0.75 + 0.75, output_neuron*1.5 - output_neurons*0.75 + 0.75], c='gray', lw=1, zorder=1)
            ax.add_artist(line)

    # 设置图参数
    ax.set_xlim(-1, 5)
    ax.set_ylim(-2, max(input_neurons, hidden_neurons, output_neurons)*1.5)
    plt.axis('off')  # 不显示坐标轴

    # 调整层名称的绘制位置，确保不被遮挡
    for i, name in enumerate(layer_names):
        plt.text(i*2, max(input_neurons, hidden_neurons, output_neurons)*0.75 + 1, name, horizontalalignment='center', fontsize=14, zorder=5)

    plt.title("简单神经网络图解", fontsize=16)
    return fig

fig = plot_neural_network_adjusted()
plt.show()
```

![image](https://github.com/user-attachments/assets/7d3a3a0b-1209-4a60-a384-c473ca186acb)

![image](https://github.com/user-attachments/assets/8cecdadc-72f3-4a22-869e-590b6375b45d)

## 小结
这节课我们只是进行概念性的介绍，目的在于认识基本的机器学习算法，对于初学者而言，理解这些基本的概念至关重要，通过这两节课的学习，我们了解了一些比较常见的机器学习算法，通过一些简单的代码示例，理解了各个算法的实际应用案例。虽然有些抽象，但是慢慢看还是能够理解的。建议你深入研究下 sklearn 这个库，里面包含这类场景的机器学习算法，然后自己动手敲一下这些示例代码感受一下。

## 思考题
实际上我们上面讲的神经网络之所以强大，就是因为有激活函数，使神经网络呈现为非线性的，那么你可以思考一下，为什么激活函数可以使神经网络呈非线性？如果没有激活函数，神经网络会出现什么问题？

# 第三章：打入核心，挑战底层技术原理 (8讲) 10｜经典算法之RNN：开发人员绕不开的循环神经网络
![image](https://github.com/user-attachments/assets/7015f775-6506-466d-a932-ec9e3fbe332a)

## 循环神经网络
![image](https://github.com/user-attachments/assets/80d02254-e098-4bf7-8242-401978704505)

## 基本结构与原理
![image](https://github.com/user-attachments/assets/b96722c8-9a3a-472f-af5e-2687628463ee)

![image](https://github.com/user-attachments/assets/a2e97ab2-36a0-418d-98b1-4229a303aec9)

![image](https://github.com/user-attachments/assets/415cf60d-3fda-4387-b225-afc1fff180d2)

## 关键挑战
RNN 通过当前的隐藏状态来记住序列中之前的信息。这种记忆一般是短期的，因为随着时间步的增加，早期输入对当前状态的影响会逐渐减弱，在标准 RNN 中，尤其当遇到梯度消失情况时，就会遇到短期记忆的问题，几乎无法更新权重。

### 梯度消失
我们先来看下什么是梯度？梯度是指函数在某一点的斜率，在深度学习中，该函数一般指具有多个变量的损失函数，变量就是模型的权重。损失函数衡量的是模型预测与实际数据之间的差异，一般情况下，我们要尽可能地让损失函数的值最小。如何找到这个最小值呢？需要进行梯度下降，也就是说，**我们要不断调整参数（权重），使损失函数的值降到最小，这个过程就是梯度下降**。

为什么会产生梯度消失呢？一般有两个原因。
1. **深层网络中的连乘效应**：在深层网络中，梯度是通过链式法则进行反向传播的。如果每一层的梯度都小于 1，那么随着层数的增加，这些小于 1 的值会连乘在一起，导致最终的梯度非常小。
2. **激活函数的选择**：使用某些激活函数，如 tanh，函数的取值范围是 -1～1，小于 1 的数进行连乘，也会快速降低梯度值。

这里我解释下反向传播，在深度学习中，**训练神经网络涉及两个主要的传播阶段：前向传播和反向传播**。在前向传播阶段，输入数据从网络的输入层开始，逐层向前传递至输出层。每一层都会对其输入进行计算，如加权求和，然后应用激活函数等，并将计算结果传递给下一层，直到最终产生输出。这个过程的目标是根据当前的网络参数、权重和偏置等得到预测输出。

一旦在输出层得到了预测输出，就会计算损失函数，即预测输出与实际目标输出之间的差异。接下来，这个损失会被用来计算损失函数相对于网络中每个参数的梯度，这是通过链式法则实现的。这个计算过程从输出层开始，沿着网络向后，即向输入层的方向，逐层进行，这就是“反向传播”的由来。

这些梯度表示了为了减少损失，各个参数需要如何调整。最后，这些梯度会用来更新网络的参数，通常是通过梯度下降或其变体算法实现。而**在反向传播过程中，每到达一层，都会触发激活函数**，这就是我们上面说的第 2 点原因。

由此可见，如果要解决梯度消失的问题，我们就从这两个原因入手。
1. [长短期记忆（LSTM）](https://time.geekbang.org/course/detail/100077201-420627?utm_campaign=geektime_search&utm_content=geektime_search&utm_medium=geektime_search&utm_source=geektime_search&utm_term=geektime_search)和[门控循环单元（GRU）](https://time.geekbang.org/course/detail/100077201-418603?utm_campaign=geektime_search&utm_content=geektime_search&utm_medium=geektime_search&utm_source=geektime_search&utm_term=geektime_search)是专门为了避免梯度消失问题而设计的。它们通过引入门控机制来调节信息的流动，保留长期依赖信息，从而避免梯度在反向传播过程中消失。
2. 使用 ReLU 及其变体激活函数，在正区间内的梯度保持恒定，不会随着输入的增加而减少到 0，这有助于减轻梯度消失问题。

### 梯度爆炸
与梯度消失相对的问题是**梯度爆炸，当模型的梯度在反向传播过程中变得非常大，以至于更新后的权重偏离了最优值，导致模型无法收敛，甚至发散**。

通常梯度爆炸发生原因有三个。
1. **深层网络的连乘效应**：在深层网络中，梯度是通过链式法则进行反向传播的。如果每一层的梯度都大于 1，那么随着层数的增加，这些大于 1 的值会连乘在一起，导致最终的梯度非常大。
2. **权重初始化不当**：如果网络的权重初始化得太大，那么在前向传播过程中信号的大小会迅速增加，同样，反向传播时梯度也会迅速增加。
3. **使用不恰当的激活函数**：某些激活函数（如 ReLU）在正区间的梯度为常数。如果网络架构设计不当，使用这些激活函数也可能导致梯度爆炸。

梯度爆炸和梯度消失基本相反，解决方法一样，要么**使用长短期记忆和门控循环单元调整网络结构**，要么**替换激活函数**，还有一种办法是**进行梯度裁剪**，梯度裁剪意思是在训练过程中，通过限制梯度的最小 / 大值来防止梯度消失 / 爆炸，间接地保持梯度的稳定性。

### 长短期记忆（LSTM）
![image](https://github.com/user-attachments/assets/5b3ec68b-2435-45ba-8e67-3d55400ca47f)

通过这些机制，LSTM 能够在处理序列数据时，有效地保留长期的依赖信息，就像是记住故事中的关键情节和角色，同时避免了标准 RNN 中常见的梯度消失问题。这使得 LSTM 特别适用于需要理解整个序列背景的任务，比如语言翻译，需要理解整个句子的含义，或者股票价格预测，需要考虑长期的价格变化趋势。

## RNN 实际应用场景
### 文本生成
文本生成是 RNN 的一个典型应用，通过学习大量的文本数据，RNN 能够生成具有相似风格的文本。我们看一个简单的文本生成模型的代码示例。
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# 数据预处理
text = "Here is some sample text to demonstrate text generation with RNN. This is a simple example."
tokens = text.lower().split()
tokenizer = {word: i + 1 for i, word in enumerate(set(tokens))}
total_words = len(tokenizer) + 1

# 创建输入序列
sequences = []
for line in text.split('.'):
    token_list = [tokenizer[word] for word in line.lower().split() if word in tokenizer]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in sequences])
sequences = [torch.tensor(seq) for seq in sequences]
sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.x = sequences[:, :-1]
        self.y = sequences[:, -1]
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TextDataset(sequences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 构建模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

model = RNNModel(total_words, 64, 20)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
for epoch in range(100):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 文本生成
def generate_text(seed_text, next_words, model, max_sequence_len):
    model.eval()
    for _ in range(next_words):
        token_list = [tokenizer[word] for word in seed_text.lower().split() if word in tokenizer]
        token_list = torch.tensor(token_list).unsqueeze(0)
        token_list = nn.functional.pad(token_list, (max_sequence_len - 1 - token_list.size(1), 0), 'constant', 0)
        with torch.no_grad():
            predicted = model(token_list)
            predicted = torch.argmax(predicted, dim=-1).item()
        output_word = ""
        for word, index in tokenizer.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

print(generate_text("Here is", 4, model, max_sequence_len))
```

核心过程我在代码中注释过了，这只是一个简单的代码示例，实际使用的时候，你可能需要使用更大的文本数据集来训练模型，调整模型架构，比如增加层数、调整 LSTM 单元数量、超参数等，以及实施更复杂的数据预处理和文本生成策略，以达到更好的生成效果。

## 小结
这节课内容实际是有点难度的，我们通过简单的例子学习了 RNN 的基本概念，然后通过敲代码进行练习。**RNN 的优势在于它的记忆能力，通过隐藏层循环结构捕捉序列的长期依赖关系，特别适合用于文本生成、语音识别等领域**。同时，RNN 也有局限性，比如梯度消失、梯度爆炸等，而引入 LSTM 可以一定程度上解决这些问题。

## 思考题
既然引入 LSTM 可以解决一系列问题，那目前主流的大语言模型为什么没有使用 RNN 架构呢？请你说说你的理解。

# 第三章：打入核心，挑战底层技术原理 (8讲) 11｜关于自然语言处理，你需要了解的基本概念
![image](https://github.com/user-attachments/assets/40a63922-651d-48a6-b905-fbd787562bc4)

## 

# 第三章：打入核心，挑战底层技术原理 (8讲) 12｜深入理解Word2Vec：解开词向量生成的奥秘
# 第三章：打入核心，挑战底层技术原理 (8讲) 13｜深入理解Seq2Seq：让我们看看语言翻译是怎么来的
# 第三章：打入核心，挑战底层技术原理 (8讲) 14｜Transformer技术原理：为什么说Transformer是大模型架构的基石？（上）
# 第三章：打入核心，挑战底层技术原理 (8讲) 15｜Transformer技术原理：为什么说Transformer是大模型架构的基石？（下）

# 第四章：终极玩法，从0到1构建大模型 (10讲) 16｜从零开始，构建一个具有100M参数规模的Transformer模型
# 第四章：终极玩法，从0到1构建大模型 (10讲) 17｜模型解剖：探究模型内部到底是什么？
# 第四章：终极玩法，从0到1构建大模型 (10讲) 18｜大模型预训练：Pre-Training如何让模型变聪明？
# 第四章：终极玩法，从0到1构建大模型 (10讲) 19｜深入理解DeepSpeed，提高大模型训练效率
# 第四章：终极玩法，从0到1构建大模型 (10讲) 20｜人类意图对齐，让模型拥有更高的情商
# 第四章：终极玩法，从0到1构建大模型 (10讲) 21｜模型测评：如何评估大模型的表现？
# 第四章：终极玩法，从0到1构建大模型 (10讲) 22｜模型轻量化：如何让模型运行在低配置设备上？
# 第四章：终极玩法，从0到1构建大模型 (10讲) 23｜模型核心技术指标：如何提高上下文长度？
# 第四章：终极玩法，从0到1构建大模型 (10讲) 24｜架构设计（上）：企业如何设计大模型应用架构？
# 第四章：终极玩法，从0到1构建大模型 (10讲) 25｜架构设计（下）：企业如何搭建 AI 中台？

# 第五章：热点速递，AI行业发展趋势解读 (5讲) 26｜为什么说Mamba是Transformer的最强挑战者？
在过去的几年里，Transformer 模型在自然语言处理领域占据了主导地位。自从 2017 年谷歌提出 Transformer 以来，BERT、GPT-3 等基于 Transformer 的模型取得了巨大的成功。

然而技术的进步从未停止，最近出现了一种新型模型——**Mamba，被认为是 Transformer 的最强挑战者**。那么，Mamba 凭什么能与 Transformer 一较高下呢？这节课我就来带你看看 Mamba 的过人之处。

## Transformer 的局限
![image](https://github.com/user-attachments/assets/5f41021d-784a-44fa-88d4-d499c9ac601c)

![image](https://github.com/user-attachments/assets/57330639-7d31-46e7-a536-c0b88e6e2bf5)

## Mamba 的优势
1. 基于 S4 架构
   ![image](https://github.com/user-attachments/assets/e2d6e61a-5209-4d4b-9fed-4c8333482c25)
2. 高效性
   ![image](https://github.com/user-attachments/assets/4fc9dd1f-dbd3-451c-b8aa-63cd3c40771c)
3. 适应性
   ![image](https://github.com/user-attachments/assets/87591846-00a7-4388-881a-841167702272)
4. 内存利用
   ![image](https://github.com/user-attachments/assets/664c0883-eec1-4a79-94e3-3fa2a1df56d0)
5. 训练速度
   ![image](https://github.com/user-attachments/assets/982106c0-f590-429d-a2b3-ba6b4184ebd2)
6. 性能表现
   ![image](https://github.com/user-attachments/assets/cf4962ea-bc39-497a-bb53-b4a392b53cb6)

## Mamba 架构之状态空间模型（SSM）
### 模型结构
![image](https://github.com/user-attachments/assets/b19f39db-98f5-43ad-a6c6-a1048dc8e4ca)

![image](https://github.com/user-attachments/assets/4fa3fdf9-f38d-4557-b9e2-ae6325b3319a)

![image](https://github.com/user-attachments/assets/488b787f-94df-48c1-aef4-55f16e3d7beb)

![image](https://github.com/user-attachments/assets/7e7d7877-3ce4-45be-b741-e96be9392db3)

![image](https://github.com/user-attachments/assets/5e567052-c82b-43c1-a1a1-1193c776f077)

### 连续信号到离散信号
![image](https://github.com/user-attachments/assets/990e1153-9f5e-46a3-bc28-5a41c750731c)

![image](https://github.com/user-attachments/assets/4dc7c03b-465c-4538-8581-d628a143aab9)

### 循环表示
![image](https://github.com/user-attachments/assets/66980e1f-ed42-4ae8-b2a5-86dcee9f805b)

![image](https://github.com/user-attachments/assets/b5c8ebda-bf1b-460e-a3b7-4fdfd9fc18e0)

### 卷积表示
![image](https://github.com/user-attachments/assets/6e0c04f0-6b76-4113-ad53-4077dd2e4d5a)

![image](https://github.com/user-attachments/assets/3feb038e-bbfc-47ad-96d3-648228d849a9)

![image](https://github.com/user-attachments/assets/d8235671-9c89-414a-a1c6-22d106d5e25b)

### 三种方式对比
![image](https://github.com/user-attachments/assets/204e174d-8c0b-4588-b0a8-d35bd3ceeb53)

![image](https://github.com/user-attachments/assets/9ca89ddb-84a8-4412-8898-58f363642c39)

### 矩阵 A 的重要性
![image](https://github.com/user-attachments/assets/22810ff9-4016-4722-a074-4305519435c0)

![image](https://github.com/user-attachments/assets/e6697daf-d773-4989-9b63-7d7a4cd0aa84)

## Mamba 架构之神奇的 S4
![image](https://github.com/user-attachments/assets/6579bdf0-cbc5-44be-867c-4c9a85ba9e47)

## Mamba 架构之选择性 SSM
![image](https://github.com/user-attachments/assets/cd164ec6-c710-4f6e-89f9-6646413daa76)

## Mamba 的劣势
![image](https://github.com/user-attachments/assets/2a41bfd8-5e3c-48f0-831a-bbab28420b9b)

## 小结
Mamba 在多个方面有出色表现，包括高效性、适应性、内存利用、训练速度和性能表现。这些优势使 Mamba 成为 Transformer 的强有力竞争者。无论是在学术研究还是工业应用中，Mamba 都有潜力带来显著的改进。

但 Mamba 也同样存在一些缺点，比如它复杂性高、资源需求大、迁移学习难度大、实验验证比较困难。一切都还需要时间去证明，而我们要做的就是时时跟进发展动态，以审慎的态度去了解、去尝试。

## 思考题
结合今天学习的内容，你来思考一下，Mamba 在变成熟的道路上，可能会遇到最大的难题是什么？如何解决？

# 第五章：热点速递，AI行业发展趋势解读 (5讲) 27｜机器人+大模型会产生什么化学反应？
![image](https://github.com/user-attachments/assets/fd928b8e-e6eb-44b0-8dd4-99d5cb134dd0)

![image](https://github.com/user-attachments/assets/77350788-b2da-44ad-80ef-c3685b3c3404)

## 机器人行业的困境
![image](https://github.com/user-attachments/assets/92a33068-3592-4f53-b9d5-caca22ea2116)

## 大模型如何赋能机器人？
![image](https://github.com/user-attachments/assets/6f2a63b6-65b8-4970-a5f8-a9f846f7f772)

![image](https://github.com/user-attachments/assets/aa7067ad-53c7-4d44-86fe-287cfbc93f4b)

## 技术框架
### RT-2
![image](https://github.com/user-attachments/assets/74be4f8c-3bc8-4ad9-a71e-c64057e0e9ce)

### Q-Transformer
![image](https://github.com/user-attachments/assets/e022b3c8-bd45-4339-8922-2e4976ff3685)

## 创业公司
![image](https://github.com/user-attachments/assets/47ad56e9-d70b-4c40-b02c-b835fd4cd485)

## 机器人的发展障碍
![image](https://github.com/user-attachments/assets/909b6748-4d8a-473e-94f5-b5a823eddd15)

## 小结
这节课我带你学习了大模型对机器人行业的赋能，大模型赋予了机器人强大的任务理解能力和代码生成能力，随着 AI 大模型技术的不断发展，机器人行业也会迎来春天，尤其是人形机器人，未来 10～20 年，肯定会有黑马杀出来。现在稍微领先一些的技术框架是 RT-2 和 Q-Transformer。如果你感兴趣的话，可以选择自己喜欢的技术框架和公司深入了解一下。

## 思考题
畅想一下，当人形机器人行业发展成熟后，可以为我们的生活带来什么变化？

# 第五章：热点速递，AI行业发展趋势解读 (5讲) 28｜Sora的突破：揭秘AI世界模拟器背后的技术演进
# 第五章：热点速递，AI行业发展趋势解读 (5讲) 29｜人工智能+无人机：掀起智能飞行领域革命
# 第五章：热点速递，AI行业发展趋势解读 (5讲) 30｜AI发展的下一阶段：什么是Q-Star(*)？
# 结束语 (2讲) 结束语｜相信自己，未来无限可能
## 超前意识：关于人工智能 +
![image](https://github.com/user-attachments/assets/0c307c33-d02b-4b8f-b2cf-264fa2f53dd5)

## 大胆去做：世界是一个草台班子
![image](https://github.com/user-attachments/assets/3fe51de1-9600-46b7-8462-fbe043a9b6f1)

## 静待结果：永葆一颗期待的心
![image](https://github.com/user-attachments/assets/9f470bae-a29c-4f17-ab36-24d487aca0f1)

## 关于信念：坚定不移地相信自己
![image](https://github.com/user-attachments/assets/2e78f05d-c27e-495b-8b80-6aea8ad9bde8)

# 结束语 (2讲) 期末测试｜来赴一场满分之约！
