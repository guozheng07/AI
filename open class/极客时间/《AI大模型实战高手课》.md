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
# 第二章：超燃实战，深度玩转 AI 模型 (4讲) 06｜RAG实战：基于ChatGLM3-6B+LangChain+Faiss搭建企业内部知识库
# 第二章：超燃实战，深度玩转 AI 模型 (4讲) 07｜大模型API封装：自建大模型如何对外服务？
# 第三章：打入核心，挑战底层技术原理 (8讲) 08｜关于机器学习，你需要了解的基本概念（一）
# 第三章：打入核心，挑战底层技术原理 (8讲) 09｜关于机器学习，你需要了解的基本概念（二）
# 第三章：打入核心，挑战底层技术原理 (8讲) 10｜经典算法之RNN：开发人员绕不开的循环神经网络
# 第三章：打入核心，挑战底层技术原理 (8讲) 11｜关于自然语言处理，你需要了解的基本概念
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
