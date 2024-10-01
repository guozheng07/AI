# 开篇词（1讲）｜带你亲证AI应用开发的“奇点”时刻
## AI 应用开发——新的历史节点
![image](https://github.com/user-attachments/assets/e201d131-8cdf-4ee1-aff1-bb48a0f17585)

## 何谓 LangChain？释放大语言模型潜能的利器
![image](https://github.com/user-attachments/assets/b6c88aaf-e8c4-4178-937f-55c63ca3c641)

![image](https://github.com/user-attachments/assets/161b8b43-b429-4275-a0c6-f808e73c354e)

## 打通 LangChain 从原理到应用的最后一公里
课程介绍：

![image](https://github.com/user-attachments/assets/d71123cf-5c94-45d2-83f8-3e361d31ebb2)

LangChain 中的具体组件包括：
- **模型（Models）**，包含各大语言模型的 LangChain 接口和调用细节，以及输出解析机制。
- **提示模板（Prompts）**，使提示工程流线化，进一步激发大语言模型的潜力。
- **数据检索（Indexes）**，构建并操作文档的方法，接受用户的查询并返回最相关的文档，轻松搭建本地知识库。
- **记忆（Memory）**，通过短时记忆和长时记忆，在对话过程中存储和检索数据，让 ChatBot 记住你是谁。
- **链（Chains）**，是 LangChain 中的核心机制，以特定方式封装各种功能，并通过一系列的组合，自动而灵活地完成常见用例。
- **代理（Agents）**，是另一个 LangChain 中的核心机制，通过“代理”让大模型自主调用外部工具和内部工具，使强大的“智能化”自主 Agent 成为可能！你的 App 将产生自驱力！

## LangChain 有趣用例抢先看
### 应用 1：情人节玫瑰宣传语

![image](https://github.com/user-attachments/assets/8385ca0f-75b0-444b-9002-d8d992cbeef5)

完成了上面两个步骤，就可以写代码了。
```
import os
os.environ["OPENAI_API_KEY"] = '你的OpenAI Key'
from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct",max_tokens=200)
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")

print(text)
```

![image](https://github.com/user-attachments/assets/625bebde-dfb4-4882-b458-8d794c819600)

### 应用 2：海报文案生成器
![image](https://github.com/user-attachments/assets/88439829-981e-4e56-aa1b-759ee54c85c1)

![image](https://github.com/user-attachments/assets/150a53bf-0320-4ffa-b401-7340d0e15796)

```
pip install --upgrade langchain
pip install transformers
pip install pillow
pip install torch torchvision torchaudio
```

```
#---- Part 0 导入所需要的类
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

#---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)
# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)

#---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
   
    name = "Image captioner"
    description = "为图片创作说明文案."

    def _run(self, url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

#---- PartIII 初始化并运行LangChain智能代理
# 设置OpenAI的API密钥并初始化大语言模型（OpenAI的Text模型）
os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
llm = OpenAI(temperature=0.2)

# 使用工具初始化智能代理并运行它
tools = [ImageCapTool()]
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)
img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'
agent.invoke(input=f"{img_url}\n请创作合适的中文推广文案")
```

根据输入的图片 URL，由 OpenAI 大语言模型驱动的 LangChain Agent，首先利用图像字幕生成工具将图片转化为字幕，然后对字幕做进一步处理，生成中文推广文案。

![image](https://github.com/user-attachments/assets/4f9b0b82-4b4d-4c2d-943b-f13f32875f47)

![image](https://github.com/user-attachments/assets/261e749f-1319-45cc-8a8f-b92b2a642721)

# 启程篇（2讲）01｜LangChain系统安装和快速入门
## 什么是大语言模型
![image](https://github.com/user-attachments/assets/1d67ed77-91e3-4ac8-95f0-3d5cbd0b1941)

![image](https://github.com/user-attachments/assets/26b08654-a2b6-4940-b9c5-07f299ec0ff8)

## 安装 LangChain
LangChain 支持 Python 和 JavaScript 两个开发版本，我们这个教程中全部使用 Python 版本进行讲解。

![image](https://github.com/user-attachments/assets/98449042-a919-41e2-b754-dfd2e4d2165a)

LangChain 的 [GitHub](https://github.com/langchain-ai/langchain) 社区非常活跃，你可以在这里找到大量的教程和最佳实践，也可以和其他开发者分享自己的经验和观点。

LangChain 也提供了详尽的 [API 文档](https://python.langchain.com/docs/introduction/)，这是你在遇到问题时的重要参考。不过呢，我觉得因为 LangChain 太新了，有时你可能会发现文档中有一些错误。在这种情况下，你可以考虑更新你的版本，或者在官方平台上提交一个问题反馈。

## OpenAI API
![image](https://github.com/user-attachments/assets/02f51f58-3bfd-4876-802d-53c956692cd8)

![image](https://github.com/user-attachments/assets/9b80a05a-8640-461b-a5e3-bed8ad03a990)

![image](https://github.com/user-attachments/assets/1d2e7bc1-4328-4bd8-8467-4bb9252dbd9a)

- **Chat Model，聊天模型**，用于产生人类和 AI 之间的对话，代表模型当然是 gpt-3.5-turbo（也就是 ChatGPT）和 GPT-4。当然，OpenAI 还提供其它的版本，gpt-3.5-turbo-0613 代表 ChatGPT 在 2023 年 6 月 13 号的一个快照，而 gpt-3.5-turbo-16k 则代表这个模型可以接收 16K 长度的 Token，而不是通常的 4K。（注意了，gpt-3.5-turbo-16k 并未开放给我们使用，而且你传输的字节越多，花钱也越多）
- **Text Model，文本模型**，在 ChatGPT 出来之前，大家都使用这种模型的 API 来调用 GPT-3，文本模型的代表作是 text-davinci-003（基于 GPT3）。而在这个模型家族中，也有专门训练出来做文本嵌入的 text-embedding-ada-002，也有专门做相似度比较的模型，如 text-similarity-curie-001。上面这两种模型，提供的功能类似，都是接收对话输入（input，也叫 prompt），返回回答文本（output，也叫 response）。但是，它们的调用方式和要求的输入格式是有区别的，这个我们等下还会进一步说明。

下面我们用简单的代码段说明上述两种模型的调用方式。

### 调用 Text 模型（GPT3.5 之前的版本）
- 第 1 步，先注册好你的 API Key。
- 第 2 步，用 pip install openai 命令来安装 OpenAI 库。
- 第 3 步，导入 OpenAI API Key。
  ![image](https://github.com/user-attachments/assets/89d6d508-d451-4631-a0c6-54e78474c378)
- 第 4 步，导入 OpenAI 库，并创建一个 Client。
  ```
  from openai import OpenAI
  client = OpenAI()
  ```
- 第 5 步，指定 gpt-3.5-turbo-instruct（也就是 Text 模型）并调用 completions 方法，返回结果。
  ```
  response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  temperature=0.5,
  max_tokens=100,
  prompt="请给我的花店起个名")
  ```
  在使用 OpenAI 的文本生成模型时，你可以通过一些参数来控制输出的内容和样式。这里我总结为了一些常见的参数。
  ![image](https://github.com/user-attachments/assets/52d96727-5567-459c-8713-1d0dbee1b419)
- 第 6 步，打印输出大模型返回的文字。
  ```
  print(response.choices[0].text.strip())
  ```

当你调用 OpenAI 的 Completion.create 方法时，它会返回一个响应对象，该对象包含了模型生成的输出和其他一些信息。这个响应对象是一个字典结构，包含了多个字段。

在使用 Text 模型（如 text-davinci-003）的情况下，响应对象的主要字段包括：

![image](https://github.com/user-attachments/assets/f72088e5-1308-479f-a189-86e574ded09a)

choices 字段是一个列表，因为在某些情况下，你可以要求模型生成多个可能的输出。每个选择都是一个字典，其中包含以下字段：
- text：模型生成的文本。
- finish_reason：模型停止生成的原因，可能的值包括 stop（遇到了停止标记）、length（达到了最大长度）或 temperature（根据设定的温度参数决定停止）。

所以，response.choices[0].text.strip() 这行代码的含义是：从响应中获取第一个（如果在调用大模型时，没有指定 n 参数，那么就只有唯一的一个响应）选择，然后获取该选择的文本，并移除其前后的空白字符。这通常是你想要的模型的输出。

至此，任务完成，模型的输出如下：
```
心动花庄、芳华花楼、轩辕花舍、簇烂花街、满园春色
```

### 调用 Chat 模型
整体流程上，Chat 模型和 Text 模型的调用是类似的，只是前面加了一个 chat，然后输入（prompt）和输出（response）的数据格式有所不同。

示例代码如下：
```
response = client.chat.completions.create(  
  model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名"},
    ],
  temperature=0.8,
  max_tokens=60
)
```

这段代码中，除去刚才已经介绍过的 temperature、max_tokens 等参数之外，**有两个专属于 Chat 模型的概念，一个是消息，一个是角色**！

先说**消息**，消息就是传入模型的提示。此处的 messages 参数是一个列表，包含了多个消息。每个消息都有一个 role（可以是 system、user 或 assistant）和 content（消息的内容）。系统消息设定了对话的背景（你是一个很棒的智能助手），然后用户消息提出了具体请求（请给我的花店起个名）。模型的任务是基于这些消息来生成回复。

再说**角色**，在 OpenAI 的 Chat 模型中，system、user 和 assistant 都是消息的角色。每一种角色都有不同的含义和作用。
- system：系统消息主要用于设定对话的背景或上下文。这可以帮助模型理解它在对话中的角色和任务。例如，你可以通过系统消息来设定一个场景，让模型知道它是在扮演一个医生、律师或者一个知识丰富的 AI 助手。系统消息通常在对话开始时给出。
- user：用户消息是从用户或人类角色发出的。它们通常包含了用户想要模型回答或完成的请求。用户消息可以是一个问题、一段话，或者任何其他用户希望模型响应的内容。
- assistant：助手消息是模型的回复。例如，在你使用 API 发送多轮对话中新的对话请求时，可以通过助手消息提供先前对话的上下文。然而，请注意在对话的最后一条消息应始终为用户消息，因为模型总是要回应最后这条用户消息。

在使用 Chat 模型生成内容后，返回的响应，也就是 response 会包含一个或多个 choices，每个 choices 都包含一个 message。每个 message 也都包含一个 role 和 content。role 可以是 system、user 或 assistant，表示该消息的发送者，content 则包含了消息的实际内容。

一个典型的 response 对象可能如下所示：
```
{
 'id': 'chatcmpl-2nZI6v1cW9E3Jg4w2Xtoql0M3XHfH',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-4',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': '你的花店可以叫做"花香四溢"。'
     },
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}
```

以下是各个字段的含义：

![image](https://github.com/user-attachments/assets/a50c2810-b632-4c8a-9f19-212cce84365e)

这就是 response 的基本结构，其实它和 Text 模型返回的响应结构也是很相似，只是 choices 字段中的 Text 换成了 Message。你可以通过解析这个对象来获取你需要的信息。例如，要获取模型的回复，可使用 response[‘choices’][0][‘message’][‘content’]。

### Chat 模型 vs Text 模型
![image](https://github.com/user-attachments/assets/3aa07070-8315-4e56-bbbb-23dd93bad558)

## 通过 LangChain 调用 Text 和 Chat 模型
### 调用 Text 模型
```
import os
os.environ["OPENAI_API_KEY"] = '你的Open API Key'
from langchain.llms import OpenAI

# 通过 OpenAI 类创建对象
llm = OpenAI(  
    model="gpt-3.5-turbo-instruct",
    temperature=0.8,
    max_tokens=60,)
response = llm.predict("请给我的花店起个名")

print(response)
```

输出：
```
花之缘、芳华花店、花语心意、花风旖旎、芳草世界、芳色年华
```

### 调用 Chat 模型
```
import os
os.environ["OPENAI_API_KEY"] = '你的Open API Key'
from langchain.chat_models import ChatOpenAI

# 通过 ChatOpenAI 类创建对象
chat = ChatOpenAI(model="gpt-4",
                    temperature=0.8,
                    max_tokens=60)
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]
response = chat(messages)

print(response)
```

输出：
```
content='当然可以，叫做"花语秘境"怎么样？' 
additional_kwargs={} example=False
```

另外，无论是 langchain.llms 中的 OpenAI（Text 模型），还是 langchain.chat_models 中的 ChatOpenAI 中的 ChatOpenAI（Chat 模型），其返回的结果 response 变量的结构，都比直接调用 OpenAI API 来得简单一些。这是因为，LangChain 已经对大语言模型的 output 进行了解析，只保留了响应中最重要的文字部分。

## 总结
![image](https://github.com/user-attachments/assets/c7547388-9830-4422-884d-4fba06a85ee0)

## 思考题
![image](https://github.com/user-attachments/assets/712f2e71-f895-4e4d-bb32-08f08403aa51)

## 延伸阅读
1. LangChain 官方文档（[Python 版](https://python.langchain.com/docs/tutorials/)）（[JavaScript 版](https://js.langchain.com/docs/tutorials/)），这是你学习专栏的过程中，有任何疑惑都可以随时去探索的知识大本营。我个人觉得，目前 LangChain 的文档还不够体系化，有些杂乱，讲解也不大清楚。但是，这是官方文档，会维护得越来越好。
2. [OpenAI API 官方文档](https://platform.openai.com/docs/introduction)，深入学习 OpenAI API 的地方。
3. [HuggingFace 官方网站](https://huggingface.co/)，玩开源大模型的好地方。

# 启程篇（2讲）02｜用LangChain快速构建基于“易速鲜花”本地知识库的智能问答系统
## 项目及实现框架
我们先来整体了解一下这个项目。

**项目名称**：“易速鲜花”内部员工知识库问答系统。

**项目介绍**：“易速鲜花”作为一个大型在线鲜花销售平台，有自己的业务流程和规范，也拥有针对员工的 SOP 手册。新员工入职培训时，会分享相关的信息。但是，这些信息分散于内部网和 HR 部门目录各处，有时不便查询；有时因为文档过于冗长，员工无法第一时间找到想要的内容；有时公司政策已更新，但是员工手头的文档还是旧版内容。

基于上述需求，我们将开发一套基于各种内部知识手册的 “Doc-QA” 系统。这个系统将充分利用 LangChain 框架，处理从员工手册中产生的各种问题。这个问答系统能够理解员工的问题，并基于最新的员工手册，给出精准的答案。

**开发框架**：下面这张图片描述了通过 LangChain 框架实现一个知识库文档系统的整体框架。

![image](https://github.com/user-attachments/assets/18fe378a-4598-41fb-a76e-24b54bfe4c43)

整个框架分为这样三个部分。
- 数据源（Data Sources）：数据可以有很多种，包括 PDF 在内的非结构化的数据（Unstructured Data）、SQL 在内的结构化的数据（Structured Data），以及 Python、Java 之类的代码（Code）。在这个示例中，我们聚焦于对非结构化数据的处理。
- 大模型应用（Application，即 LLM App）：以大模型为逻辑引擎，生成我们所需要的回答。
- 用例（Use-Cases）：大模型生成的回答可以构建出 QA/ 聊天机器人等系统。

**核心实现机制**：这个项目的核心实现机制是下图所示的数据处理管道（Pipeline）。

![image](https://github.com/user-attachments/assets/b701a170-b6ee-4624-b967-7db35dabe90c)

在这个管道的每一步中，LangChain 都为我们提供了相关工具，让你轻松实现基于文档的问答功能。

具体流程分为下面 5 步。
1. Loading：文档加载器把 Documents **加载**为以 LangChain 能够读取的形式。
2. Splitting：文本分割器把 Documents **切分**为指定大小的分割，我把它们称为“文档块”或者“文档片”。
3. Storage：将上一步中分割好的“文档块”以“嵌入”（Embedding）的形式**存储**到向量数据库（Vector DB）中，形成一个个的“嵌入片”。
4. Retrieval：应用程序从存储中**检索**分割后的文档（例如通过比较余弦相似度，找到与输入问题类似的嵌入片）。
5. Output：把问题和相似的嵌入片传递给语言模型（LLM），使用包含问题和检索到的分割的提示**生成答案**。

## 数据的准备和载入
“易速鲜花”的内部资料包括 pdf、word 和 txt 格式的各种文件，我已经放在[这里](https://github.com/huangjia2019/langchain-in-action/tree/main/02_%E6%96%87%E6%A1%A3QA%E7%B3%BB%E7%BB%9F/OneFlower)供你下载。

我们首先用 LangChain 中的 document_loaders 来加载各种格式的文本文件。（这些文件我把它放在 OneFlower 这个目录中了，如果你创建自己的文件夹，就要调整一下代码中的目录）

在这一步中，我们从 pdf、word 和 txt 文件中加载文本，然后将这些文本存储在一个列表中。（注意：可能需要安装 PyPDF、Docx2txt 等库）

```
import os
os.environ["OPENAI_API_KEY"] = '你的Open AI API Key'

# 1.Load 导入 Document Loaders
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

# 加载 Documents
base_dir = '.\OneFlower' # 文档的存放目录
documents = []
for file in os.listdir(base_dir): 
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'): 
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
```

这里我们首先导入了 OpenAI 的 API Key。因为后面我们需要利用 Open AI 的两种不同模型做以下两件事：
- 用 OpenAI 的 Embedding 模型为文档做嵌入。
- 调用 OpenAI 的 GPT 模型来生成问答系统中的回答。

当然了，LangChain 所支持的大模型绝不仅仅是 Open AI 而已，你完全可以遵循这个框架，把 Embedding 模型和负责生成回答的语言模型都替换为其他的开源模型。

在运行上面的程序时，除了要导入正确的 Open AI Key 之外，还要注意的是工具包的安装。使用 LangChain 时，根据具体的任务，往往需要各种不同的工具包（比如上面的代码需要 PyPDF 和 Docx2txt 工具）。它们安装起来都非常简单，如果程序报错缺少某个包，只要通过 pip install 安装相关包即可。

## 文本的分割
接下来需要将加载的文本分割成更小的块，以便进行嵌入和向量存储。这个步骤中，我们使用 LangChain 中的 RecursiveCharacterTextSplitter 来分割文本。

```
# 2.Split 将 Documents 切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
```

现在，我们的文档被切成了一个个 200 字符左右的文档块。这一步，是为把它们存储进下面的向量数据库做准备。

## 向量数据库存储
紧接着，我们将这些分割后的文本转换成嵌入的形式，并将其存储在一个向量数据库中。在这个例子中，我们使用了 OpenAIEmbeddings 来生成嵌入，然后使用 Qdrant 这个向量数据库来存储嵌入（这里需要 pip install qdrant-client）。

文本的“嵌入”和向量数据库的说明：

![image](https://github.com/user-attachments/assets/88f6a162-1a95-4180-afae-e9353f85cccc)

![image](https://github.com/user-attachments/assets/79a2b9d9-719f-4508-8e29-dff53c8ce4e9)

向量数据库有很多种，比如 Pinecone、Chroma 和 Qdrant，有些是收费的，有些则是开源的。

LangChain 中支持很多向量数据库，这里我们选择的是开源向量数据库 Qdrant。（注意，需要安装 qdrant-client）具体实现代码如下：
```
# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=OpenAIEmbeddings(), # 用OpenAI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name
```
目前，易速鲜花的所有内部文档，都以“文档块嵌入片”的格式被存储在向量数据库里面了。那么，我们只需要查询这个向量数据库，就可以找到大体上相关的信息了。

## 相关信息的获取
![image](https://github.com/user-attachments/assets/a7e88353-fc88-4e03-94d4-b9f3b1325b04)

![image](https://github.com/user-attachments/assets/e9904d85-239f-4675-86c7-c390a9c06176)

在这里，我们正在处理的是文本数据，目标是建立一个问答系统，需要从语义上理解和比较问题可能的答案。因此，我建议使用余弦相似度作为度量标准。通过比较问题和答案向量在语义空间中的方向，可以找到与提出的问题最匹配的答案。

在这一步的代码部分，我们会创建一个聊天模型。然后需要创建一个 RetrievalQA 链，它是一个检索式问答模型，用于生成问题的答案。

在 RetrievalQA 链中有下面两大重要组成部分。
- LLM 是大模型，负责回答问题。
- retriever（vectorstore.as_retriever()）负责根据问题检索相关的文档，找到具体的“嵌入片”。这些“嵌入片”对应的“文档块”就会作为知识信息，和问题一起传递进入大模型。本地文档中检索而得的知识很重要，因为**从互联网信息中训练而来的大模型不可能拥有“易速鲜花”作为一个私营企业的内部知识**。

具体代码如下：
```
# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain.chat_models import ChatOpenAI # ChatOpenAI模型
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个大模型工具 - OpenAI的GPT-3.5
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)
```

现在我们已经为后续的步骤做好了准备，下一步就是接收来自系统用户的具体问题，并根据问题检索信息，生成回答。

## 生成回答并展示
这一步是问答系统应用的主要 UI 交互部分，这里会创建一个 Flask 应用（需要安装 Flask 包）来接收用户的问题，并生成相应的答案，最后通过 index.html 对答案进行渲染和呈现。

在这个步骤中，我们使用了之前创建的 RetrievalQA 链来获取相关的文档和生成答案。然后，将这些信息返回给用户，显示在网页上。

```
# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # 接收用户输入作为问题
        question = request.form.get('question')        
        
        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})
        
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)
```

相关 HTML 网页的关键代码如下：
```
<body>
    <div class="container">
        <div class="header">
            <h1>易速鲜花内部问答系统</h1>
            <img src="{{ url_for('static', filename='flower.png') }}" alt="flower logo" width="200"> 
        </div>
        <form method="POST">
            <label for="question">Enter your question:</label><br>
            <input type="text" id="question" name="question"><br>
            <input type="submit" value="Submit">
        </form>
        {% if result is defined %}
            <h2>Answer</h2>
            <p>{{ result.result }}</p>
        {% endif %}
    </div>
</body>
```

运行程序之后，我们跑起一个网页 http://127.0.0.1:5000/。与网页进行互动时，可以发现，问答系统完美生成了专属于异速鲜花内部资料的回答。

![image](https://github.com/user-attachments/assets/dbf4c320-dd3d-4f93-ad8d-f3b5ab2e0fb6)

## 总结
![image](https://github.com/user-attachments/assets/c4c8c1f3-4231-40e7-bde8-8812618f6ed5)

## 思考题
1. 请你用自己的话简述一下这个基于文档的 QA（问答）系统的实现流程？
2. LangChain 支持很多种向量数据库，你能否用另一种常用的向量数据库 Chroma 来实现这个任务？
3. LangChain 支持很多种大语言模型，你能否用 HuggingFace 网站提供的开源模型 google/flan-t5-x1 代替 GPT-3.5 完成这个任务？

# 基础篇（11讲）03｜模型I/O：输入提示、调用模型、解析输出
从这节课开始，我们将对 LangChain 中的六大核心组件一一进行详细的剖析。

模型，位于 LangChain 框架的最底层，它是基于语言模型构建的应用的**核心元素**，因为所谓 LangChain 应用开发，就是以 LangChain 作为框架，通过 API 调用大模型来解决具体问题的过程。

可以说，整个 LangChain 框架的逻辑都是由 LLM 这个发动机来驱动的。没有模型，LangChain 这个框架也就失去了它存在的意义。那么这节课我们就详细讲讲模型，最后你会收获一个能够自动生成鲜花文案的应用程序。

## Model I/O
我们可以把对模型的使用过程拆解成三块，分别是**输入提示**（对应图中的 Format）、**调用模型**（对应图中的 Predict）和**输出解析**（对应图中的 Parse）。这三块形成了一个整体，因此在 LangChain 中这个过程被统称为 **Model I/O**（Input/Output）。

![image](https://github.com/user-attachments/assets/3c3b958c-1db5-48e5-bdfa-b4579d2b66bb)

在模型 I/O 的每个环节，LangChain 都为咱们提供了模板和工具，快捷地形成调用各种语言模型的接口。
1. 提示模板：使用模型的第一个环节是把提示信息输入到模型中，你可以创建 LangChain 模板，根据实际需求动态选择不同的输入，针对特定的任务和应用调整输入。
2. 语言模型：LangChain 允许你通过通用接口来调用语言模型。这意味着无论你要使用的是哪种语言模型，都可以通过同一种方式进行调用，这样就提高了灵活性和便利性。
3. 输出解析：LangChain 还提供了从模型输出中提取信息的功能。通过输出解析器，你可以精确地从模型的输出中获取需要的信息，而不需要处理冗余或不相关的数据，更重要的是还可以把大模型给回的非结构化文本，转换成程序可以处理的结构化数据。

下面我们用示例的方式来深挖一下这三个环节。先来看看 LangChain 中提示模板的构建。

### 提示模板
![image](https://github.com/user-attachments/assets/8e33aeff-5f3d-46cf-9ccf-8538050f8177)

这个提示模板的生成方式如下：
```
# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate
# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template) 
# 打印LangChain提示模板的内容
print(prompt)
```

提示模板的具体内容如下：
```
input_variables=['flower_name', 'price'] 
output_parser=None partial_variables={} 
template='/\n您是一位专业的鲜花店文案撰写员。
\n对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？\n'
template_format='f-string' 
validate_template=True
```

![image](https://github.com/user-attachments/assets/f70dbda1-c198-4715-9004-1631d2f06011)

### 语言模型
LangChain 中支持的模型有三大类。
- 大语言模型（LLM） ，也叫 Text Model，这些模型将文本字符串作为输入，并返回文本字符串作为输出。Open AI 的 text-davinci-003、Facebook 的 LLaMA、ANTHROPIC 的 Claude，都是典型的 LLM。
- 聊天模型（Chat Model），主要代表 Open AI 的 ChatGPT 系列模型。这些模型通常由语言模型支持，但它们的 API 更加结构化。具体来说，这些模型将聊天消息列表作为输入，并返回聊天消息。
- 文本嵌入模型（Embedding Model），这些模型将文本作为输入并返回浮点数列表，也就是 Embedding。而文本嵌入模型如 OpenAI 的 text-embedding-ada-002，我们之前已经见过了。文本嵌入模型负责把文档存入向量数据库，和我们这里探讨的提示工程关系不大。

然后，我们将调用语言模型，让模型帮我们写文案，并且返回文案的结果。
```
# 导入LangChain中的提示模板
from langchain import PromptTemplate
# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template) 
# 打印LangChain提示模板的内容
print(prompt)

# 设置OpenAI API Key
import os
os.environ["OPENAI_API_KEY"] = '你的Open AI API Key'

# 导入LangChain中的OpenAI模型接口
from langchain import OpenAI
# 创建模型实例
model = OpenAI(model_name='gpt-3.5-turbo-instruct')

# 多种花的列表
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 生成多种花的文案
for flower, price in zip(flowers, prices):
    # 使用提示模板生成输入
    input_prompt = prompt.format(flower_name=flower, price=price)

    # 得到模型的输出
    output = model.invoke(input_prompt)

    # 打印输出内容
    print(output)
```

![image](https://github.com/user-attachments/assets/bd290d41-dd7b-45a8-be84-15de34fe5dd8)

![image](https://github.com/user-attachments/assets/d941b8a9-3ae8-4e5f-96c1-a9aa8cb66485)

### 输出解析
LangChain 提供的解析模型输出的功能，使你能够更容易地从模型输出中获取结构化的信息，这将大大加快基于语言模型进行应用开发的效率。

为什么这么说呢？请你思考一下刚才的例子，你只是让模型生成了一个文案。这段文字是一段字符串，正是你所需要的。但是，在开发具体应用的过程中，很明显**我们不仅仅需要文字，更多情况下我们需要的是程序能够直接处理的、结构化的数据**。

比如说，在这个文案中，如果你希望模型返回两个字段：
- description：鲜花的说明文本
- reason：解释一下为何要这样写上面的文案

下面，我们就通过 LangChain 的输出解析器来重构程序，让模型有能力生成结构化的回应，同时对其进行解析，直接将解析好的数据存入 CSV 文档。
```
# 导入OpenAI Key
import os
os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate
# 创建原始提示模板
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""

# 通过LangChain调用模型
from langchain_openai import OpenAI
# 创建模型实例
model = OpenAI(model_name='gpt-3.5-turbo-instruct')

# 导入结构化输出解析器和ResponseSchema
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# 定义我们想要接收的响应模式
response_schemas = [
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="问什么要这样写这个文案")
]
# 创建输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指示
format_instructions = output_parser.get_format_instructions()
# 根据原始模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, 
                partial_variables={"format_instructions": format_instructions}) 

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 创建一个空的DataFrame用于存储结果
import pandas as pd
df = pd.DataFrame(columns=["flower", "price", "description", "reason"]) # 先声明列名

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower_name=flower, price=price)

    # 获取模型的输出
    output = model.invoke(input)
    
    # 解析模型的输出（这是一个字典结构）
    parsed_output = output_parser.parse(output)

    # 在解析后的输出中添加“flower”和“price”
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output  

# 打印字典
print(df.to_dict(orient='records'))

# 保存DataFrame到CSV文件
df.to_csv("flowers_with_descriptions.csv", index=False)
```

![image](https://github.com/user-attachments/assets/26f63e75-bf68-41ae-923d-f0fc5f7a7ac7)

输出：
```
[{'flower': '玫瑰', 'price': '50', 'description': 'Luxuriate in the beauty of this 50 yuan rose, with its deep red petals and delicate aroma.', 'reason': 'This description emphasizes the elegance and beauty of the rose, which will be sure to draw attention.'}, 
{'flower': '百合', 'price': '30', 'description': '30元的百合，象征着坚定的爱情，带给你的是温暖而持久的情感！', 'reason': '百合是象征爱情的花，写出这样的描述能让顾客更容易感受到百合所带来的爱意。'}, 
{'flower': '康乃馨', 'price': '20', 'description': 'This beautiful carnation is the perfect way to show your love and appreciation. Its vibrant pink color is sure to brighten up any room!', 'reason': 'The description is short, clear and appealing, emphasizing the beauty and color of the carnation while also invoking a sense of love and appreciation.'}]
```
## 思考题
1. 请你用自己的理解，简述 LangChain 调用大语言模型来做应用开发的优势。
2. 在上面的示例中，format_instructions，也就是输出格式是怎样用 output_parser 构建出来的，又是怎样传递到提示模板中的？
3. 加入了 partial_variables，也就是输出解析器指定的 format_instructions 之后的提示，为什么能够让模型生成结构化的输出？你可以打印出这个提示，一探究竟。
4. 使用输出解析器后，调用模型时有没有可能仍然得不到所希望的输出？也就是说，模型有没有可能仍然返回格式不够完美的输出？

# 基础篇（11讲）04｜提示工程（上）：用少样本FewShotTemplate和ExampleSelector创建应景文案
这节课我就带着你进一步深究，如何利用 LangChain 中的提示模板，做好提示工程。

![image](https://github.com/user-attachments/assets/13c51b7f-342a-416a-b956-b19290b50144)

## 提示的结构
当然了，从大原则到实践，还是有一些具体工作需要说明，下面我们先看一个实用的提示框架。

![image](https://github.com/user-attachments/assets/9c3275e9-744c-4f82-92fe-394ccdef22df)

在这个提示框架中：
- **指令**（Instuction）告诉模型这个任务大概要做什么、怎么做，比如如何使用提供的外部信息、如何处理查询以及如何构造输出。这通常是一个提示模板中比较固定的部分。一个常见用例是告诉模型“你是一个有用的 XX 助手”，这会让他更认真地对待自己的角色。
- **上下文**（Context）则充当模型的额外知识来源。这些信息可以手动插入到提示中，通过矢量数据库检索得来，或通过其他方式（如调用 API、计算器等工具）拉入。一个常见的用例时是把从向量数据库查询到的知识作为上下文传递给模型。
- **提示输入**（Prompt Input）通常就是具体的问题或者需要大模型做的具体事情，这个部分和“指令”部分其实也可以合二为一。但是拆分出来成为一个独立的组件，就更加结构化，便于复用模板。这通常是作为变量，在调用模型之前传递给提示模板，以形成具体的提示。
- **输出指示器**（Output Indicator）标记​​要生成的文本的开始。这就像我们小时候的数学考卷，先写一个“解”，就代表你要开始答题了。如果生成 Python 代码，可以使用 “import” 向模型表明它必须开始编写 Python 代码（因为大多数 Python 脚本以 import 开头）。这部分在我们和 ChatGPT 对话时往往是可有可无的，当然 LangChain 中的代理在构建提示模板时，经常性的会用一个“Thought：”（思考）作为引导词，指示模型开始输出自己的推理（Reasoning）。

下面，就让我们看看如何使用 LangChain 中的各种提示模板做提示工程，将更优质的提示输入大模型。

## LangChain 提示模板的类型
LangChain 中提供 String（StringPromptTemplate）和 Chat（BaseChatPromptTemplate）两种基本类型的模板，并基于它们构建了不同类型的提示模板：

![image](https://github.com/user-attachments/assets/e5d13cb5-2131-4525-8346-3f056f27028a)

这些模板的导入方式如下：
```
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
```
### 使用 PromptTemplate
```
from langchain import PromptTemplate

template = """\
你是业务咨询顾问。
你给一个销售{product}的电商公司，起一个好的名字？
"""
prompt = PromptTemplate.from_template(template)

print(prompt.format(product="鲜花"))
```

![image](https://github.com/user-attachments/assets/ae746b8e-3bb8-4dcb-969d-a1d1735f5856)

当然，也可以通过提示模板类的构造函数，在创建模板时手工指定 input_variables，示例如下：
```
prompt = PromptTemplate(
    input_variables=["product", "market"], 
    template="你是业务咨询顾问。对于一个面向{market}市场的，专注于销售{product}的公司，你会推荐哪个名字？"
)
print(prompt.format(product="鲜花", market="高端"))
```
上面的方式直接生成了提示模板，并没有通过 from_template 方法从字符串模板中创建提示模板。二者效果是一样的。

### 使用 ChatPromptTemplate
对于 OpenAI 推出的 ChatGPT 这一类的聊天模型，LangChain 也提供了一系列的模板，这些模板的不同之处是它们有对应的角色。

![image](https://github.com/user-attachments/assets/c505b5a2-cba3-4d9d-a875-3358f23a2fdd)

下面，给出一个示例。
```
# 导入聊天消息类模板
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# 模板的构建
template="你是一位专业顾问，负责为专注于{product}的公司起名。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="公司主打产品是{product_detail}。"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 格式化提示消息生成提示
prompt = prompt_template.format_prompt(product="鲜花装饰", product_detail="创新的鲜花设计。").to_messages()

# 下面调用模型，把提示传入模型，生成结果
import os
os.environ["OPENAI_API_KEY"] = '你的OpenAI Key'
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI()
result = chat(prompt)
print(result)
```

输出：
```
content='1. 花语创意\n2. 花韵设计\n3. 花艺创新\n4. 花漾装饰\n5. 花语装点\n6. 花翩翩\n7. 花语之美\n8. 花馥馥\n9. 花语时尚\n10. 花之魅力' 
additional_kwargs={} 
example=False
```

讲完上面两种简单易用的提示模板，下面开始介绍今天的重点内容，FewShotPromptTemplate。FewShot，也就是少样本这一概念，是提示工程中非常重要的部分，对应着 OpenAI 提示工程指南中的第 2 条——给模型提供参考（也就是示例）。

### 使用 FewShotPromptTemplate
在提示工程（Prompt Engineering）中，Few-Shot 和 Zero-Shot 学习的概念也被广泛应用。
- 在 Few-Shot 学习设置中，模型会被给予几个示例，以帮助模型理解任务，并生成正确的响应。
- 在 Zero-Shot 学习设置中，模型只根据任务的描述生成响应，不需要任何示例。

下面，就让我们来通过 LangChain 中的 FewShotPromptTemplate 构建出最合适的鲜花文案。
1. 创建示例样本
   首先，创建一些示例。samples 这个列表包含了四个字典，每个字典代表了一种花的类型、适合的场合，以及对应的广告文案。这些示例样本，就是构建 FewShotPrompt 时，作为例子传递给模型的参考信息。
   ```
   # 1. 创建一些示例
   samples = [
     {
       "flower_type": "玫瑰",
       "occasion": "爱情",
       "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
     },
     {
       "flower_type": "康乃馨",
       "occasion": "母亲节",
       "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
     },
     {
       "flower_type": "百合",
       "occasion": "庆祝",
       "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
     },
     {
       "flower_type": "向日葵",
       "occasion": "鼓励",
       "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
     }
   ]
   ```
2. 创建提示模板
   配置一个提示模板，将一个示例格式化为字符串。这个格式化程序应该是一个 PromptTemplate 对象。
   ```
   # 2. 创建一个提示模板
   from langchain.prompts.prompt import PromptTemplate
   template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
   prompt_sample = PromptTemplate(input_variables=["flower_type", "occasion", "ad_copy"], 
                                  template=template)
   print(prompt_sample.format(**samples[0])) # 只输入了一个示例
   ```
   在这个步骤中，我们创建了一个 PromptTemplate 对象。这个对象是根据指定的输入变量和模板字符串来生成提示的。在这里，输入变量包括 "flower_type"、"occasion"、"ad_copy"，模板是一个字符串，其中包含了用大括号包围的变量名，它们会被对应的变量值替换。
3. 创建 FewShotPromptTemplate 对象
   ```
   # 3. 创建一个FewShotPromptTemplate对象
   from langchain.prompts.few_shot import FewShotPromptTemplate
   prompt = FewShotPromptTemplate(
       examples=samples,
       example_prompt=prompt_sample,
       suffix="鲜花类型: {flower_type}\n场合: {occasion}",
       input_variables=["flower_type", "occasion"]
   )
   print(prompt.format(flower_type="野玫瑰", occasion="爱情"))
   ```
   可以看到，FewShotPromptTemplate 是一个更复杂的提示模板，它包含了多个示例和一个提示。这种模板可以使用多个示例来指导模型生成对应的输出。目前我们创建一个新提示，其中包含了根据指定的花的类型“野玫瑰”和场合“爱情”。
4. 调用大模型创建新文案
   ```
   # 4. 把提示传递给大模型
   import os
   os.environ["OPENAI_API_KEY"] = '你的Open AI Key'
   from langchain.llms import OpenAI
   model = OpenAI(model_name='gpt-3.5-turbo-instruct')
   result = model(prompt.format(flower_type="野玫瑰", occasion="爱情"))
   print(result)
   ```
### 使用示例选择器
如果我们的示例很多，那么一次性把所有示例发送给模型是不现实而且低效的。另外，每次都包含太多的 Token 也会浪费流量（OpenAI 是按照 Token 数来收取费用）。

LangChain 给我们提供了示例选择器，来选择最合适的样本。（注意，因为示例选择器使用向量相似度比较的功能，此处需要安装向量数据库，这里我使用的是开源的 Chroma，你也可以选择之前用过的 Qdrant。）下面，

就是使用示例选择器的示例代码。
```
# 5. 使用示例选择器
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 初始化示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=prompt_sample, 
    suffix="鲜花类型: {flower_type}\n场合: {occasion}", 
    input_variables=["flower_type", "occasion"]
)
print(prompt.format(flower_type="红玫瑰", occasion="爱情"))
```
在这个步骤中，它首先创建了一个 SemanticSimilarityExampleSelector 对象，这个对象可以根据语义相似性选择最相关的示例。

然后，它创建了一个新的 FewShotPromptTemplate 对象，这个对象使用了上一步创建的选择器来选择最相关的示例生成提示。然后，我们又用这个模板生成了一个新的提示，因为我们的提示中需要创建的是红玫瑰的文案，所以，示例选择器 example_selector 会根据语义的相似度（余弦相似度）找到最相似的示例，也就是“玫瑰”，并用这个示例构建了 FewShot 模板。

这样，我们就避免了把过多的无关模板传递给大模型，以节省 Token 的用量。

## 总结
总的来说，提供示例对于解决某些任务至关重要，通常情况下，FewShot 的方式能够显著提高模型回答的质量。不过，当少样本提示的效果不佳时，这可能表示模型在任务上的学习不足。在这种情况下，我们建议对模型进行微调或尝试更高级的提示技术。

下一节课，我们将在探讨输出解析的同时，讲解另一种备受关注的提示技术，被称为“思维链提示”（Chain of Thought，简称 CoT）。这种技术因其独特的应用方式和潜在的实用价值而引人注目。

## 思考题
![image](https://github.com/user-attachments/assets/994741be-56dc-4c94-a0cf-05bcc167b880)

# 基础篇（11讲）05｜提示工程（下）：用思维链和思维树提升模型思考质量
## 什么是 Chain of Thought
CoT 这个概念来源于学术界，是谷歌大脑的 Jason Wei 等人于 2022 年在论文《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（自我一致性提升了语言模型中的思维链推理能力）》中提出来的概念。它提出，如果生成一系列的中间推理步骤，就能够显著提高大型语言模型进行复杂推理的能力。

### Few-Shot CoT
![image](https://github.com/user-attachments/assets/8542c55b-8b13-4edc-a992-8a28ec912ecb)

![image](https://github.com/user-attachments/assets/2dd59cab-134d-423d-8bb4-62294def72a7)

在这个过程中，整体上，思维链引导 AI 从理解问题，到搜索信息，再到制定决策，最后生成销售列表。这种方法不仅使 AI 的推理过程更加清晰，也使得生成的销售列表更加符合用户的需求。具体到每一个步骤，也可以通过思维链来设计更为详细的提示模板，来引导模型每一步的思考都遵循清晰准确的逻辑。

**其实 LangChain 的核心组件 Agent 的本质就是进行好的提示工程，并大量地使用预置的 FewShot 和 CoT 模板**。这个在之后的课程学习中我们会理解得越来越透彻。

### Zero-Shot CoT
![image](https://github.com/user-attachments/assets/fe437346-6985-4360-9934-19ef95ef8cd0)

## Chain of Thought 实战
**项目需求**：在这个示例中，你正在开发一个 AI 运营助手，我们要展示 AI 如何根据用户的需求推理和生成答案。然后，AI 根据当前的用户请求进行推理，提供了具体的花卉建议并解释了为什么选择这些建议。

在这个过程中，AI 需要理解客户的需求之后，按部就班的思考，然后给出最符合逻辑的回答。

### CoT 的模板设计
![image](https://github.com/user-attachments/assets/f6d2fa91-fb48-40e3-b6fb-5fca0e8174b5)

AI 的模板开始于对其角色的阐述，并给出了一些先前的对话示例（Few-Shot Learning）来帮助 AI 理解如何处理这种类型的请求。这些示例展示了 AI 如何根据思维链进行思考，给出深思熟虑之后的答案。

程序的完整代码如下：
```
# 设置环境变量和API密钥
import os
os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

# 创建聊天模型
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)

# 设定 AI 的角色和目标
role_template = "你是一个为花店电商公司工作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定"

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
作为一个为花店电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。 

我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一种象征爱情的花。
  AI：首先，我理解你正在寻找一种可以象征爱情的花。在许多文化中，红玫瑰被视为爱情的象征，这是因为它们的红色通常与热情和浓烈的感情联系在一起。因此，考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，同时也可以传达出强烈的感情，这是你在寻找的。

示例 2:
  人类：我想要一些独特和奇特的花。
  AI：从你的需求中，我理解你想要的是独一无二和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，它们在世界上的许多地方都被视为奢侈品和美的象征。因此，我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
"""
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])

prompt = chat_prompt.format_prompt(human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?").to_messages()

# 接收用户的询问，返回回答结果
response = llm(prompt)
print(response)
```

![image](https://github.com/user-attachments/assets/a46351a7-7e80-4200-b6cf-1096aac6ff61)

## Tree of Thought
![image](https://github.com/user-attachments/assets/db171bb9-bc82-4fa0-8eea-ed5b371f7339)

ToT 框架为每个任务定义具体的思维步骤和每个步骤的候选项数量。例如，要解决一个数学推理任务，先把它分解为 3 个思维步骤，并为每个步骤提出多个方案，并保留最优的 5 个候选方案。然后在多条思维路径中搜寻最优的解决方案。

这种方法的优势在于，模型可以通过观察和评估其自身的思维过程，更好地解决问题，而不仅仅是基于输入生成输出。这对于需要深度推理的复杂任务非常有用。此外，通过引入强化学习、集束搜索等技术，可以进一步提高搜索策略的性能，并让模型在解决新问题或面临未知情况时有更好的表现。

下面我们应用 ToT 的思想，给出一个鲜花运营方面的示例。
![image](https://github.com/user-attachments/assets/0770e666-c86d-43e1-a134-08a90322e09d)

这个例子，可以作为 FewShot 示例之一，传递给模型，让他学着实现 ToT。

通过**在具体的步骤中产生多条思考路径**，ToT 框架为解决复杂问题提供了一种新的方法，这种方法结合了语言模型的生成能力、搜索算法以及强化学习，以达到更好的效果。

## 总结
这节课我们介绍了 Chain of Thought（CoT，即“思维链”）和 Tree of Thoughts（ToT，即“思维树”）这两个非常有趣的概念，并探讨了如何利用它们引导大型语言模型进行更深入的推理。
- CoT 的核心思想是通过生成一系列中间推理步骤来增强模型的推理能力。在 Few-Shot CoT 和 Zero-Shot CoT 两种应用方法中，前者通过提供链式思考示例传递给模型，后者则直接告诉模型进行要按部就班的推理。
- ToT 进一步扩展了 CoT 的思想，通过搜索由连贯的语言序列组成的思维树来解决复杂问题。我通过一个鲜花选择的实例，展示了如何在实际应用中使用 ToT 框架。有朋友在 GitHub 上开了一个 [Repo](https://github.com/kyegomez/tree-of-thoughts)，专门给大家介绍 ToT 的应用方法和实例，他们还给出了几个非常简单的通用 ToT 提示语，就像下面这样。

![image](https://github.com/user-attachments/assets/59c1d7bb-77b7-4028-9f5f-fc04b0c5c63b)

## 思考题
1. 我们的 CoT 实战示例中使用的是 Few-Shot CoT 提示，请你把它换为 Zero-Shot CoT，跑一下程序，看看结果。
2. 请你设计一个你工作场景中的任务需求，然后用 ToT 让大语言模型帮你解决问题。

# 基础篇（11讲）06｜调用模型：使用OpenAI API还是微调开源Llama2/ChatGLM？
之前，我们花了两节课的内容讲透了提示工程的原理以及 LangChain 中的具体使用方式。今天，我们来着重讨论 Model I/O 中的第二个子模块，LLM。

![image](https://github.com/user-attachments/assets/b36dac78-2b1a-4d11-8a71-e465ff1fdf69)

## 大语言模型发展史
![image](https://github.com/user-attachments/assets/6c401447-7c8a-414a-a93f-867bf4c3ca60)

![image](https://github.com/user-attachments/assets/5c3f1a14-fc71-4a10-abaf-1aca1e9f284f)

## 预训练 + 微调的模式
![image](https://github.com/user-attachments/assets/a8f509e0-ff66-4dc0-a642-6131258d34aa)

图中的“具体任务”，其实也可以更换为“具体领域”。那么总结来说：
- **预训练**：在大规模无标注文本数据上进行模型的训练，目标是让模型学习自然语言的基础表达、上下文信息和语义知识，为后续任务提供一个通用的、丰富的语言表示基础。
- **微调**：在预训练模型的基础上，可以根据特定的下游任务对模型进行微调。现在你经常会听到各行各业的人说：我们的优势就是领域知识嘛！我们比不过国内外大模型，我们可以拿开源模型做垂直领域嘛！做垂类模型！—— 啥叫垂类？指的其实就是根据领域数据微调开源模型这件事儿。

![image](https://github.com/user-attachments/assets/55bfe96e-92bd-4870-bc64-1c27c14a96b6)

## 用 HuggingFace 跑开源模型
### 注册并安装 HuggingFace
- 第一步，还是要登录 [HuggingFace](https://huggingface.co/) 网站，并拿到专属于你的 Token（https://huggingface.co/settings/tokens，手机上有记录）。
- 第二步，用 pip install transformers 安装 HuggingFace Library。详见[这里](https://huggingface.co/docs/transformers/installation)。
- 第三步，在命令行中运行 huggingface-cli login，设置你的 API Token。
  ![image](https://github.com/user-attachments/assets/f276e127-2545-47e3-b63f-9e57d23c3c8d)
  当然，也可以在程序中设置你的 API Token，但是这不如在命令行中设置来得安全。
  ```
  # 导入HuggingFace API Token
   import os
   os.environ['HUGGINGFACEHUB_API_TOKEN'] = '你的HuggingFace API Token'
  ```
### 申请使用 Meta 的 Llama2 模型
在 HuggingFace 的 Model 中，找到 meta-llama/Llama-2-7b。注意，各种各样版本的 Llama2 模型多如牛毛，我们这里用的是最小的 7B 版。此外，还有 13b\70b\chat 版以及各种各样的非 Meta 官方版。

选择 meta-llama/Llama-2-7b 这个模型后，你能够看到这个模型的基本信息。如果你是第一次用 Llama，你需要申请 Access，因为我已经申请过了，所以屏幕中间有句话：“You have been granted access to this model”。从申请到批准，大概是几分钟的事儿。

### 通过 HuggingFace 调用 Llama
好，万事俱备，现在我们可以使用 HuggingFace 的 Transformers 库来调用 Llama！
```
# 导入必要的库
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 加载预训练的模型
# 使用 device_map 参数将模型自动加载到可用的硬件设备上，例如GPU
model = AutoModelForCausalLM.from_pretrained(
          "meta-llama/Llama-2-7b-chat-hf", 
          device_map = 'auto')  

# 定义一个提示，希望模型基于此提示生成故事
prompt = "请给我讲个玫瑰的爱情故事?"

# 使用分词器将提示转化为模型可以理解的格式，并将其移动到GPU上
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 使用模型生成文本，设置最大生成令牌数为2000
outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

# 将生成的令牌解码成文本，并跳过任何特殊的令牌，例如[CLS], [SEP]等
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的响应
print(response)
```

这段程序是一个很典型的 HuggingFace 的 Transformers 库的用例，该库提供了大量预训练的模型和相关的工具。
- 导入 AutoTokenizer：这是一个用于自动加载预训练模型的相关分词器的工具。分词器负责将文本转化为模型可以理解的数字格式。
- 导入 AutoModelForCausalLM：这是用于加载因果语言模型（用于文本生成）的工具。
- 使用 from_pretrained 方法来加载预训练的分词器和模型。其中，device_map = 'auto' 是为了自动地将模型加载到可用的设备上，例如 GPU。
- 然后，给定一个提示（prompt）："请给我讲个玫瑰的爱情故事?"，并使用分词器将该提示转换为模型可以接受的格式，return_tensors="pt" 表示返回 PyTorch 张量。语句中的 .to("cuda") 是 GPU 设备格式转换，因为我在 GPU 上跑程序，不用这个的话会报错，如果你使用 CPU，可以试一下删掉它。
- 最后使用模型的 .generate() 方法生成响应。max_new_tokens=2000 限制生成的文本的长度。使用分词器的 .decode() 方法将输出的数字转化回文本，并且跳过任何特殊的标记。

因为是在本地进行推理，耗时时间比较久。大概需要 30s～2min 产生结果。

![image](https://github.com/user-attachments/assets/942dfd3e-b66b-4da6-a76a-cf85e87b5314)

## LangChain 和 HuggingFace 的接口
### 通过 HuggingFace Hub
第一种集成方式，是通过 HuggingFace Hub。HuggingFace Hub 是一个开源模型中心化存储库，主要用于分享、协作和存储预训练模型、数据集以及相关组件。

我们给出一个 HuggingFace Hub 和 LangChain 集成的代码示例。
```
# 导入HuggingFace API Token
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '你的HuggingFace API Token'

# 导入必要的库
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# 初始化HF LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    #repo_id="meta-llama/Llama-2-7b-chat-hf",
)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """

# 创建Prompt          
prompt = PromptTemplate(template=template, input_variables=["question"])

# 调用LLM Chain --- 我们以后会详细讲LLM Chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
)

# 准备问题
question = "Rose is which type of flower?"

# 调用模型并返回结果
print(llm_chain.run(question))
```
![image](https://github.com/user-attachments/assets/4514dc30-83c9-42a3-996a-8cef3edac71f)

### 通过 HuggingFace Pipeline
既然 HuggingFace Hub 还不能完成 Llama-2 的测试，让我们来尝试另外一种方法，HuggingFace Pipeline。HuggingFace 的 Pipeline 是一种高级工具，它简化了多种常见自然语言处理（NLP）任务的使用流程，使得用户不需要深入了解模型细节，也能够很容易地利用预训练模型来做任务。

```
# 指定预训练模型的名称
model = "meta-llama/Llama-2-7b-chat-hf"

# 从预训练模型中加载词汇器
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# 创建一个文本生成的管道
import transformers
import torch
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length = 1000
)

# 创建HuggingFacePipeline实例
from langchain import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline = pipeline, 
                          model_kwargs = {'temperature':0})

# 定义输入模板，该模板用于生成花束的描述
template = """
              为以下的花束生成一个详细且吸引人的描述：
              花束的详细信息：
              ```{flower_details}```
           """

# 使用模板创建提示
from langchain import PromptTemplate,  LLMChain
prompt = PromptTemplate(template=template, 
                     input_variables=["flower_details"])

# 创建LLMChain实例
from langchain import PromptTemplate
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 需要生成描述的花束的详细信息
flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"

# 打印生成的花束描述
print(llm_chain.run(flower_details))
```

![image](https://github.com/user-attachments/assets/1bf7e5b3-9d6a-4a94-bd1b-a5ccf8f7a97b)

至此，通过 HuggingFace 接口调用各种开源模型的尝试成功结束。下面，我们进行最后一个测试，看看 LangChain 到底能否直接调用本地模型。

## 用 LangChain 调用自定义语言模型
![image](https://github.com/user-attachments/assets/eaee1c5b-e54c-454d-9fa5-7093c6c67455)

让我们先从 HuggingFace 的这里，下载一个 llama-2-7b-chat.ggmlv3.q4_K_S.bin 模型，并保存在本地。

然后，为了使用 llama-2-7b-chat.ggmlv3.q4_K_S.bin 这个模型，你需要安装 pip install llama-cpp-python 这个包。

```
# 导入需要的库
from llama_cpp import Llama
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM

# 模型的名称和路径常量
MODEL_NAME = 'llama-2-7b-chat.ggmlv3.q4_K_S.bin'
MODEL_PATH = '/home/huangj/03_Llama/'

# 自定义的LLM类，继承自基础LLM类
class CustomLLM(LLM):
    model_name = MODEL_NAME

    # 该方法使用Llama库调用模型生成回复
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt) + 5
        # 初始化Llama模型，指定模型路径和线程数
        llm = Llama(model_path=MODEL_PATH+MODEL_NAME, n_threads=4)
        # 使用Llama模型生成回复
        response = llm(f"Q: {prompt} A: ", max_tokens=256)
        
        # 从返回的回复中提取文本部分
        output = response['choices'][0]['text'].replace('A: ', '').strip()

        # 返回生成的回复，同时剔除了问题部分和额外字符
        return output[prompt_length:]

    # 返回模型的标识参数，这里只是返回模型的名称
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    # 返回模型的类型，这里是"custom"
    @property
    def _llm_type(self) -> str:
        return "custom"
    

# 初始化自定义LLM类
llm = CustomLLM()

# 使用自定义LLM生成一个回复
result = llm("昨天有一个客户抱怨他买了花给女朋友之后，两天花就枯了，你说作为客服我应该怎么解释？")

# 打印生成的回复
print(result)
```

## 思考题
![image](https://github.com/user-attachments/assets/de76c51d-aeb4-4aa3-8ce9-9613ca66f73d)

# 基础篇（11讲）07｜输出解析：用OutputParser生成鲜花推荐列表
今天我要带着你深入研究一下 LangChain 中的输出解析器，并用一个新的解析器——Pydantic 解析器来重构第 4 课中的程序。这节课也是模型 I/O 框架的最后一讲。

## LangChain 中的输出解析器
输出解析器是**一种专用于处理和构建语言模型响应的类**。一个基本的输出解析器类通常需要实现两个核心方法。
- get_format_instructions：这个方法需要返回一个字符串，用于指导如何格式化语言模型的输出，告诉它应该如何组织并构建它的回答。
- parse：这个方法接收一个字符串（也就是语言模型的输出）并将其解析为特定的数据结构或格式。这一步通常用于确保模型的输出符合我们的预期，并且能够以我们需要的形式进行后续处理。

还有一个可选的方法。

- parse_with_prompt：这个方法接收一个字符串（也就是语言模型的输出）和一个提示（用于生成这个输出的提示），并将其解析为特定的数据结构。这样，你可以根据原始提示来修正或重新解析模型的输出，确保输出的信息更加准确和贴合要求。

在 LangChain 中，通过实现 get_format_instructions、parse 和 parse_with_prompt 这些方法，针对不同的使用场景和目标，设计了各种输出解析器。让我们来逐一认识一下。
1. 列表解析器（List Parser）：这个解析器用于处理模型生成的输出，当需要模型的输出是一个列表的时候使用。例如，如果你询问模型“列出所有鲜花的库存”，模型的回答应该是一个列表。
2. 日期时间解析器（Datetime Parser）：这个解析器用于处理日期和时间相关的输出，确保模型的输出是正确的日期或时间格式。
3. 枚举解析器（Enum Parser）：这个解析器用于处理预定义的一组值，当模型的输出应该是这组预定义值之一时使用。例如，如果你定义了一个问题的答案只能是“是”或“否”，那么枚举解析器可以确保模型的回答是这两个选项之一。
4. 结构化输出解析器（Structured Output Parser）：这个解析器用于处理复杂的、结构化的输出。如果你的应用需要模型生成具有特定结构的复杂回答（例如一份报告、一篇文章等），那么可以使用结构化输出解析器来实现。
5. Pydantic（JSON）解析器：这个解析器用于处理模型的输出，当模型的输出应该是一个符合特定格式的 JSON 对象时使用。它使用 Pydantic 库，这是一个数据验证库，可以用于构建复杂的数据模型，并确保模型的输出符合预期的数据模型。
6. 自动修复解析器（Auto-Fixing Parser）：这个解析器可以自动修复某些常见的模型输出错误。例如，如果模型的输出应该是一段文本，但是模型返回了一段包含语法或拼写错误的文本，自动修复解析器可以自动纠正这些错误。
7. 重试解析器（RetryWithErrorOutputParser）：这个解析器用于在模型的初次输出不符合预期时，尝试修复或重新生成新的输出。例如，如果模型的输出应该是一个日期，但是模型返回了一个字符串，那么重试解析器可以重新提示模型生成正确的日期格式。

上面的各种解析器中，前三种很容易理解，而结构化输出解析器你已经用过了。所以接下来我们重点讲一讲 Pydantic（JSON）解析器、自动修复解析器和重试解析器。

### Pydantic（JSON）解析器实战
1. 第一步：创建模型实例
   先通过环境变量设置 OpenAI API 密钥，然后使用 LangChain 库创建了一个 OpenAI 的模型实例。这里我们仍然选择了 text-davinci-003 作为大语言模型。
   ```
   # ------Part 1
   # 设置OpenAI API密钥
   import os
   os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
   
   # 创建模型实例
   from langchain import OpenAI
   model = OpenAI(model_name='gpt-3.5-turbo-instruct')
   ```
2. 第二步：定义输出数据的格式
   先创建了一个空的 DataFrame，用于存储从模型生成的描述。接下来，通过一个名为 FlowerDescription 的 Pydantic BaseModel 类，定义了期望的数据格式（也就是数据的结构）。
   ```
   # ------Part 2
   # 创建一个空的DataFrame用于存储结果
   import pandas as pd
   df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])
   
   # 数据准备
   flowers = ["玫瑰", "百合", "康乃馨"]
   prices = ["50", "30", "20"]
   
   # 定义我们想要接收的数据格式
   from pydantic import BaseModel, Field
   class FlowerDescription(BaseModel):
       flower_type: str = Field(description="鲜花的种类")
       price: int = Field(description="鲜花的价格")
       description: str = Field(description="鲜花的描述文案")
       reason: str = Field(description="为什么要这样写这个文案")
   ```
   在这里我们用到了负责数据格式验证的 Pydantic 库来创建带有类型注解的类 FlowerDescription，它可以自动验证输入数据，确保输入数据符合你指定的类型和其他验证条件。
3. 第三步：创建输出解析器
   先使用 LangChain 库中的 PydanticOutputParser 创建了输出解析器，该解析器将用于解析模型的输出，以确保其符合 FlowerDescription 的格式。然后，使用解析器的 get_format_instructions 方法获取了输出格式的指示。
   ```
   # ------Part 3
   # 创建输出解析器
   from langchain.output_parsers import PydanticOutputParser
   output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)
   
   # 获取输出格式指示
   format_instructions = output_parser.get_format_instructions()
   # 打印提示
   print("输出格式：",format_instructions)
   ```
   程序输出如下：
   ```
   输出格式： The output should be formatted as a JSON instance that conforms to the JSON schema below.

   As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}
   the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
   
   Here is the output schema:
   
   {"properties": {"flower_type": {"title": "Flower Type", "description": "\u9c9c\u82b1\u7684\u79cd\u7c7b", "type": "string"}, "price": {"title": "Price", "description": "\u9c9c\u82b1\u7684\u4ef7\u683c", "type": "integer"}, "description": {"title": "Description", "description": "\u9c9c\u82b1\u7684\u63cf\u8ff0\u6587\u6848", "type": "string"}, "reason": {"title": "Reason", "description": "\u4e3a\u4ec0\u4e48\u8981\u8fd9\u6837\u5199\u8fd9\u4e2a\u6587\u6848", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}
   ```
   上面这个输出，这部分是通过 output_parser.get_format_instructions() 方法生成的，这是 Pydantic (JSON) 解析器的核心价值，值得你好好研究研究。同时它也算得上是一个很清晰的提示模板，能够为模型提供良好的指导，描述了模型输出应该符合的格式。（其中 description 中的中文被转成了 UTF-8 编码。）
   它指示模型输出 JSON Schema 的形式，定义了一个有效的输出应该包含哪些字段，以及这些字段的数据类型。例如，它指定了 "flower_type" 字段应该是字符串类型，"price" 字段应该是整数类型。这个指示中还提供了一个例子，说明了什么是一个格式良好的输出。
   下面，我们会把这个内容也传输到模型的提示中，**让输入模型的提示和输出解析器的要求相互吻合，前后就呼应得上**。
4. 第四步：创建提示模板
   定义了一个提示模板，该模板将用于为模型生成输入提示。模板中包含了你需要模型填充的变量（如价格和花的种类），以及之前获取的输出格式指示。
   ```
   # ------Part 4
   # 创建提示模板
   from langchain import PromptTemplate
   prompt_template = """您是一位专业的鲜花店文案撰写员。
   对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
   {format_instructions}"""
   
   # 根据模板创建提示，同时在提示中加入输出解析器的说明
   prompt = PromptTemplate.from_template(prompt_template, 
          partial_variables={"format_instructions": format_instructions}) 
   
   # 打印提示
   print("提示：", prompt)
   ```
5. 第五步：生成提示，传入模型并解析输出
   这部分是程序的主体，我们循环来处理所有的花和它们的价格。对于每种花，都根据提示模板创建了输入，然后获取模型的输出。然后使用之前创建的解析器来解析这个输出，并将解析后的输出添加到 DataFrame 中。最后，你打印出了所有的结果，并且可以选择将其保存到 CSV 文件中。
   ```
   # ------Part 5
   for flower, price in zip(flowers, prices):
       # 根据提示准备模型的输入
       input = prompt.format(flower=flower, price=price)
       # 打印提示
       print("提示：", input)
   
       # 获取模型的输出
       output = model(input)
   
       # 解析模型的输出
       parsed_output = output_parser.parse(output)
       parsed_output_dict = parsed_output.dict()  # 将Pydantic格式转换为字典
   
       # 将解析后的输出添加到DataFrame中
       df.loc[len(df)] = parsed_output.dict()
   
   # 打印字典
   print("输出的数据：", df.to_dict(orient='records'))
   ```
   这一步中，你使用你的模型和输入提示（由鲜花种类和价格组成）生成了一个具体鲜花的文案需求（同时带有格式描述），然后传递给大模型，也就是说，提示模板中的 flower 和 price，此时都被具体的花取代了，而且模板中的 {format_instructions}，也被替换成了 JSON Schema 中指明的格式信息。
   ![image](https://github.com/user-attachments/assets/8a891b29-8eff-478c-9128-b75e128ded41)
   ![image](https://github.com/user-attachments/assets/2d8e06e5-e6f0-4a42-8996-db24999da111)

### 自动修复解析器（OutputFixingParser）实战
首先，让我们来设计一个解析时出现的错误。
```
# 导入所需要的库和模块
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 使用Pydantic创建一个数据格式，表示花
class Flower(BaseModel):
    name: str = Field(description="name of a flower")
    colors: List[str] = Field(description="the colors of this flower")
# 定义一个用于获取某种花的颜色列表的查询
flower_query = "Generate the charaters for a random flower."

# 定义一个格式不正确的输出
misformatted = "{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"

# 创建一个用于解析输出的Pydantic解析器，此处希望解析为Flower格式
parser = PydanticOutputParser(pydantic_object=Flower)
# 使用Pydantic解析器解析不正确的输出
parser.parse(misformatted)
```
![image](https://github.com/user-attachments/assets/1f4f3feb-aaea-4f3c-b803-ae83cfd4c6f6)

![image](https://github.com/user-attachments/assets/9536c0c7-6dbc-477f-a017-dc477e5d0ab5)

### 重试解析器（RetryWithErrorOutputParser）实战
![image](https://github.com/user-attachments/assets/93875b1b-b556-430f-a7a3-3cf164228441)

首先还是设计一个解析过程中的错误。
```
# 定义一个模板字符串，这个模板将用于生成提问
template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""

# 定义一个Pydantic数据格式，它描述了一个"行动"类及其属性
from pydantic import BaseModel, Field
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

# 使用Pydantic格式Action来初始化一个输出解析器
from langchain.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Action)

# 定义一个提示模板，它将用于向模型提问
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="What are the colors of Orchid?")

# 定义一个错误格式的字符串
bad_response = '{"action": "search"}'
parser.parse(bad_response) # 如果直接解析，它会引发一个错误
```

![image](https://github.com/user-attachments/assets/8a8d3dea-49f2-40b2-8f76-b69949c0ef89)

![image](https://github.com/user-attachments/assets/a3ff404d-0c9e-4bdf-815e-351398bc287b)

## 总结
结构化解析器和 Pydantic 解析器都旨在从大型语言模型中获取格式化的输出。结构化解析器更适合简单的文本响应，而 Pydantic 解析器则提供了对复杂数据结构和类型的支持。选择哪种解析器取决于应用的具体需求和输出的复杂性。

自动修复解析器主要适用于纠正小的格式错误，它更加“被动”，仅在原始输出出现问题时进行修复。重试解析器则可以处理更复杂的问题，包括格式错误和内容缺失。它通过重新与模型交互，使得输出更加完整和符合预期。

在选择哪种解析器时，需要考虑具体的应用场景。如果仅面临格式问题，自动修复解析器可能足够；但如果输出的完整性和准确性至关重要，那么重试解析器可能是更好的选择。

## 思考题
1. 到目前为止，我们已经使用了哪些 LangChain 输出解析器？请你说一说它们的用法和异同。同时也请你尝试使用其他类型的输出解析器，并把代码与大家分享。
2. 为什么大模型能够返回 JSON 格式的数据，输出解析器用了什么魔法让大模型做到了这一点？
3. 自动修复解析器的“修复”功能具体来说是怎样实现的？请做 debug，研究一下 LangChain 在调用大模型之前如何设计“提示”。
4. 重试解析器的原理是什么？它主要实现了解析器类的哪个可选方法？

# 基础篇（11讲）08｜链（上）：写一篇完美鲜花推文？用SequencialChain链接不同的组件
# 基础篇（11讲）09｜链（下）：想学“育花”还是“插花”？用RouterChain确定客户意图
# 基础篇（11讲）10｜记忆：通过Memory记住客户上次买花时的对话细节
# 基础篇（11讲）11｜代理（上）：ReAct框架，推理与行动的协同
# 基础篇（11讲）12｜代理（中）：AgentExecutor究竟是怎样驱动模型和工具完成任务的？
# 基础篇（11讲）13｜代理（下）：结构化工具对话、Self-Ask with Search以及Plan and execute代理

# 应用篇（6讲）14｜工具和工具箱：LangChain中的Tool和Toolkits一览
# 应用篇（6讲）15｜检索增强生成：通过RAG助力鲜花运营
# 应用篇（6讲）16｜连接数据库：通过链和代理查询鲜花信息
# 应用篇（6讲）17｜回调函数：在AI应用中引入异步通信机制
# 应用篇（6讲）18｜CAMEL：通过角色扮演脑暴一个鲜花营销方案
# 应用篇（6讲）19｜BabyAGI：根据气候变化自动制定鲜花存储策略

# 实战篇（4讲）20｜部署一个鲜花网络电商的人脉工具（上）
# 实战篇（4讲）21｜部署一个鲜花网络电商的人脉工具（下）
# 实战篇（4讲）22｜易速鲜花聊天客服机器人的开发（上）
# 实战篇（4讲）23｜易速鲜花聊天客服机器人的开发（下）

# 结束语 & 结课测试 (2讲) 结课测试｜来赴一场满分之约
# 结束语 & 结课测试 (2讲) 结束语｜人生的价值就在于创造

# 加餐分享 (2讲) 直播加餐｜LangChain表达式语言LCEL初探
# 加餐分享 (2讲) 直播加餐｜LangChain的Tracing、Debug和LangSmith平台解读
