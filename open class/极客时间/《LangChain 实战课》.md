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
# 基础篇（11讲）05｜提示工程（下）：用思维链和思维树提升模型思考质量
# 基础篇（11讲）06｜调用模型：使用OpenAI API还是微调开源Llama2/ChatGLM？
# 基础篇（11讲）07｜输出解析：用OutputParser生成鲜花推荐列表
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
