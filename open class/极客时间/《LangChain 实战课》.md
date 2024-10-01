# 开篇词｜带你亲证AI应用开发的“奇点”时刻
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

# 01｜LangChain系统安装和快速入门
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
