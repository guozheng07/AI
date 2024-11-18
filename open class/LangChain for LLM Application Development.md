![image](https://github.com/user-attachments/assets/3d5075cc-d764-427e-8ea5-31166fc2b395)# 课程地址
https://learn.deeplearning.ai/courses/langchain/lesson/1/introduction

# 课程内容
## Introuction
![image](https://github.com/user-attachments/assets/a2eeec0c-6158-467f-bd77-53274d704d75)

![image](https://github.com/user-attachments/assets/1602f8d9-31df-4cc7-9f43-285426903370)

## Models, Prompt and parsers
### 概述
- Direct API calls to OpenAI
- API calls through LangChain:
  - Prompts
  - Models
  - Output parsers

### Get your OpenAI API Key
安装依赖
```
#!pip install python-dotenv
#!pip install openai
```

查找和加载环境变量，获取 OpenAI API的密钥
```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```

Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

获取 llm_model
```
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
```

### Chat API : OpenAI
Let's start with a direct API calls to OpenAI.

定义输入和输出
```
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]
```

示例1：测试1+1
![image](https://github.com/user-attachments/assets/6fabc3f1-23d9-4acf-880d-167d1f0d8583)

示例2：用指定的语气翻译邮件内容
![image](https://github.com/user-attachments/assets/460b5fb7-64c0-468b-92b6-d9bded6fa1a2)

### Chat API : LangChain
Let's try how we can do the same using LangChain.

安装依赖
```
#!pip install --upgrade langchain
```

#### 定义 model
```
from langchain.chat_models import ChatOpenAI

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)
chat
```

#### 定义 Prompt template
定义模版字符串
```
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
```
设置可复用的模版
```
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)
```
查看 prompt 的原始输入
```
prompt_template.messages[0].prompt
```
查看 prompt 中输入变量
```
prompt_template.messages[0].prompt.input_variables
```
设置 style 变量
```
customer_style = """American English \
in a calm and respectful tone
"""
```
设置 text 变量
```
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
```
输入最终的 prompt，获取输出
```
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
print(customer_response.content)
```

#### Output Parsers
示例：将客户评论作为输入，提取 gift、delivery_days、price_value 这3个字段，格式化输出为 json

**方法一：在提示词中指明输出格式**
```
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""
```

```
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)
```

```
messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)
```

打印输出类型，发现是 str（一个看着像 json 的长字符串，不是我们想要的效果）
```
type(response.content)

```
直接获取 gift 的值是报错，因为输出本质是一个字符串
```
# You will get an error by running this line of code 
# because'gift' is not a dictionary
# 'gift' is a string
response.content.get('gift')
```

**方法二：使用 langchain 的解析器（StructuredOutputParser 和 ResponseSchema ）格式化输出**
```
langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
```
定义响应结构
```
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
```
格式化输出
```
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
```
查看输出解析器的格式说明
```
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
```
新的评论模版（包括格式指令）&输入信息
```
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)
```
输出信息（json字符串）
```
response = chat(messages)
print(response.content)
```
使用 output_parser 进行解析，得到 json
```
output_dict = output_parser.parse(response.content)
output_dict
```
格式为 dict，可以正常获取 delivery_days 的值
```
type(output_dict)
output_dict.get('delivery_days')
```

## Memory
### 概述
langchain 提供以下用于来存储和积累对话的便捷内存：
- ConversationBufferMemory
  - 这是最简单的内存类型。
  - 它会存储所有的对话历史，不做任何过滤或压缩。
  - 适用于需要完整对话历史的场景，但在长对话中可能会导致内存占用过大。
- ConversationBufferWindowMemory
  - **这种内存类型只保存最近的 N 条对话**。
  - 它通过滑动窗口的方式管理对话历史，旧的对话会被丢弃。
  - 适用于只需要最近上下文的场景，可以有效控制内存使用。
- ConversationTokenBufferMemory
  - **这种内存类型基于 token 数量来限制存储的对话历史**。
  - 它会计算对话历史的 token 数，并在达到限制时删除最早的对话。
  - 适用于需要精确控制输入 token 数量的场景，特别是在使用有 token 限制的模型时。
- ConversationSummaryMemory
  - **这种内存类型会对对话历史进行摘要，也会限制 token 数量**。
  - 它使用语言模型来生成对话的摘要，而不是存储完整的对话历史。
  - 适用于需要长期记忆但又不想存储大量原始对话的场景。

### ConversationBufferMemory
```
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings('ignore')
```

```
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
```

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
```

```
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```

```
conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
```
第三轮对话时，可以记住前两轮的对话：
![image](https://github.com/user-attachments/assets/49caf06c-9811-4076-8d50-fbca2b066b4e)

```
print(memory.buffer)
```
![image](https://github.com/user-attachments/assets/8b6b7567-0ab2-4c4f-a97b-8ab27215b478)

```
memory.load_memory_variables({})
```
![image](https://github.com/user-attachments/assets/0cf46f2e-6c77-4ebe-87cd-29474a46c1dd)

使用 ConversationBufferMemory 创建新的内存，并使用 save_context 方法添加新的上下文到内存
```
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
```

![image](https://github.com/user-attachments/assets/48b922ea-e266-43c6-bb46-f73754ffdad4)

![image](https://github.com/user-attachments/assets/358699e4-6e68-4254-94c1-eb3506d33950)

### 总结
大语言模型本身是“无状态的”。
- 每轮对话都是独立的。

chatbots/聊天机器人拥有记忆是通过提供完成的对话作为上下文。

### ConversationBufferWindowMemory
```
from langchain.memory import ConversationBufferWindowMemory
```

```
# 参数1代表只记忆一次对话交流
memory = ConversationBufferWindowMemory(k=1)         
```

```
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
```
![image](https://github.com/user-attachments/assets/05f2bb07-0b4c-4e53-86fe-e150bc09cda7)

示例（第三轮对话忘记了第一轮对话中的名字）：
![image](https://github.com/user-attachments/assets/50c0eac8-7225-47e1-96a4-79af077f5725)

### ConversationTokenBufferMemory
```
#!pip install tiktoken
```

```
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model)
```

```
# 设置最大 token 是50
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
```
内存只保留了最新的对话内容（token 为50）
![image](https://github.com/user-attachments/assets/0257607a-21a2-4afd-af72-408deda8fdbf)

### ConversationSummaryMemory
```
from langchain.memory import ConversationSummaryBufferMemory
```

```
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

# token 数量为 100
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
```
三轮对话的 token 超出了 100，可以看到截取最大 token 为 100 的内容时，只保留了第三轮对话的输出，并生成了摘要（非原文）
![image](https://github.com/user-attachments/assets/98d81c2b-6d6f-4922-a1a1-db491a1ccdca)

建立会话，验证只保留了上述截图中 100 个 token 的上下文（verbose 设置为 true，可以看到具体信息）
```
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
conversation.predict(input="What would be a good demo to show?")
```
在回答前，记录的对话（之前的内容）和输出：
![image](https://github.com/user-attachments/assets/b9ba3c3d-a5d7-43f3-b491-6d246a70db6c)

回答后的对话为（新的内容）：
![image](https://github.com/user-attachments/assets/08811906-2dde-4db7-8994-dda698d7279a)

### 总结
1. 除了上述4种内存类型，langchain 实际上也支持其他内存类型：其中最强大的是**向量数据库**，使用该方式可以检索内存中最相关的文本块。
2. langchain 还支持**实体记忆**，可以记忆有关特定人员、特定其他实体的详细信息。

![image](https://github.com/user-attachments/assets/aab15c57-f72f-467e-a0b0-05ac9d065822)

## Chains

## Question and Answer

## Evaluation

## Agents

## Concluion
