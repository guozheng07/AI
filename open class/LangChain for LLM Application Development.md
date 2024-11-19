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
### 导读
- LLMChain：这是LangChain中最基本的链类型。它将提示模板、语言模型和（可选的）输出解析器组合在一起。
- Sequential Chains：用于将多个链按顺序连接起来的工具。
  - SimpleSequentialChain：这是最简单的序列链，其中**每个步骤的输出直接作为下一个步骤的输入。适用于链中只有一个 input 和 一个 output**。
  - SequentialChain：这是一个更灵活的序列链，**允许指定每个步骤的输入和输出。适用于链中有多个 input 和 多个 output**。
- Router Chain：这是一种**可以根据输入动态选择使用哪个链**的工具。

![image](https://github.com/user-attachments/assets/2776b051-c325-40fb-9daf-7e44b6bd3377)
![image](https://github.com/user-attachments/assets/c5d1eb95-3954-461f-a350-5762839db5d3)
![image](https://github.com/user-attachments/assets/c5079094-9731-4b64-9d95-45569ee85960)

```
import warnings
warnings.filterwarnings('ignore')
```

```
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
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
#!pip install pandas
```

```
import pandas as pd
df = pd.read_csv('Data.csv')
```

![image](https://github.com/user-attachments/assets/1c6b1028-deb0-49d1-a14c-fd46772d9b7d)

### LLMChain
```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
```

```
llm = ChatOpenAI(temperature=0.9, model=llm_model)
```

```
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
```

使用 LLMChain 将对话模型和提示模版组合在一起，形成1个链：
```
chain = LLMChain(llm=llm, prompt=prompt)
```

![image](https://github.com/user-attachments/assets/0099c7ab-c980-4865-9594-9345485ac275)

### SimpleSequentialChain
```
from langchain.chains import SimpleSequentialChain
```
创建对话模型和链1：
```
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)
```
创建链2：
```
# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```
通过 SimpleSequentialChain，将链1和链2组合（链1输出的公司名称输入给链2）
```
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )
```

![image](https://github.com/user-attachments/assets/d88bb4cf-51a6-40c1-ae09-ee290a74154a)

### SequentialChain
```
from langchain.chains import SequentialChain
```

示例：将4个链进行组合

#### 方式一：手动调用链，每次通过调用 LLMChain 的 prompt 中指定输入，output_key 指定输出
创建对话模型和链1：
```
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: 输入为 Review，用 output_key 指定输出为 English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )
```
创建链2：
```
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: 输入为链1的输出 English_Review，输出为 summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )
```
创建链3：
```
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: 输入为 Review，输出为 language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )
```
创建链4：
```
# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: 输入为链2的输出 summary 和链3的输出 language，输出为 followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )
```

#### 方式二：SequentialChain 的 chains 指定链的调用关系，input_variables 定义初始输入，output_variables 定义中间输出（只要变量名匹配）
```
# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)
```

输出结果：
![image](https://github.com/user-attachments/assets/13623b77-6582-42c3-8208-c6fd78046259)

### Router Chain
4个模版
```
# 处理物理问题
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

# 处理数学问题
math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

# 处理历史问题
history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

# 处理计算机科学问题
computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""
```
4个模版的描述
```
prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]
```

```
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
```

```
llm = ChatOpenAI(temperature=0, model=llm_model)
```

**步骤1：获取目标链信息**
```
# 目标链集合
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
```
**步骤2：设置默认链**
```
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
```
**步骤3：在不同链之前定义 LLM to route 使用的模版（根据输入内容选择合适的链），包含要完成的任务的说明，以及输出应采用特定格式**
~~~
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""
~~~
**步骤4：创建 Router 链**
```
# 通过 format 创建完成的路由器模版，代替上述定义的目标链集合 destinations
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
# 基于路由器模版创建 prompt 模版
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), # RouterOutputParser 解析路由链的输出，确定下一步应该执行哪个链或操作
)
# 通过 LLMRouterChain 创建 Router 链
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
```
**步骤5：通过 MultiPromptChain 创建完整链条**（Router 链、目标链和默认链）
```
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
```

问一个物理问题，找到物理链：
![image](https://github.com/user-attachments/assets/63babfb8-53a0-4805-b4c0-14eecc483383)

问一个数学问题，找到数学链：
![image](https://github.com/user-attachments/assets/b99f81f6-a362-4dc5-b122-81e821302914)

问一个生物问题，找到默认链：
![image](https://github.com/user-attachments/assets/13bac869-8df9-46b6-8c96-6ef7bd79dd40)

## Question and Answer
## LangChain: Q&A over Documents
```
#pip install --upgrade langchain
```

```
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
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
# 导入了 RetrievalQA 链，它是一个用于问答任务的高级抽象。
from langchain.chains import RetrievalQA
# 导入了 ChatOpenAI 模型，它是一个封装了 OpenAI 聊天模型的接口。
from langchain.chat_models import ChatOpenAI
# 导入了 CSVLoader，用于加载 CSV 文件中的数据。
from langchain.document_loaders import CSVLoader
# 导入了 DocArrayInMemorySearch，它是一个内存中的向量存储，用于存储和检索文档嵌入。
from langchain.vectorstores import DocArrayInMemorySearch
# 导入了用于在 Jupyter notebook 中显示内容的函数。
from IPython.display import display, Markdown
# 导入了 OpenAI 语言模型。
from langchain.llms import OpenAI
```
解析文件：
```
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
```
导入一个索引 VectorstoreIndexCreator：
```
from langchain.indexes import VectorstoreIndexCreator
```

```
# pip install docarray
```
创建一个 vector store：
```
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
```
询问的问题：
```
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one.
```
使用 index.query 创建响应并传入 query 查询
```
llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)
```

![image](https://github.com/user-attachments/assets/4c0bdfd3-38ae-401a-9a02-57f5c6778b71)

**上述问答系统的实现依赖于嵌入向量和向量存储**：

![image](https://github.com/user-attachments/assets/14637556-32b6-427d-aa3e-2cac48f7f4d8)

![image](https://github.com/user-attachments/assets/2d9fad10-e6eb-4395-9a3a-c5e83fe42646)

![image](https://github.com/user-attachments/assets/ecac3600-8cfc-419e-b215-b6c9c8674dbb)

下面一步步实现该过程：

### Step By Step
**步骤1：解析文件**
```
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)
```

**步骤2：加载文件内容**
```
docs = loader.load()
```
查看文件内容（文件很小，不用做分块）：
```
docs[0]
```
![image](https://github.com/user-attachments/assets/fc5468d1-87cc-4164-a4b1-bffb028053ab)

**步骤3：引入嵌入向量**
```
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

示例：使用 embed_query 创建一个特定文本的嵌入（观察发生了什么）
```
embed = embeddings.embed_query("Hi my name is Harrison")
```

查看整体嵌入内容的长度，及本次嵌入的内容（5个词汇对应 embed 中最后5个元素）
![image](https://github.com/user-attachments/assets/15315146-58f0-4c19-95e8-32a56dde101d)

**步骤4：通过 DocArrayInMemorySearch 进行向量存储**
```
# 对文档进行整体向量存储
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
```
**步骤5：使用向量存储来查找文本片段**
```
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
```

查看结果：返回了4个文档
![image](https://github.com/user-attachments/assets/caf7b528-45a6-4ba7-8451-d678eca85c78)

**步骤6：创建一个基于该向量存储的检索器**
```
retriever = db.as_retriever()
```

**步骤7：创建 chat 模型**
```
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
```

**步骤8：将响应的文档合并成一段文本，并进行 chat 模型输出**
```
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 
```

![image](https://github.com/user-attachments/assets/8377b9cd-cf84-4af3-bab5-4a58d396c6db)

**步骤9：创建了一个基于"stuff"链类型的检索问答（RetrievalQA）系统**
```
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    # "stuff"链类型将所有相关文档合并成一个单一的上下文，然后将这个上下文和问题一起传递给语言模型
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```

![image](https://github.com/user-attachments/assets/9d265817-9442-4cd4-911c-169acb59fa0b)

```
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

display(Markdown(response))
```

### 扩展
上述文档较小，没有进行分块，因此用了 **stuff** 方法。**如果文档有分块，怎么在不同块中获取答案**？用以下方法：
- **Map_reduce**
- **Refine**
- **Map_rerank**

![image](https://github.com/user-attachments/assets/cedbfc29-3071-4917-9d8c-b93d2271be22)

- stuff
  - **最简单的策略，将所有文档内容合并成一个大的上下文**。
  - 适用于处理少量或较短的文档。
- Map_reduce
  - **这种策略首先对每个文档单独应用LLM，然后将结果合并**。
  - 适用于处理大量文档或长文档。
- Refine
  - **这种策略逐步处理每个文档，不断完善答案**，例如申诉原因分类（聚合成不同的一级、二级原因）。
  - 适用于需要综合考虑多个文档的情况。
- Map_rerank
  - **这种策略对每个文档单独生成答案，然后对结果进行重新排序**。
  - 适用于需要从多个可能的答案中选择最佳答案的情况。

## Evaluation
### 导读
评估应用程序的表现：
- Example generation
- Manual evaluation (and debuging)
- LLM-assisted evaluation
- LangChain evaluation platform

```
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
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

### Create our QandA application
```
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
```

```
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()
```

```
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
```

```
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
```

上述内容在上节课讲过，创建一个 qa 系统，下面列举 Example generation（示例生成）的一些常见方法。

### Example generation

#### 方法1：Coming up with test datapoints
手动：查看测试数据点。
![image](https://github.com/user-attachments/assets/c4fe9af7-8718-4971-8465-67d27ca19980)

#### 方法2：Hard-coded examples
手动：列举难的示例，基于生成的答案进行对比。
```
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
```

#### 方法3：LLM-Generated examples
自动：使用 QAGenerateChain 链，将从每个文档创建一个问题答案对。
```
from langchain.evaluation.qa import QAGenerateChain

# 将模型本身传入 QAGenerateChain 链
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

# 创建一堆例子
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
```

![image](https://github.com/user-attachments/assets/e6d01323-4239-45d2-89c3-e88457f3ff59)

![image](https://github.com/user-attachments/assets/e3eb58d4-8f91-4c79-bdf9-fda7d3bae6a0)

#### 方法4：Combine examples
```
examples += new_examples
```

```
qa.run(examples[0]["query"])
```

### Manual Evaluation
```
import langchain
# 设置为 True 可以进行 debug（可以看到 token 信息）
langchain.debug = False
```

![image](https://github.com/user-attachments/assets/a8c53eee-9c6a-4df5-b4c6-0bd4385414b0)

### LLM assisted evaluation
```
predictions = qa.apply(examples)
```

```
from langchain.evaluation.qa import QAEvalChain
```

```
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)
```

传递示例，进行预测
```
graded_outputs = eval_chain.evaluate(examples, predictions)
```

```
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```
![image](https://github.com/user-attachments/assets/b9d68c6d-7e0f-4f62-ad30-1c9c9b51a0f1)

## Agents
### 导读
- Using built in LangChain tools: DuckDuckGo search and Wikipedia（主要用作外部知识源或工具，用于增强AI模型的信息检索能力）
  - **DuckDuckGo：进行网络搜索，获取实时信息**
    - 提供隐私保护的搜索结果
    - 可以获取最新的网络信息
    - 适用于需要实时或广泛信息的查询
  - **Wikipedia：检索特定主题的百科知识**
    - 提供结构化的百科全书信息
    - 适合获取概念解释、历史背景等深度知识
    - 信息相对可靠，但可能不是最新
- Defining your own tools

```
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
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

### Built-in LangChain tools
```
#!pip install -U wikipedia
```
**步骤1：引入依赖**
```
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
```
**步骤2：建立 chat 模型**
```
llm = ChatOpenAI(temperature=0, model=llm_model)
```
**步骤3：使用 load_tools 函数加载预定义的工具，tools 变量将包含加载的工具列表，可以在后续的 Agent 或 Chain 中使用**
```
# "llm-math": 这是一个数学计算工具，能够使用语言模型进行复杂的数学运算。
# "wikipedia": 这是维基百科查询工具，可以从维基百科获取信息。
tools = load_tools(["llm-math","wikipedia"], llm=llm)
```
**步骤4：创建和配置一个 agent 实例**
```
agent= initialize_agent(
    tools, # 工具
    llm, # 语言模型
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # 代理类型，这种类型的 Agent 能够在没有先前训练的情况下，根据工具描述来选择和使用工具。
    handle_parsing_errors=True, # 允许Agent处理解析错误，提高鲁棒性。
    verbose = True) # 启用详细输出模式，方便调试和观察Agent的决策过程。
```
**步骤5：提问**
![image](https://github.com/user-attachments/assets/18ae066b-e9ef-4644-b6df-5360fc1e6fe6)

### Wikipedia example
```
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 
```

### Python Agent
```
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
```

```
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
```

```
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
```

```
import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False
```

### Define your own tool
```
#!pip install DateTime
```

```
from langchain.agents import tool
from datetime import date
```

```
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
```

```
agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```

## Concluion
