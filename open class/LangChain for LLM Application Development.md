# 课程地址
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
新的评论模版
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

```
response = chat(messages)
print(response.content)
```

```
output_dict = output_parser.parse(response.content)
output_dict
```

```
type(output_dict)
output_dict.get('delivery_days')
```

## Memory

## Chains

## Question and Answer

## Evaluation

## Agents

## Concluion
