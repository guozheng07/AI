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


## Memory

## Chains

## Question and Answer

## Evaluation

## Agents

## Concluion
