课程介绍

![image](https://github.com/user-attachments/assets/b42a7eca-367b-46bc-91a8-9b506b44cd95)

课程仓库

https://github.com/chaocai2001/gpt_in_practice

# 基础篇（4讲）01｜第一个大模型程序：Hello GPT
## 在微软云中部署服务
## 实战
- 与 GPT 对话最重要，最基本的 api -> ChatComleion，可传入以下参数：
  ![image](https://github.com/user-attachments/assets/72313688-7024-42d3-83ff-968d88d34712)
  - temperature：控制生成结果的随机性。
  - max_tokens：限制生成结果的 tokens 长度（正常结束时返回的finish_reason字段是stop，被强制截断是length）。
  - n：生成结果的个数，影响成本，**一般是1**。
- python 快速计算 token 长度的库：tiktoken
  ![image](https://github.com/user-attachments/assets/6ecf64ff-039d-4920-b23a-71d55842a337)

## 作业
让 GPT 把中文翻译成英文，并计算输入和输出的 token 长度

![image](https://github.com/user-attachments/assets/f6a6270b-c388-4e6b-8310-13c1ba43f22f)

总结：英语提示词，更省 token，产生的费用会更低。

# 基础篇（4讲）02｜提示词技巧：获得代码友好的回复
## 典型应用场景
### 意图识别
```
import openai
response = openai.ChatCompletion.create(
    engine=deployment, # 如果直接访问OpenAI GPT服务的同学，这里不要使用engine这个参数，要使用model，如： model=“gpt-4”
    model=model,
    temperature = 0,
    messages=[
        {"role": "system", "content": """
          Recognize the intent from the user's input 
         """},
        # {"role": "user", "content": "订明天早5点北京到上海的飞机"}
        {"role": "user", "content": "提醒我明早8点有会议"}
    ]
  )
print(response.choices[0].message.content)
```
### 生成SQL
```
import openai, os
from langchain.llms import OpenAI

system_prompt =  """  You are a software engineer, you can anwser the user request based on the given tables:
                  table “students“ with the columns [id, name, course_id, score] 
                  table "courses" with the columns [id, name] 
                  """

prompt = system_prompt

response = openai.ChatCompletion.create(
    engine=deployment, # 如果直接访问OpenAI GPT服务的同学，这里不要使用engine这个参数，要使用model，如： model=“gpt-4”
    model=model,
    temperature = 0,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "计算所有学生英语课程的平均成绩"},
    ],
    max_tokens = 500
  )
print(response.choices[0].message.content)
```
## 提示词技巧
![image](https://github.com/user-attachments/assets/b42dfdf0-d79f-431d-a83c-b57ea3cc0fe9)

### 规范输出的格式
```
import openai
response = openai.ChatCompletion.create(
    engine=deployment, # engine = "deployment_name".
    temperature = 0,
    # 修改了 messages
    messages=[
        {"role": "system", "content": """
          Recognize the intent from the user's input and format output as JSON string. 
        The output JSON string includes: "intention", "paramters" """},
        {"role": "user", "content": "提醒我明早8点有会议"}
    ]
  )
print(response.choices[0].message.content)
```

```
import openai, os
from langchain.llms import OpenAI

# 修改了 system_prompt
system_prompt =  """  You are a software engineer, you can write a SQL string as the anwser according to the user request 
               The user's requirement is based on the given tables:
                  table “students“ with the columns [id, name, course_id, score];
                  table "courses" with the columns [id, name]."""

prompt = system_prompt

response = openai.ChatCompletion.create(
    engine=deployment,
    model=model,
    temperature = 0,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "列出英语课程成绩大于80分的学生, 返回结果只包括SQL"},
    ],
    max_tokens = 500
  )
print(response.choices[0].message.content)
```
### 文本规范异常输出的格式
```
import openai, os
from langchain.llms import OpenAI

# 旧版提示词
system_prompt =  """  You are a software engineer, you can write a SQL string as the anwser according to the user request 
               The user's requirement is based on the given tables:
                  table “students“ with the columns [id, name, course_id, score];
                  table "courses" with the columns [id, name]."""

# 新版提示词
system_prompt_with_negative =  """  
You are a software engineer, you can write a SQL string as the anwser according to the user request.
Also, when you cannot create the SQL query for the user's request based on the given tables, please, only return "invalid request"
               The user's requirement is based on the given tables:
                  table “students“ with the columns [id, name, course_id, score];
                  table "courses" with the columns [id, name]."""

# prompt = system_prompt
prompt = system_prompt_with_negative

response = openai.ChatCompletion.create(
    engine=deployment,
    model=model,
    temperature = 0,
    messages=[
        {"role": "system", "content": prompt},
        // 没有年龄信息，会报错
        {"role": "user", "content": "列出年龄大于13的学生"}
    ],
    max_tokens = 500
  )
print(response.choices[0].message.content)
```
## 总结
![image](https://github.com/user-attachments/assets/0adddebf-40aa-4d61-98f6-1c84c755269f)

![image](https://github.com/user-attachments/assets/de2b5aab-d401-4553-b98a-68b647b901de)

## 作业
![image](https://github.com/user-attachments/assets/c1105b9f-17b4-46d6-b401-414e3dc6d91f)

```
import openai
def analyze_user_review(text):
    messages = []
    messages.append( {"role": "system", 
                      "content": """
                      You are an assistant. 
                      Please, analyze the user reviews according to the following instruction: 
                      If the review is postive, you should output 'Y', otherwise output 'N'
                      """})
    messages.append( {"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        engine=deployment, 
        messages=messages,
        temperature=0.5,
        max_tokens = 100
    )
    return response["choices"][0]["message"]["content"]
```

# 基础篇（4讲）03 | 初识LangChain：你的瑞士军刀
![image](https://github.com/user-attachments/assets/3c11af0d-cc8b-4293-ad11-50c9421212ff)

## 提示词模版
![image](https://github.com/user-attachments/assets/0f89d387-806d-4049-b600-7b95cd1ac164)

通过使用提示词模版，直接更换 product 的值即可，不用每次重复修改提示词。

## 与 API 交互
![image](https://github.com/user-attachments/assets/85f48e80-af96-4c62-a6c2-272507c243d8)

1. 不科学的做法：直接调用相应网站使用的接口（可能会被认为是爬虫）
  ```
  from langchain import PromptTemplate, OpenAI, LLMChain
  from langchain.chains import LLMRequestsChain
  from langchain.chat_models import AzureChatOpenAI
  # from langchain.chat_models import ChatOpenAI #直接访问OpenAI的GPT服务
  
  #llm = ChatOpenAI(model_name="gpt-4", temperature=0) #直接访问OpenAI的GPT服务
  llm = AzureChatOpenAI(deployment_name = deployment, model_name=model, temperature=0, max_tokens=200) #通过Azure的OpenAI服务
  
  def query_baidu(question):
        template = """Between >>> and <<< are the raw search result text from web.
        Extract the answer to the question '{query}' or say "not found" if the information is not contained.
        Use the format
        Extracted:<answer or "not found">
        >>> {requests_result} <<<
        Extracted:"""
  
        PROMPT = PromptTemplate(
            input_variables=["query", "requests_result"],
            template=template,
        )
  
        inputs = {
            "query": question,
            "url": "http://www.baidu.com/s?wd=" + question.replace(" ", "+")
        }
        requests_chain = LLMRequestsChain(llm_chain = LLMChain(llm=llm, prompt=PROMPT), output_key="query_info", verbose=True)
        res = requests_chain.run(inputs)
        return res
  
  ```

  执行：
  
  query_baidu("今天北京天气？")

2. 科学的做法：使用官方提供的 api，例如 google 提供的 api 来查询天气信息
  ```
  # import os
  # os.environ["SERPER_API_KEY"] = ""
  # https://serper.dev
  from langchain.utilities import GoogleSerperAPIWrapper
  def query_web(question):
      search = GoogleSerperAPIWrapper()
      return search.run(question)
  ```

  执行：
  
  query_web("今天北京天气？")

## 链式请求
![image](https://github.com/user-attachments/assets/426528ba-0bd0-4962-a6ed-317b2c815bc0)

示例：先总结，再翻译：

![image](https://github.com/user-attachments/assets/b22bb1ea-ec7f-4924-ac8d-438c591b1e98)

```
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import SequentialChain
from langchain.chat_models import AzureChatOpenAI
# from langchain.chat_models import ChatOpenAI #直接访问OpenAI的GPT服务

# llm = ChatOpenAI(model_name="gpt-4", temperature=0) #直接访问OpenAI的GPT服务
llm = AzureChatOpenAI(deployment_name = deployment, model_name=model, temperature=0, max_tokens=200)

summarizing_prompt_template = """
Summarize the following content into a sentence less than 20 words:
---
{content}

"""
summarizing_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(summarizing_prompt_template),
    output_key = "summary" # 前一个输出
)

translating_prompt_template = """
translate "{summary}" into Chinese: # 后一个输入

"""

translating_chain = LLMChain(
    llm = llm,
    prompt=PromptTemplate.from_template(translating_prompt_template),
    output_key = "translated"
)

overall_chain = SequentialChain(
    chains=[summarizing_chain, translating_chain],
    input_variables=["content"],
    output_variables=[ "summary","translated"],
    verbose=True
)
```

执行：
```
res = overall_chain("""
LangChain is a framework for developing applications powered by language models. It enables applications that are:

Data-aware: connect a language model to other sources of data
Agentic: allow a language model to interact with its environment
The main value props of LangChain are:

Components: abstractions for working with language models, along with a collection of implementations for each abstraction. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not
Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level tasks
Off-the-shelf chains make it easy to get started. For more complex applications and nuanced use-cases, components make it easy to customize existing chains or build new ones.
""")

print("summary:"+res["summary"])

print("中文:"+res["translated"])
```

## 作业
![image](https://github.com/user-attachments/assets/1d80f3be-4064-4026-8ad3-457b03663c88)

搜索：
```
from langchain.chains import LLMRequestsChain
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(deployment_name = deployment, model_name=model, temperature=0, max_tokens=200)
template = """Between >>> and <<< are the raw search result text from web.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
  input_variables=["query", "requests_result"],
  template=template,
)

query_chain = LLMRequestsChain(llm_chain = LLMChain(llm=llm, prompt=PROMPT), output_key="query_info")
```

翻译：
```
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import SequentialChain

translating_prompt_template = """
translate "{query_info}" into English:

"""

translating_chain = LLMChain(
    llm = llm,
    prompt=PromptTemplate.from_template(translating_prompt_template),
    output_key = "translated"
)
```

**通过 SequentialChain 将两个模型序列化组合/串联**（看https://v02.api.js.langchain.com/classes/langchain.chains.SequentialChain.html，这个 api 好像已经废弃了）
```
llm = AzureChatOpenAI(deployment_name = deployment, model_name=model, temperature=0, max_tokens=200)

def overall(question):
    inputs = {
      "query": question,
      "url": "http://www.baidu.com/s?wd=" + question.replace(" ", "+")
      # "url": "https://cn.bing.com/search?q=" + question.replace(" ", "+")
    }
    
    overall_chain = SequentialChain(
        chains=[query_chain, translating_chain],
        input_variables=["query","url"],
        output_variables=["translated"],
        verbose=True
    )
    
    return overall_chain(inputs)["translated"]

overall("北京今天天气")
```

# 基础篇（4讲）04｜保持会话状态：让ChatBot获得记忆
## 没有记忆的模型
GPT 是无状态的，示例：

![image](https://github.com/user-attachments/assets/39ccd43b-74a0-4b72-aa7b-5b718ac3e602)

## 利用 Gradio 快速构建原型/验证页面（可以保存历史对话记录）
一个比较好的工具：快速建立验证页面 & 记忆历史对话 & 生成链接在本地和公有地址使用：

![image](https://github.com/user-attachments/assets/43092678-9ec1-4360-9e3d-4b448df46078)

## 构建有状态的对话（手动组装，为 GPT 提供外部记忆/历史对话，实现一个有记忆的 ChatBot）

```
import openai
def get_response(msg):
    # print(msg)
    response = openai.ChatCompletion.create(
        engine=deployment, # engine = "deployment_name".
        messages=msg,
        temperature = 0.9, 
        max_tokens = 600
    )
    return response.choices[0].message.content
```

处理历史对话：
```
def history_to_prompt(chat_history): # 将对话内容保存在一个List里
    msg = [{"role": "system", "content": "You are an AI assistant."}]
    i = 0
    for round_trip in chat_history: # 将List里的内容，组成 ChatCompletion的 messages部分，{role，content} dict
        msg.append({"role": "user", "content": round_trip[0]})
        msg.append({"role": "assistant", "content": round_trip[1]})
    return msg

def respond(message, chat_history):
    his_msg = history_to_prompt(chat_history) # 将历史会话转化为 ChatCompletion 需要的 messages 格式
    his_msg.append({"role": "user", "content": message}) # 放入当前用户问题
    bot_message = get_response(his_msg)
    chat_history.append((message, bot_message)) # 将用户问题和返回保存到 历史记录 List
    return "", chat_history
```

```
import gradio as gr
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=480) # just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch(share=True)
```

## 构建有状态的对话（通过 langchain 的记忆功能 ConversationBufferWindowMemory，实现一个有记忆的 ChatBot）
```
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
# from langchain.chat_models import ChatOpenAI #直接访问OpenAI的GPT服务

#llm = ChatOpenAI(model_name="gpt-4", temperature=0) #直接访问OpenAI的GPT服务
llm = AzureChatOpenAI(deployment_name = deployment, model_name=model, temperature=0, max_tokens=200) #通过Azure的OpenAI服务


# 记忆最近10轮的对话（太久的对话参考意义可能不大，可以节省 token）
memory = ConversationBufferWindowMemory(k=10) 


def get_response(input):
    print("------------")
    print(memory.load_memory_variables({}))
    print("------------")
    conversation_with_memory = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=False
    )
    return conversation_with_memory.predict(input=input)
```

```
import gradio as gr
def respond(message, chat_history):
    bot_message = get_response(message)
    chat_history.append((message, bot_message))
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=300) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch(share=True) # 设为 true，可以在托管的主机上创建一个公共可访问的链接
```

## 作业
尝试 langchain 不同类型的 memory。

![image](https://github.com/user-attachments/assets/a3df4d4b-8e16-4c90-9a63-1593da6cd0f3)

使用 ConversationSummaryBufferMemory：
```
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

llm = AzureChatOpenAI(deployment_name="gpt4", temperature=0.3, max_tokens=1000,
                     streaming=True)

#memory = ConversationSummaryBufferMemory(k=10) 
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)

def get_response(input):
    print("------------")
    print(memory.load_memory_variables({}))
    print("------------")
    conversation_with_memory = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=False
    )
    return conversation_with_memory.predict(input=input)
```
```
import gradio as gr
def respond(message, chat_history):
        bot_message = get_response(message)
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #对话框
    msg = gr.Textbox(label="Prompt") #输入框
    btn = gr.Button("Submit") #提交按钮
    #提交
    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) 
gr.close_all()
demo.launch()
```

![image](https://github.com/user-attachments/assets/b2051b5f-9018-473c-a3ee-630c9e33d62d)

# 企业应用篇（8讲）05 | 对话启发式UI：交互方式的新思考
## 启发式交互
与大模型进行自然语法交流，直接生成页面，对传统的低代码实现有一定的冲击，带来了新的思考。

## Copilot（副驾驶，还需要人来把控）
处理问题工单：可以结合人为评论、AI自己的评论、历史相关工单的记录和日志等生成决策，进行自动执行或人工执行。

![image](https://github.com/user-attachments/assets/781ed77d-b1e8-4aad-a545-596ca3e6e89d)

![image](https://github.com/user-attachments/assets/0d00d2af-3812-43cf-9957-f2061651cc71)

## 可以利用的不仅是文字生成能力
![image](https://github.com/user-attachments/assets/7c555244-0c3f-4314-8cd8-d36d8645e816)

## 作业
![image](https://github.com/user-attachments/assets/84dedda8-f421-4e27-b06d-505e881c3a50)

一个企业应用的实践：利用 Kubernetes GPT 去不断扫描你的集群，给到你它发现的问题，并提供建议。

![image](https://github.com/user-attachments/assets/b692bd8c-c3a4-4aaa-8f06-8065bf1f2310)

# 企业应用篇（8讲）06 | Function Calling：让GPT学会使用工具
![image](https://github.com/user-attachments/assets/12a8116a-a6fe-41c3-a331-2d7c77b23498)

![image](https://github.com/user-attachments/assets/efa3f793-f679-4ce1-9a86-bb58ec7a177c)

![image](https://github.com/user-attachments/assets/a6683ac8-20f4-4a84-9b30-e908ef6ea613)

![image](https://github.com/user-attachments/assets/8b748100-904e-4282-af1d-19a98e706ef0)

```
deployment = "gpt4"
model = "gpt-4"

# 获得当前集群的状态
def get_current_cluster_state(cluster_name):
    print(f"cluster:{cluster_name}")
    return  """ERROR: Failed to pull image "chaocai/docker/dsp:latest"""

import openai
import json

# 定义一个函数：key是方法名，value是方法
funcs = {"get_current_cluster_state": get_current_cluster_state}

def run(input):
    msg=[{"role":"user","content":input}]
    ret = run_conversation(msg)
    return ret["content"] 

def run_conversation(msg):
    response = openai.ChatCompletion.create(
        engine=deployment,
        model=model,
        messages=msg,
        functions=[
            {
                "name": "get_current_cluster_state",
                "description": "Get the current state in a given cluster",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cluster_name": {
                            "type": "string",
                            "description": "the name of the cluster",
                        },
                        
                    },
                    "required": ["cluster_name"],
                },
            }
        ],
        function_call="auto",
    )

    message = response["choices"][0]["message"]
    print("----- message ----")
    print(message)
    print("----- message ----")
    # 如果不需要调用function，则直接返回结果
    if not message.get("function_call"):
        return message
    
    # 从返回值中获取调用的方法
    function_name = message["function_call"]["name"]
    # 从返回值中获取方法的参数值
    function_args = json.loads(message["function_call"]["arguments"])
    print(function_args)
    res = funcs[function_name](function_args["cluster_name"]) 
    message["content"]=None
    msg.append(message)
    msg.append({
                "role": "function",
                "name": function_name,
                "content": res,
            })
    # 再次调用 GPT
    return run_conversation(msg)
```

执行：
```
print(run("What's wrong with the cluster 'DSP'? And if there's an error, give me some suggestion. "))
```

一个大语言模型可以和你的企业环境进行交互，变成了一个 DevOps 工程师。

![image](https://github.com/user-attachments/assets/0e85fbae-69c6-4179-928a-be999cf600a5)

## 作业
![image](https://github.com/user-attachments/assets/1de0f5d8-fb23-4f65-ae8e-753aa1f8a0f4)

# 企业应用篇（8讲）07｜LangChain Agent：让GPT学会使用工具
# 企业应用篇（8讲）08｜In-context learning：学习解决特定任务
# 企业应用篇（8讲）09｜ReAct模式：构建自己的AutoGPT
# 企业应用篇（8讲）10｜文本分片及向量化：让大模型应用企业内部数据
# 企业应用篇（8讲）11｜LangChain Retrieval：连接大模型和内部文本
# 企业应用篇（8讲）12｜整合所学：构建多模态Chatbot

# 研发效率篇（6讲）13｜研发全过程中的应用：硅基工程师诞生
# 研发效率篇（6讲）14｜代码生成：解决代码生成的依赖性并增强确定性
# 研发效率篇（6讲）15｜有效利用LLM开发：编写大模型友好的代码
# 研发效率篇（6讲）16｜云原生部署任务实践：让你成为更好的DevOps工程师
# 研发效率篇（6讲）17｜HuggingFace与Pre-trained Model：借助AI社区的力量
# 研发效率篇（6讲）18｜架构展望：集成大模型的应用参考架构
