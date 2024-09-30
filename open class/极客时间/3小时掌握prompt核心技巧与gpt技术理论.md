# GPT 模型实战：技巧与原则
![image](https://github.com/user-attachments/assets/6f70d572-8d92-4af3-b4b7-dd591ab77e3f)

# 快速掌握 Prompt 工程核心技巧
GPT 模型实战：Official Playground

- **针对长内容生成，system message 很有用，其中的内容可以被全局记住**，而不需要在对话的上下文中记忆，从而节省大量tokens。
![image](https://github.com/user-attachments/assets/2c28f845-240a-4c5a-b9c9-f96b8753af3e)

- **中文的 prompt 中有步骤时，增加“单次回复全部执行完”的说明**，否则大模型输出的内容可能是单个步骤的内容，继续提交时才会输出下一步骤的内容（英文的 prompt 生成的结果明显比中文的 prompt 好）。
![image](https://github.com/user-attachments/assets/abbf3eb5-5e95-4eb3-9e55-c75edb90e0db)

- 通过指定格式构造标注数据
![image](https://github.com/user-attachments/assets/bdd6f1b8-adde-4e77-a8d9-cca04d51b203)

# GPT 技术理论贯通
1.NLP 语言模型技术发展一览

![image](https://github.com/user-attachments/assets/750e9c27-252b-42ea-ae49-ac419b2ad789)

2.预训练语言模型

![image](https://github.com/user-attachments/assets/725cbafb-dc7b-4f91-b9b3-75473d5b7ad9)

3.预训练语言模型的三种网络架构（2018～2020）

![image](https://github.com/user-attachments/assets/1e323f2d-faa0-4670-9e2e-dffcf2850b20)

![image](https://github.com/user-attachments/assets/b15f445e-2993-4238-9617-bf3dabfc4ba5)

4.语言模型进展（到2020年）

![image](https://github.com/user-attachments/assets/f2f482bf-0a5d-45fc-9225-4bd3b57efaaa)

- ELMo：两个方向训练单项模型，最后拼接在一起。

![image](https://github.com/user-attachments/assets/07d51e59-12d6-49eb-920f-c18236e2b142)

- GPT-1：将 transformer 的解码器拿出来，层数从6加到12，忽略了编码器的输入，把中间的 attention 去掉，在此基础上对不同的下游任务做微调。**需要改模型。**

![image](https://github.com/user-attachments/assets/a869f609-7919-4ea6-bb1d-ff0c51a264ef)

![image](https://github.com/user-attachments/assets/1f7558f2-b1bb-4d6f-ba75-dcb5cc398b33)

![image](https://github.com/user-attachments/assets/1237a96f-e98e-4eda-820a-29c73e0f2db1)

- GPT2：相比 GPT-1，扩大了训练集和参数，并改变了思路，不再是训练好大模型，再为多种下游任务做微调，而是把两件事合二为一，训练一个大模型，天然支持多种下游任务，为 prompt learning 埋下了种子。

- GPT3：相比 GPT-2，进一步扩大数据集（例如 Common Crawl/网络爬虫数据集），并进行上下文学习/In-context learning（替换梯度下降）。**模型更大，给出示例，进行上下文学习，不再是更新模型本身**

![image](https://github.com/user-attachments/assets/c8395b1e-aa78-473d-8909-0f50eb103e96)

![image](https://github.com/user-attachments/assets/c4101e9e-c0bf-4d60-ab02-b30991d7ee24)

![image](https://github.com/user-attachments/assets/84635c4d-0c5f-44b1-b726-fef83b7b2d14)

![image](https://github.com/user-attachments/assets/1793fab0-1e52-42d0-b299-c531b31df028)

few-shot/一些示例一般在3～10个之间。

- GPT1～GPT3 总结：
![image](https://github.com/user-attachments/assets/4570b6ed-9f49-4878-afa1-4430e5193a55)

- 3个关键概念
![image](https://github.com/user-attachments/assets/316eb7c3-e63b-4795-a0b3-19b32304e4ca)

5.ChatGPT

![image](https://github.com/user-attachments/assets/e81b61da-f1ee-4fb1-8596-9d944db2a5b3)

![image](https://github.com/user-attachments/assets/69bd86d1-6e95-410a-b073-0acf8dd89ddd)

![image](https://github.com/user-attachments/assets/f4ae44b9-e5ec-434f-b918-87bd81ec0a0a)

- GPT3的两条迭代路径：学习代码、指令微调。两者结合并经过有监督学习，最终得到了一个模型 Text-davinci-002。
- Text-davinci-003和 ChatGPT 是在 Text-davinci-002 上做了人类反馈的微调，使回答更舒服。

![image](https://github.com/user-attachments/assets/9c484dc6-5a72-4711-bf40-ad88b8219c4b)

![image](https://github.com/user-attachments/assets/d587195a-5e31-4d72-ae64-904ab4742a9e)

![image](https://github.com/user-attachments/assets/e3af03a0-35af-48ea-9c9f-6e373e1dfb7e)

![image](https://github.com/user-attachments/assets/670396a5-ed8c-4992-9b3b-2b7d98dbd2ac)

6.GPT4

![image](https://github.com/user-attachments/assets/402fb7c3-9af2-47cd-910b-75b7086923e2)

![image](https://github.com/user-attachments/assets/be4006bd-eb63-4ae8-a3d0-fe29fe45ba1b)

![image](https://github.com/user-attachments/assets/64cc57b6-eb78-4ada-8391-1d05325b0cdd)

![image](https://github.com/user-attachments/assets/67462622-63f7-4f80-9235-c4368faeea59)

7.相关论文

GPT3：
- [《Improving Language Understanding
by Generative Pre-Training》](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [《Language Models are Unsupervised Multitask Learners》](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask)
- [《Language Models are Few-Shot Learners》](https://arxiv.org/abs/2005.14165)

GPT4：
- [《Sparks of Artificial General Intelligence: Early experiments with GPT-4》](https://arxiv.org/abs/2303.12712)
- [《GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models》](https://arxiv.org/abs/2303.10130)
- [《GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE》](https://www.semianalysis.com/p/gpt-4-architecture-infrastructure)
