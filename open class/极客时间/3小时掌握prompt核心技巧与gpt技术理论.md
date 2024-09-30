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

4.语言模型进展
- ELMo：两个方向训练单项模型，最后拼接在一起。

![image](https://github.com/user-attachments/assets/07d51e59-12d6-49eb-920f-c18236e2b142)

- GPT-1：将 transformer 的解码器拿出来，层数从6加到12，忽略了编码器的输入，把中间的 attention 去掉，在此基础上对不同的下游任务做微调。**需要改模型。**

![image](https://github.com/user-attachments/assets/a869f609-7919-4ea6-bb1d-ff0c51a264ef)

![image](https://github.com/user-attachments/assets/1f7558f2-b1bb-4d6f-ba75-dcb5cc398b33)

![image](https://github.com/user-attachments/assets/1237a96f-e98e-4eda-820a-29c73e0f2db1)

- GPT2：相比 GPT-1，扩大了训练集和参数，并改变了思路，不再是训练好大模型，再为多种下游任务做微调，而是把两件事合二为一，训练一个大模型，天然支持多种下游任务，为 prompt learning 埋下了种子。**不需要更改模型。**

- GPT3：相比 GPT-2，进一步扩大数据集（例如 Common Crawl/网络爬虫数据集），并进行上下文学习/In-context learning（替换梯度下降）。**模型更大，给出示例，进行上下文学习。**

![image](https://github.com/user-attachments/assets/c8395b1e-aa78-473d-8909-0f50eb103e96)

![image](https://github.com/user-attachments/assets/c4101e9e-c0bf-4d60-ab02-b30991d7ee24)

2020年半监督序列学习总结

![image](https://github.com/user-attachments/assets/f2f482bf-0a5d-45fc-9225-4bd3b57efaaa)


