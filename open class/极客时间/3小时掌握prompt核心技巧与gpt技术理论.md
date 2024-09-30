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

