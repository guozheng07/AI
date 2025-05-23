# 课程材料
[开发者的大模型快速入门指南0516.pdf](https://github.com/user-attachments/files/17083500/0516.pdf)

# 课程总结
## 1.AIGC 什么？
- UGC：用户生成内容（Web 2.0概念）
- AIGC：AI 生成内容

## 2.AIGC 技术分类
![image](https://github.com/user-attachments/assets/40fed720-40fa-42b9-bc89-e2bca54e2758)

runway：runwayml.com（视频编辑器，更进一步的文生视频）
- Edit video using text
- Generate video using text
- Make video using text

## 3.重塑能力
![image](https://github.com/user-attachments/assets/8efa92af-ed0f-4628-8ca9-a171cb17d33d)

## 4.AI 大模型四阶技术对比
![image](https://github.com/user-attachments/assets/cc2a0adf-eb04-4303-ad51-88b107ec4c64)

推荐论文：
- [《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》](https://github.com/user-attachments/files/17083507/Chain-of-Thought.Prompting.Elicits.Reasoning.pdf)
- [《Self-Consistency Improves Chain of Thought Reasoning in Language Models》](https://github.com/user-attachments/files/17083916/Self-Consistency.Improves.Chain.of.Thought.Reasoning.in.Language.Models.pdf)
- [《Tree-of-Thought Deliberate Problem Solving with Large Language Models》](https://github.com/user-attachments/files/17083917/Tree-of-Thought.Deliberate.Problem.Solving.with.Large.Language.Models.pdf)

## 5.AI Agents
电影代表：钢铁侠的人工智能贾维斯

现实产品：Google 2024 大会发布的的 Project Astra

AI Agents 基础：ReAct 范式
![image](https://github.com/user-attachments/assets/4de63133-c058-4b94-a4a0-60f4bda896f0)
- LM：大语言模型
- Env：环境

AI Agents 类型
![image](https://github.com/user-attachments/assets/f4020ff3-2b0d-4a28-b87d-a4f665296ad1)

## 6.LangChain
典型使用场景：RAG
![image](https://github.com/user-attachments/assets/a80c326d-aecf-4183-b39e-3655aaa36808)

技术生态与社区
![image](https://github.com/user-attachments/assets/fb3c30a6-7078-4ffa-bd71-7a2df569fca9)
- LangChain：LLMs 应用开发框架
- LangChain-Community：第三方集成
- LangChain-Core：LCEL 等协议
- LangChain Templates：开箱即用APP示例
- LangServe：Chains 生产部署（REST API）
- LangSmith：一站式开发者平台

## 7.大模型训练技术：Fine-tuning & Pre-training
**Pre-Training vs Fine-Tuning**
- Pre-Training 和 Fine-Tuning 是深度学习，特别是在自然语言处理（NLP）领域中，训练大模型（如 LLaMA、GPT、Gemini 等）的两个关键步骤。这两个步骤共同构成了一种有效的策略，用于利用大量未标记数据学习通用知识，然后通过少量标记数据将这些知识应用于特定任务。
- Pre-Training 是指在大量未标记数据上训练深度学习模型的过程。这一步骤的目的是使模型能够学习到数据的通用特征和模式，从而捕获语言的基本语法和语义信息。这一阶段不需要人工标记的数据，因此可以使用互联网上可获得的大规模文本语料库。
- **在 Pre-Training 之后**，模型将进行 Fine-Tuning，以适应特定的下游任务。在这个阶段，模型使用较小的、针对特定任务标记过的数据集进行训练。Fine-Tuning 的目的是调整和优化预训练语言模型（Pre-Trained LM）的权重，使其能够在特定任务上表现良好，如情感分析、文本分类、问答等。通过 Fine-Tuning，模型能够利用在Pre-Training 阶段学到的通用知识，并将其应用于具体任务。

**Fine-Tuning vs Instruction-Tuning**
- Fine-Tuning 和 Instruction-Tuning 都旨在改善预训练语言模型（如GPT-3等）的性能，但关注点和方法有所不同。
- 在 Fine-Tuning 过程中，模型在一个大型的数据集上进行预训练，学习语言的通用表示。然后，在特定任务的较小数据集上继续训练（即 Fine-Tuning），调整预训练的参数**以优化任务特定（Task-specific）的性能**。从而提高任务的准确率和效率。
- Instruction-Tuning 目标是提高模型**对自然语言指令的响应能力**，创建一个更加通用的模型。通过在广泛的任务类型上使用指令性示例来训练模型，使模型能够理解和执行各种各样的指令。与 Fine-Tuning 针对单一任务不同，这种方法希望模型不仅能理解任务的指令，还能根据这些指令**生成适当的响应格式或输出**。

![image](https://github.com/user-attachments/assets/b046f3ea-1206-4e1b-9bec-b3acd53da623)

## 8.AI大模型应用开发实战营
课程地址：https://u.geekbang.org/subject/llm?utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=pc_interstitial_1677&gk_cus_user_wechat=university

开源代码：https://github.com/DjangoPeng/openai-quickstart
