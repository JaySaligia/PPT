# 『ACL 2023 Findings』Pre-trained Language Model with Prompts for Temporal Knowledge Graph Completion

![authors](https://s1.ax1x.com/2023/05/11/p9r6BvQ.png)

![model_00](https://s1.ax1x.com/2023/05/11/p9r6VBR.png)

## Abstract
Temporal Knowledge graph completion (TKGC) is a crucial task that involves reasoning at known timestamps to complete the missing part of facts and has attracted more and more attention in recent years. Most existing methods focus on learning representations based on graph neural networks while inaccurately extracting information from timestamps and insufficiently utilizing the implied information in relations. To address these problems, we propose a novel TKGC model, namely **P**re-trained Language Model with **P**rompts for **T**KGC (PPT). We convert a series of sampled quadruples into pre-trained language model inputs and convert intervals between timestamps into different prompts to make coherent sentences with implicit semantic information. We train our model with a masking strategy to convert TKGC task into a masked token prediction task, which can leverage the semantic information in pre-trained language models. Experiments on three benchmark datasets and extensive analysis demonstrate that our model has great competitiveness compared to other models with four metrics. Our model can effectively incorporate information from temporal knowledge graphs into the language models.

:page_facing_up: Printed :link: [acl2023-findings]([https://arxiv.org/abs/2305.07912](https://aclanthology.org/2023.findings-acl.493.pdf)https://aclanthology.org/2023.findings-acl.493.pdf)
