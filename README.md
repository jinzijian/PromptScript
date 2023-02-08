## Previous work 
https://arxiv.org/abs/2204.10176
## Background:
Script Knowledge
Eg: Give You a scene like: go to the restaurant  Topic
Output:
Read the menu                    Events
Call the waitress
Enjoy the meal
Check out

Two Settings:
Input : Main events (like: go to restaurant, Go to the Laundry)  Mixed Subevents(eg: read menu, prepare clothes, checkout….)  The model need first Select and then ordering;
Input : Main events(like: go to restaurant, Go to the Laundry) Using models like GPT generate directly


## Motivation: Improve Current Model in Script Knowledge

## Datasets: Descript; OMICS; Stories; proScript [2] Wiki How

## BaseLine:
Setting1: 
Reasoning about Goals, Steps, and Temporal Ordering with WikiHow
https://arxiv.org/pdf/2210.06694.pdf	
Setting2: Generate model without prompt

## Metric:  BLEU; Rouge-L; Rouge-W

## Method: 
Step to Step Method (Improve our old paper)   (Setting 1)
                 	Select which subevents belong to the main event
Ordering(Start)  
Related Work
https://paperswithcode.com/task/sentence-ordering
https://arxiv.org/pdf/2104.07064v2.pdf (Important)
https://paperswithcode.com/paper/improving-graph-based-sentence-ordering-with

One Stage Method    (Setting 1)
Consider Include and ordering together (Innovation)
Generate Method  (Setting 2)
Prompt  
https://arxiv.org/abs/2201.11903
Knowledge graph search for better prompt
GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models



Instuct GPT candidate 生成不同的steps



检索 —》 batter method   topic include/exclude events
 + 生成补全 + instruct gpt
Order : prompt  判断连续性 —》 topic 和 events 分开
选出第一个events   startwith   —-》效果很差      prompt
Order   Instruct GPT

Topic 有哪些子任务 + 生成补全



chatGPT 对冷门topic的效果
Gpt3 gptj opt
Chatgpt 数据增强
Topic base prompt


Week1: 
要写unit test

处理下数据集：topic + subennts  ====> Yigeng
Evaluate: 得到一串script + 选择ground truth中最好的一个 + metrics  ===> Qisen
Survey + 实现下instruct GPT； 假如一定要选择第一个 有没有更好的方法 ===》Yongchao
https://arxiv.org/pdf/2104.07064v2.pdf (Important) 纯粹order
Instuct GPT去order    
如何找最好的选第一个的方法和order方法   —-》 Zijian Jin
chatGPT 对冷门topic的效果  —-》 Zijian Jin



