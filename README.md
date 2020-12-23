# 2020CCF-NER
2020 CCF大数据与计算智能大赛-非结构化商业文本信息中隐私信息识别-第7名方案

bert base + flat + crf + fgm + swa + pu learning策略 + clue数据集 = test1单模0.906

词向量：https://github.com/Embedding/Chinese-Word-Vectors SGNS(Mixed-large 综合)

loss mask相关代码为pu learning策略的实现

主要模块版本

transformers 3.0.2 

pytorchcrf 1.2.0 

torchcontrib 0.0.2
