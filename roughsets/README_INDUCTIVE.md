# GraIL-ART 归纳关系预测指南

本文档提供了使用修复版GraIL-ART框架进行归纳关系预测的详细指南。

## 框架概述

GraIL-ART是一个基于图神经网络和规则的关系预测框架，它分为三个阶段：

1. **第一阶段**：离线规则学习，从知识图谱中挖掘规则并编码
2. **第二阶段**：联合多任务训练，将图神经网络与规则结合
3. **第三阶段**：归纳关系预测，在包含未见实体的新数据上进行预测

主要改进点：
- 增强规则匹配器解决规则格式不一致问题
- 优化参数配置提升规则贡献率
- 提供归纳评估脚本分析模型性能

## 使用指南

### 准备工作

确保安装了所有必要依赖：
```bash
pip install -r requirements.txt
```

### 1. 运行归纳评估

使用以下命令评估模型在归纳场景下的性能：

```bash
python -m roughsets.run_inductive_evaluation \
    --dataset fb237_v1_ind \
    --model_dataset fb237_v1 \
    --experiment_name fb237_v1_fixed \
    --hop 2
```

参数说明：
- `dataset`: 归纳测试数据集，通常使用_ind后缀版本
- `model_dataset`: 模型训练时使用的数据集
- `experiment_name`: 实验名称，用于加载模型
- `hop`: 子图跳数，通常为2-3
- `enclosing_sub_graph`: 可选参数，指定是否使用封闭子图

评估结果将保存在`experiments/实验名称/inductive_results_数据集.json`文件中。

### 2. 分析封闭子图的影响

分析封闭子图设置对规则贡献率的影响：

```bash
python -m roughsets.analyze_enclosing_subgraph_impact \
    --dataset fb237_v1 \
    --ind_dataset fb237_v1_ind \
    --experiment_name fb237_v1_fixed \
    --output_dir roughsets/analysis
```

该脚本会生成分析报告和对比图表，帮助理解封闭子图对模型性能的影响。

### 3. 使用模型进行推理

使用训练好的模型对新的三元组进行推理：

```bash
python -m roughsets.model_inference_example \
    --experiment_name fb237_v1_fixed \
    --dataset fb237_v1 \
    --hop 2
```

可选参数：
- `enclosing_sub_graph`: 指定是否使用封闭子图
- `disable_cuda`: 禁用CUDA，使用CPU

## 文件说明

- `roughsets/evaluate_inductive.py`: 归纳测试评估脚本
- `roughsets/run_inductive_evaluation.py`: 启动归纳评估的入口脚本
- `roughsets/analyze_enclosing_subgraph_impact.py`: 分析封闭子图影响的工具
- `roughsets/model_inference_example.py`: 模型推理示例
- `roughsets/rule_matcher_enhanced.py`: 增强版规则匹配器实现

## 关键改进点

### 增强版规则匹配器

修复了原始规则匹配器在处理不同格式规则时的问题：

```python
# 初始化增强版规则匹配器
rule_matcher = EnhancedRuleMatcher(ruleset, device, no_rule_idx, relation2id, id2relation)

# 匹配规则（支持ID和路径格式）
rule_ids, rule_masks = rule_matcher.match_rules(
    batched_graph, r_label, max_rules=50, apply_dropout=False)
```

### 优化参数配置

根据实验结果，推荐以下参数配置：

- 辅助损失权重(aux_loss_weight): 0.3
- 规则丢弃率(rule_dropout): 0.1
- 子图跳数(hop): 2-3
- 封闭子图: 根据任务需求选择

## 性能指标

归纳评估中主要关注以下指标：

- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
- **Hits@K**: 排名在前K的比例(K=1,3,5,10)
- **规则匹配率**: 匹配到规则的样本比例
- **规则贡献率**: 规则对最终预测的影响程度

## 常见问题解答

**Q: 规则匹配率为0怎么办？**  
A: 检查规则格式和ID映射文件是否正确，确保使用增强版规则匹配器。

**Q: 封闭子图应该开启吗？**  
A: 这取决于具体任务。通常封闭子图会降低规则贡献率但可能提高整体性能。建议使用分析工具比较两种设置的效果。

**Q: 如何提高规则贡献率？**  
A: 尝试降低规则丢弃率，提高辅助损失权重，使用更多高质量规则。

## 参考文献

- GraIL: [Learning Inductive Representation of Knowledge Graphs](https://arxiv.org/abs/1911.06962)
- ART: [Augmenting Recurrent Transformers with Rule-based Reasoning](https://arxiv.org/abs/2205.11498) 