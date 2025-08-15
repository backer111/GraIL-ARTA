# GraIL-ART 知识图谱关系预测框架

## 项目概述

GraIL-ART是一个将图神经网络与符号规则相结合的知识图谱关系预测框架，特别适用于归纳关系预测场景（对未见实体进行预测）。本项目是对原始GraIL框架的扩展，通过整合规则增强推理能力。

## 问题修复

在框架实现过程中，我们解决了以下关键问题：

### 1. 规则匹配格式不一致问题

**症状**：
- 辅助任务损失(aux_loss)始终为0
- 规则贡献率极低(0.0098)
- 规则匹配率为0%

**原因**：
- 规则文件使用路径格式（如"/film/film/genre"）
- 训练数据使用数字ID格式（如0,1,2...）
- 原始规则匹配器无法建立映射关系

**解决方案**：
- 实现增强版规则匹配器(EnhancedRuleMatcher)，支持双向映射
- 保存实体和关系的ID映射文件，实现路径-ID转换
- 优化参数配置提升规则贡献

### 2. 参数优化

通过实验确定以下最优参数配置：
- 辅助损失权重(aux_loss_weight): 0.3 (原为0.1)
- 规则丢弃率(rule_dropout): 0.1 (原为0.3)
- 子图跳数(hop): 2

## 框架结构

GraIL-ART框架分为三个阶段：

### 第一阶段：离线规则学习
1. 从知识图谱中挖掘规则
2. 规则编码与嵌入
3. 生成规则集文件

### 第二阶段：联合多任务训练
1. 图神经网络进行关系预测
2. 规则匹配器检索相关规则
3. 门控机制整合GNN预测与规则预测
4. 多任务学习同时优化关系预测和规则匹配任务

### 第三阶段：归纳关系预测
1. 对包含未见实体的新数据进行测试
2. 评估模型在归纳场景下的性能
3. 分析规则贡献度

## 使用指南

### 环境配置

确保安装所需依赖：
```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 1.7+
- DGL 0.6+
- NumPy
- SciPy
- tqdm

### 数据准备

数据集应放在`data/数据集名称/`目录下，包含以下文件：
- train.txt：训练集三元组
- valid.txt：验证集三元组
- test.txt：测试集三元组
- 归纳测试数据：带_ind后缀的另一个数据集

数据格式为每行一条三元组：`头实体 关系 尾实体`

### 运行训练

第二阶段联合训练：
```bash
python -m roughsets.phase2_joint_training \
    --dataset fb237_v1 \
    --hop 2 \
    --experiment_name fb237_v1_fixed \
    --gpu 0 \
    --aux_loss_weight 0.3 \
    --rule_dropout 0.1
```

### 归纳评估

评估模型在归纳场景的性能：
```bash
python -m roughsets.run_inductive_evaluation \
    --dataset fb237_v1_ind \
    --model_dataset fb237_v1 \
    --experiment_name fb237_v1_fixed \
    --hop 2
```

### 模型推理

使用训练好的模型进行推理：
```bash
python -m roughsets.model_inference_example \
    --experiment_name fb237_v1_fixed \
    --dataset fb237_v1
```

## 主要文件说明

- **roughsets/phase2_joint_training.py**: 联合训练主脚本
- **roughsets/evaluate_inductive.py**: 归纳测试评估脚本
- **roughsets/rule_matcher_enhanced.py**: 增强版规则匹配器
- **roughsets/analyze_enclosing_subgraph_impact.py**: 封闭子图影响分析工具
- **roughsets/model_inference_example.py**: 模型推理示例

## 性能指标

在FB237_v1数据集上的性能：

| 指标 | 原始版本 | 修复后 |
|------|---------|--------|
| AUC  | 0.8321  | 0.8550 |
| AUC-PR | 0.8865 | 0.9074 |
| 规则匹配率 | 0% | 100% |
| 规则贡献率 | 0.0098 | 0.1547 |

## 进阶使用

### 封闭子图分析

分析封闭子图选项对规则贡献率的影响：
```bash
python -m roughsets.analyze_enclosing_subgraph_impact \
    --dataset fb237_v1 \
    --ind_dataset fb237_v1_ind \
    --experiment_name fb237_v1_fixed
```

### 模型导出

训练好的模型将保存在`experiments/实验名称/`目录下：
- `best_gnn_model.pth`: 最佳GNN模型
- `best_agent_model.pth`: 最佳规则代理模型

## 常见问题

1. **规则匹配率低怎么办？**
   - 检查规则文件格式
   - 确认ID映射文件已正确生成
   - 使用增强版规则匹配器

2. **如何提高模型性能？**
   - 优化子图提取参数
   - 调整辅助损失权重
   - 增加高质量规则数量

3. **为什么使用封闭子图？**
   - 封闭子图可以约束子图大小
   - 有助于模型聚焦于最相关的结构信息
   - 可能提高整体性能但降低规则贡献率

## 参考文献

- GraIL: [学习知识图谱的归纳表示](https://arxiv.org/abs/1911.06962)
- ART: [使用基于规则的推理增强递归Transformers](https://arxiv.org/abs/2205.11498) 