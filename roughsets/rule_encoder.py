import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
import os
from collections import defaultdict

class RuleEncoder:
    """规则编码器，用于将规则转换为嵌入向量"""
    
    def __init__(self, config):
        """
        初始化规则编码器
        
        参数:
            config: 配置字典，包含：
                - embedding_dim: 规则嵌入维度
                - learning_rate: 学习率
                - epochs: 训练轮数
                - batch_size: 批大小
                - init_scale: 嵌入初始化范围
                - device: 设备 ('cpu' 或 'cuda')
        """
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 32)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.init_scale = config.get('init_scale', 0.1)
        self.device = torch.device(config.get('device', 'cpu'))
        
        self.rule_embeds = None
        self.relation_embeds = None
        self.rule_to_idx = None
        self.relation_to_idx = None
        
    def load_ruleset(self, ruleset_file):
        """
        加载规则集
        
        参数:
            ruleset_file: 规则集文件路径
        """
        with open(ruleset_file, 'r') as f:
            self.ruleset = json.load(f)
        
        logging.info(f"Loaded {len(self.ruleset)} rules from {ruleset_file}")
        
        # 创建规则和关系的索引
        self.rule_to_idx = {rule['id']: i for i, rule in enumerate(self.ruleset)}
        
        # 构建关系到索引的映射
        relations = set(rule['relation'] for rule in self.ruleset)
        self.relation_to_idx = {rel: i for i, rel in enumerate(relations)}
        
        logging.info(f"Found {len(relations)} distinct relations in ruleset")

        # 创建冲突信息矩阵
        self.conflict_matrix = torch.zeros(len(self.ruleset), len(self.ruleset))
        for rule in self.ruleset:
            for conflict_id in rule.get('conflicts', []):
                if conflict_id in self.rule_to_idx:  # 确保冲突规则ID存在
                    self.conflict_matrix[self.rule_to_idx[rule['id']], self.rule_to_idx[conflict_id]] = 1
        
    def initialize_embeddings(self):
        """初始化规则和关系的嵌入向量"""
        # 添加一个额外的嵌入向量，用于表示NO_RULE
        self.rule_embeds = nn.Embedding(len(self.ruleset) + 1, self.embedding_dim)
        self.relation_embeds = nn.Embedding(len(self.relation_to_idx), self.embedding_dim)
        
        # 初始化嵌入向量
        nn.init.uniform_(self.rule_embeds.weight, -self.init_scale, self.init_scale)
        nn.init.uniform_(self.relation_embeds.weight, -self.init_scale, self.init_scale)
        
        # 使用冲突信息优化初始化
        with torch.no_grad():
            for i in range(len(self.ruleset)):
                for j in range(len(self.ruleset)):
                    if self.conflict_matrix[i, j] == 1:
                        # 如果规则i和规则j冲突，则初始化它们的嵌入向量，使它们远离彼此
                        direction = torch.randn(self.embedding_dim)
                        direction = direction / torch.norm(direction)
                        self.rule_embeds.weight[i] += direction * self.init_scale
                        self.rule_embeds.weight[j] -= direction * self.init_scale
        
        # 将<NO_RULE>嵌入（最后一个）初始化为接近零的值
        with torch.no_grad():
            self.rule_embeds.weight[-1].fill_(0.01)
    
    def _create_training_samples(self):
        """创建训练样本
        
        返回:
            samples: 训练样本列表 [(rule_idx, relation_idx, similar_rule_idx, dissimilar_rule_idx)]
        """
        samples = []
        
        # 按关系对规则进行分组
        rules_by_relation = defaultdict(list)
        for rule in self.ruleset:
            rules_by_relation[rule['relation']].append(rule)
        
        # 对每个规则，寻找相似和不相似的规则
        for rule in self.ruleset:
            rule_idx = self.rule_to_idx[rule['id']]
            relation = rule['relation']
            relation_idx = self.relation_to_idx[relation]
            
            # 同一关系下的其他规则为相似规则
            similar_rules = [r for r in rules_by_relation[relation] if r['id'] != rule['id']]
            # 不同关系的规则为不相似规则
            dissimilar_relations = [r for r in self.relation_to_idx.keys() if r != relation]
            
            # 如果没有足够的相似规则，继续下一个规则
            if not similar_rules:
                continue
                
            # 随机选择一个相似规则
            similar_rule = np.random.choice(similar_rules)
            similar_rule_idx = self.rule_to_idx[similar_rule['id']]
            
            # 随机选择一个不相似规则
            if dissimilar_relations:
                dissimilar_relation = np.random.choice(dissimilar_relations)
                if rules_by_relation[dissimilar_relation]:
                    dissimilar_rule = np.random.choice(rules_by_relation[dissimilar_relation])
                    dissimilar_rule_idx = self.rule_to_idx[dissimilar_rule['id']]
                    
                    # 添加训练样本
                    samples.append((rule_idx, relation_idx, similar_rule_idx, dissimilar_rule_idx))
        
        return samples
    
    def train_embeddings(self):
        """训练规则和关系的嵌入向量"""
        logging.info("Training rule embeddings...")
        
        # 初始化嵌入向量
        self.initialize_embeddings()
        
        # 创建训练样本
        samples = self._create_training_samples()
        if not samples:
            logging.warning("No training samples could be created. Using default embeddings.")
            return
        
        # 将嵌入模型和冲突矩阵移到指定设备
        self.rule_embeds = self.rule_embeds.to(self.device)
        self.relation_embeds = self.relation_embeds.to(self.device)
        self.conflict_matrix = self.conflict_matrix.to(self.device)
        
        # 定义优化器
        optimizer = optim.Adam([
            {'params': self.rule_embeds.parameters()},
            {'params': self.relation_embeds.parameters()}
        ], lr=self.learning_rate)
        
        # 定义三元组损失
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
        # 开始训练
        for epoch in range(self.epochs):
            # 随机打乱样本
            np.random.shuffle(samples)
            
            # 按批次训练
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(samples), self.batch_size):
                batch_samples = samples[i:i+self.batch_size]
                
                rule_idxs = torch.tensor([s[0] for s in batch_samples], device=self.device)
                relation_idxs = torch.tensor([s[1] for s in batch_samples], device=self.device)
                similar_idxs = torch.tensor([s[2] for s in batch_samples], device=self.device)
                dissimilar_idxs = torch.tensor([s[3] for s in batch_samples], device=self.device)
                
                # 前向传播
                anchor = self.rule_embeds(rule_idxs) + self.relation_embeds(relation_idxs)
                positive = self.rule_embeds(similar_idxs) + self.relation_embeds(relation_idxs)
                negative = self.rule_embeds(dissimilar_idxs) + self.relation_embeds(relation_idxs)
                
                # 计算三元组损失
                loss = triplet_loss(anchor, positive, negative)
                
                # 添加冲突正则化：确保冲突的规则在嵌入空间中距离较远
                conflict_reg = torch.tensor(0.0, device=self.device)  # 初始化为张量
                for rule_idx in range(len(self.ruleset)):
                    conflicts = torch.nonzero(self.conflict_matrix[rule_idx]).squeeze(1)
                    if conflicts.numel() > 0:  # 检查是否有冲突
                        # 计算规则嵌入与其冲突规则嵌入之间的相似度
                        rule_embed = self.rule_embeds.weight[rule_idx]
                        conflict_embeds = self.rule_embeds.weight[conflicts]
                        similarity = torch.cosine_similarity(rule_embed.unsqueeze(0), conflict_embeds, dim=1)
                        # 惩罚高相似度
                        conflict_reg += torch.mean(torch.relu(similarity))
                
                # 总损失
                total_loss += loss.item() + conflict_reg.item()
                loss = loss + 0.1 * conflict_reg
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                num_batches += 1
            
            # 打印训练进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/max(1, num_batches):.4f}")
        
        logging.info("Rule embedding training completed")
    
    def save_embeddings(self, output_file):
        """
        保存规则嵌入到文件
        
        参数:
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 将嵌入转换为numpy数组并保存
        rule_embeds_np = self.rule_embeds.weight.detach().cpu().numpy()
        relation_embeds_np = self.relation_embeds.weight.detach().cpu().numpy()
        
        # 创建保存的字典
        save_dict = {
            'rule_embeds': rule_embeds_np.tolist(),
            'relation_embeds': relation_embeds_np.tolist(),
            'rule_to_idx': self.rule_to_idx,
            'relation_to_idx': self.relation_to_idx,
            'no_rule_idx': len(rule_embeds_np) - 1  # <NO_RULE>的索引
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_dict, f)
        
        logging.info(f"Saved rule embeddings to {output_file}")
    
    def encode_ruleset(self, ruleset_file, output_file):
        """
        编码规则集并保存嵌入到文件
        
        参数:
            ruleset_file: 规则集文件路径
            output_file: 输出嵌入文件路径
        """
        self.load_ruleset(ruleset_file)
        self.train_embeddings()
        self.save_embeddings(output_file) 