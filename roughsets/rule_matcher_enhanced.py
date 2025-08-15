import torch
import numpy as np
import json
import os
import logging
from collections import defaultdict


class EnhancedRuleMatcher:
    """
    增强版规则匹配器类，支持关系ID映射，解决路径格式与数字ID之间的匹配问题
    """
    def __init__(self, ruleset, device, no_rule_idx, relation_mapping=None, id2relation=None):
        """
        初始化规则匹配器
        
        参数:
            ruleset: 规则集合，包含所有从知识图谱中提取的规则
            device: 计算设备
            no_rule_idx: 表示"无规则"的特殊索引
            relation_mapping: 关系路径到ID的映射字典（可选）
            id2relation: ID到关系路径的映射字典（可选）
        """
        self.ruleset = ruleset
        self.device = device
        self.no_rule_idx = no_rule_idx
        self.relation_mapping = relation_mapping  # 路径 -> ID
        self.id2relation = id2relation  # ID -> 路径
        
        # 按关系组织规则（使用路径格式）
        self.rules_by_relation_path = defaultdict(list)
        self.rule_ids_by_relation_path = defaultdict(list)
        
        # 按关系ID组织规则（供快速查询）
        self.rules_by_relation_id = defaultdict(list)
        self.rule_ids_by_relation_id = defaultdict(list)
        
        # 构建索引
        for i, rule in enumerate(ruleset):
            rel_path = rule.get('relation', None)
            if rel_path is not None:
                # 保存路径映射
                self.rules_by_relation_path[rel_path].append(rule)
                self.rule_ids_by_relation_path[rel_path].append(i)
                
                # 如果提供了映射，也保存ID映射
                if self.relation_mapping and rel_path in self.relation_mapping:
                    rel_id = self.relation_mapping[rel_path]
                    self.rules_by_relation_id[rel_id].append(rule)
                    self.rule_ids_by_relation_id[rel_id].append(i)
        
        # 初始化统计计数器
        self.stats_counter = 0
        self.accumulated_stats = defaultdict(lambda: {'total': 0, 'matched': 0})
        self.accumulated_total_samples = 0
        self.accumulated_matched_samples = 0
        
        # 统计规则覆盖关系数量
        unique_relations_with_rules = set()
        if self.relation_mapping:
            for rel_path in self.rules_by_relation_path.keys():
                if rel_path in self.relation_mapping:
                    unique_relations_with_rules.add(self.relation_mapping[rel_path])
        
        logging.info(f"规则匹配器初始化完成。共有 {len(ruleset)} 条规则，路径形式覆盖 {len(self.rules_by_relation_path)} 种关系，"
                    f"ID形式覆盖 {len(unique_relations_with_rules)} 种关系。")
    
    @classmethod
    def from_files(cls, ruleset_file, device, no_rule_idx, relation2id_file=None):
        """
        从文件创建规则匹配器
        
        参数:
            ruleset_file: 规则文件路径
            device: 计算设备
            no_rule_idx: 表示"无规则"的特殊索引
            relation2id_file: 关系ID映射文件路径（可选）
            
        返回:
            rule_matcher: 规则匹配器实例
        """
        # 加载规则集
        with open(ruleset_file, 'r') as f:
            ruleset = json.load(f)
        
        # 加载关系ID映射
        relation_mapping = None
        id2relation = None
        if relation2id_file and os.path.exists(relation2id_file):
            with open(relation2id_file, 'r') as f:
                relation_mapping = json.load(f)
                id2relation = {int(v): k for k, v in relation_mapping.items()}
        
        # 创建规则匹配器
        return cls(ruleset, device, no_rule_idx, relation_mapping, id2relation)
    
    def match_rules(self, subgraphs, relations, max_rules=10, apply_dropout=False, dropout_prob=0.0):
        """
        为一批子图匹配规则
        
        参数:
            subgraphs: 批量子图
            relations: 批量关系ID
            max_rules: 每个子图最多匹配的规则数量
            apply_dropout: 是否应用规则dropout
            dropout_prob: dropout概率
            
        返回:
            matched_rule_ids: 每个子图匹配的规则ID列表
            matched_rule_masks: 规则掩码，指示每个位置是否有有效规则
        """
        batch_size = len(relations)
        matched_rule_ids = []
        matched_rule_masks = []
        
        # 批次级统计
        batch_matches = 0
        
        for i in range(batch_size):
            relation_id = relations[i].item()
            subgraph = subgraphs[i] if isinstance(subgraphs, list) else subgraphs
            
            # 查找规则 - 优先使用ID直接查询
            rule_ids = self.rule_ids_by_relation_id.get(relation_id, [])
            
            # 如果没找到且有ID到路径的映射，尝试通过路径查找
            if not rule_ids and self.id2relation and relation_id in self.id2relation:
                relation_path = self.id2relation[relation_id]
                rule_ids = self.rule_ids_by_relation_path.get(relation_path, [])
            
            # 如果没有匹配的规则，使用no_rule_idx
            if not rule_ids:
                rule_ids_tensor = torch.full((max_rules,), self.no_rule_idx, dtype=torch.long, device=self.device)
                rule_mask = torch.zeros(max_rules, dtype=torch.bool, device=self.device)
            else:
                # 将规则ID转换为张量
                rule_ids = rule_ids[:max_rules]  # 限制规则数量
                rule_mask = torch.ones(len(rule_ids), dtype=torch.bool, device=self.device)
                
                # 如果启用dropout，随机丢弃一些规则
                if apply_dropout and dropout_prob > 0:
                    dropout_mask = torch.rand(len(rule_ids), device=self.device) >= dropout_prob
                    rule_mask = rule_mask & dropout_mask
                
                # 填充到最大规则数
                if len(rule_ids) < max_rules:
                    padding = [self.no_rule_idx] * (max_rules - len(rule_ids))
                    rule_ids = rule_ids + padding
                    padding_mask = torch.zeros(max_rules - len(rule_mask), dtype=torch.bool, device=self.device)
                    rule_mask = torch.cat([rule_mask, padding_mask])
                
                rule_ids_tensor = torch.tensor(rule_ids, dtype=torch.long, device=self.device)
            
            matched_rule_ids.append(rule_ids_tensor)
            matched_rule_masks.append(rule_mask)
            
            # 更新关系级统计
            self.accumulated_stats[relation_id]['total'] += 1
            if rule_ids and torch.any(rule_mask):
                batch_matches += 1
                self.accumulated_stats[relation_id]['matched'] += 1
        
        # 堆叠成批量张量
        matched_rule_ids = torch.stack(matched_rule_ids)
        matched_rule_masks = torch.stack(matched_rule_masks)
        
        # 更新累积统计
        self.stats_counter += 1
        self.accumulated_total_samples += batch_size
        self.accumulated_matched_samples += batch_matches
        
        # 每N=10个批次输出一次累积统计
        if self.stats_counter % 10 == 0:
            coverage = self.accumulated_matched_samples / self.accumulated_total_samples if self.accumulated_total_samples > 0 else 0
            logging.info(f"累积{self.stats_counter}批次规则匹配统计: 样本数 {self.accumulated_total_samples}, "
                        f"匹配样本数 {self.accumulated_matched_samples}, 匹配率 {coverage:.4f}")
            
            # 输出覆盖率最低的5个关系
            sorted_rel = sorted(self.accumulated_stats.items(), 
                               key=lambda x: x[1]['matched']/x[1]['total'] if x[1]['total'] > 10 else 1.0)
            
            if len(sorted_rel) >= 5:
                logging.info("覆盖率最低的5个高频关系:")
                for rel_id, stats in sorted_rel[:5]:
                    if stats['total'] >= 10:  # 只关注出现频率较高的关系
                        cov = stats['matched'] / stats['total']
                        rel_path = self.id2relation.get(rel_id, f"ID={rel_id}") if self.id2relation else f"ID={rel_id}"
                        logging.info(f"  关系 {rel_path} (ID={rel_id}): {cov:.4f} ({stats['matched']}/{stats['total']})")
        
        return matched_rule_ids, matched_rule_masks
    
    def match_rule_batched(self, pos_subgraphs, pos_relations, neg_subgraphs, neg_relations, 
                          max_rules=30, apply_dropout=False, dropout_prob=0.0):
        """
        为正负样本批量匹配规则
        
        参数:
            pos_subgraphs: 正样本子图
            pos_relations: 正样本关系
            neg_subgraphs: 负样本子图
            neg_relations: 负样本关系
            max_rules: 每个样本最多匹配的规则数量
            apply_dropout: 是否应用规则dropout
            dropout_prob: dropout概率
            
        返回:
            pos_rule_ids: 正样本匹配的规则ID
            pos_rule_masks: 正样本规则掩码
            neg_rule_ids: 负样本匹配的规则ID
            neg_rule_masks: 负样本规则掩码
        """
        # 匹配正样本规则
        pos_rule_ids, pos_rule_masks = self.match_rules(
            pos_subgraphs, pos_relations, max_rules, apply_dropout, dropout_prob
        )
        
        # 匹配负样本规则
        neg_rule_ids_list = []
        neg_rule_masks_list = []
        
        # 处理可能有多个负样本的情况
        if isinstance(neg_subgraphs[0], list):
            for i in range(len(neg_subgraphs)):
                subgraphs_i = neg_subgraphs[i]
                relations_i = neg_relations[i]
                ids, masks = self.match_rules(
                    subgraphs_i, relations_i, max_rules, apply_dropout, dropout_prob
                )
                neg_rule_ids_list.append(ids)
                neg_rule_masks_list.append(masks)
        else:
            neg_rule_ids, neg_rule_masks = self.match_rules(
                neg_subgraphs, neg_relations, max_rules, apply_dropout, dropout_prob
            )
            neg_rule_ids_list = [neg_rule_ids]
            neg_rule_masks_list = [neg_rule_masks]
            
        return pos_rule_ids, pos_rule_masks, neg_rule_ids_list, neg_rule_masks_list 