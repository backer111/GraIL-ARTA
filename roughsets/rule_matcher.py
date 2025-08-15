import torch
import numpy as np
import logging
from collections import defaultdict


class RuleMatcher:
    """
    规则匹配器类，负责在给定子图的情况下找到匹配的规则
    """
    def __init__(self, ruleset, device, no_rule_idx):
        """
        初始化规则匹配器
        
        参数:
            ruleset: 规则集合，包含所有从知识图谱中提取的规则
            device: 计算设备
            no_rule_idx: 表示"无规则"的特殊索引
        """
        self.ruleset = ruleset
        self.device = device
        self.no_rule_idx = no_rule_idx
        
        # 按关系组织规则
        self.rules_by_relation = defaultdict(list)
        self.rule_ids_by_relation = defaultdict(list)
        
        # 构建索引
        for i, rule in enumerate(ruleset):
            rel = rule.get('relation', None)
            if rel is not None:
                self.rules_by_relation[rel].append(rule)
                self.rule_ids_by_relation[rel].append(i)
        
        # 初始化统计计数器
        self.stats_counter = 0
        self.accumulated_stats = defaultdict(lambda: {'total': 0, 'matched': 0})
        self.accumulated_total_samples = 0
        self.accumulated_matched_samples = 0
        
        logging.info(f"规则匹配器初始化完成。共有 {len(ruleset)} 条规则，覆盖 {len(self.rules_by_relation)} 种关系。")
    
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
            relation = relations[i].item()
            subgraph = subgraphs[i] if isinstance(subgraphs, list) else subgraphs
            
            # 获取关系对应的规则
            rule_ids = self.rule_ids_by_relation.get(relation, [])
            
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
            self.accumulated_stats[relation]['total'] += 1
            if rule_ids and torch.any(rule_mask):
                batch_matches += 1
                self.accumulated_stats[relation]['matched'] += 1
        
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
                        logging.info(f"  关系ID {rel_id}: {cov:.4f} ({stats['matched']}/{stats['total']})")
            
            # 重置累积统计（可选，取消注释以重置）
            # self.accumulated_stats = defaultdict(lambda: {'total': 0, 'matched': 0})
            # self.accumulated_total_samples = 0
            # self.accumulated_matched_samples = 0
        
        return matched_rule_ids, matched_rule_masks
    
    def match_rule_batched(self, pos_subgraphs, pos_relations, neg_subgraphs, neg_relations, 
                          max_rules=10, apply_dropout=False, dropout_prob=0.0):
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