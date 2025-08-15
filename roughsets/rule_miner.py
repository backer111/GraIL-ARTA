import os
import logging
import numpy as np
import json
from collections import defaultdict, Counter
import networkx as nx

class RoughSetRuleMiner:
    """
    粗糙集规则挖掘器，用于从知识图谱中提取规则
    """
    
    def __init__(self, config):
        """
        初始化规则挖掘器
        
        参数:
            config: 配置字典，包含以下内容：
                - confidence_threshold: 规则置信度阈值
                - support_threshold: 规则支持度阈值
                - max_path_length: 路径规则的最大长度
                - max_rules_per_relation: 每个关系的最大规则数量
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.support_threshold = config.get('support_threshold', 5)
        self.max_path_length = config.get('max_path_length', 3)
        self.max_rules_per_relation = config.get('max_rules_per_relation', 100)
        self.ruleset = []
        self.entity_types = {}  # 实体类型字典
        self.relation_counts = Counter()  # 关系频次计数
        
    def load_kg(self, file_path):
        """
        加载知识图谱数据
        
        参数:
            file_path: 知识图谱文件路径
        """
        logging.info(f"Loading knowledge graph from {file_path}")
        self.triples = []
        self.entities = set()
        self.relations = set()
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        
        # 检测数据集类型
        self.dataset_type = None
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split('\t')
                if len(parts) >= 3:
                    # 检测数据集类型
                    if parts[1].startswith('/'):
                        self.dataset_type = 'fb15k237'  # FB15K237格式
                        logging.info("Detected FB15K237 dataset format")
                    elif parts[1].startswith('_'):
                        self.dataset_type = 'wn18rr'     # WN18RR格式
                        logging.info("Detected WN18RR dataset format")
        
        # 如果无法自动检测，默认为FB15K237格式
        if not self.dataset_type:
            self.dataset_type = 'fb15k237'
            logging.warning("Could not auto-detect dataset type, defaulting to FB15K237 format")
        
        # 重新读取整个文件
        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                self.triples.append((h, r, t))
                self.entities.add(h)
                self.entities.add(t)
                self.relations.add(r)
                self.relation_counts[r] += 1
        
        # 创建索引
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}
        
        # 构建图结构
        self.nx_graph = nx.DiGraph()
        for h, r, t in self.triples:
            # 将关系直接存储为边属性
            self.nx_graph.add_edge(h, t, relation=r)
            
        logging.info(f"Loaded {len(self.triples)} triples with {len(self.entities)} entities and {len(self.relations)} relations")
    
    def extract_entity_types(self):
        """提取实体的类型信息"""
        # 根据数据集类型确定类型关系
        if self.dataset_type == 'fb15k237':
            type_relations = ["/type/object/type"]  # FB15K常用的类型关系
        elif self.dataset_type == 'wn18rr':
            type_relations = ["_hypernym", "_instance_hypernym"]  # WN18RR的类型关系
        else:
            type_relations = ["/type/object/type"]  # 默认使用FB15K的类型关系
        
        type_relation_count = 0
        for h, r, t in self.triples:
            if r in type_relations:
                if h not in self.entity_types:
                    self.entity_types[h] = []
                self.entity_types[h].append(t)
                type_relation_count += 1
        
        logging.info(f"Extracted types for {len(self.entity_types)} entities using {type_relation_count} type relations")
    
    def extract_path_patterns(self):
        """提取路径模式规则"""
        logging.info("Extracting path pattern rules")
        path_rules = defaultdict(list)
        
        # 对每个关系，我们寻找可能的路径模式
        for target_relation in self.relations:
            if self.relation_counts[target_relation] < self.support_threshold:
                continue
                
            # 获取具有此关系的所有三元组
            target_triples = [(h, t) for h, r, t in self.triples if r == target_relation]
            
            # 如果这个关系的三元组太少，跳过
            if len(target_triples) < self.support_threshold:
                continue
            
            logging.info(f"Mining rules for relation: {target_relation} with {len(target_triples)} triples")
            
            # 我们将这些三元组按比例分为训练集和验证集
            np.random.shuffle(target_triples)
            split_idx = int(0.8 * len(target_triples))
            train_pairs = target_triples[:split_idx]
            valid_pairs = target_triples[split_idx:]
            
            # 寻找1-3跳的路径模式
            candidate_rules = []
            for path_length in range(1, self.max_path_length + 1):
                # 对每个训练对，寻找从头到尾的路径
                for h, t in train_pairs:
                    paths = self._find_paths(h, t, path_length)
                    
                    for path in paths:
                        rule = {
                            'type': 'path',
                            'relation': target_relation,
                            'path': path,
                            'confidence': None,  # 将在验证阶段计算
                            'support': 0
                        }
                        
                        # 检查该路径是否已经在候选规则中
                        if rule not in candidate_rules:
                            candidate_rules.append(rule)
                            
            # 在验证集上计算每个规则的置信度
            for rule in candidate_rules:
                correct_predictions = 0
                total_predictions = 0
                
                for h, _ in valid_pairs:
                    # 找到所有根据规则预测的尾实体
                    predicted_tails = self._follow_path(h, rule['path'])
                    
                    # 如果有预测结果
                    if predicted_tails:
                        total_predictions += len(predicted_tails)
                        # 检查原始尾实体是否在预测中
                        correct_predictions += sum(1 for pair in valid_pairs if pair[0] == h and pair[1] in predicted_tails)
                
                if total_predictions > 0:
                    rule['confidence'] = correct_predictions / total_predictions
                    rule['support'] = total_predictions
                    
                    # 如果规则满足阈值要求，添加到规则集
                    if (rule['confidence'] >= self.confidence_threshold and 
                        rule['support'] >= self.support_threshold):
                        path_rules[target_relation].append(rule)
            
            # 根据置信度排序并保留前N个规则
            path_rules[target_relation] = sorted(
                path_rules[target_relation], 
                key=lambda r: (r['confidence'], r['support']),
                reverse=True
            )[:self.max_rules_per_relation]
            
            logging.info(f"Found {len(path_rules[target_relation])} valid rules for {target_relation}")
        
        # 将路径规则添加到规则集
        for relation, rules in path_rules.items():
            self.ruleset.extend(rules)
    
    def extract_type_constraints(self):
        """提取类型约束规则"""
        logging.info("Extracting type constraint rules")
        type_rules = defaultdict(list)
        
        # 首先确保我们有实体类型信息
        if not self.entity_types:
            self.extract_entity_types()
            
        # 根据数据集类型调整阈值
        min_entities_with_types = 100  # 默认需要至少100个实体有类型
        if self.dataset_type == 'wn18rr':
            min_entities_with_types = 50  # WN18RR数据集可能实体类型较少
            
        # 如果没有足够的类型信息，则跳过
        if len(self.entity_types) < min_entities_with_types:
            logging.warning(f"Not enough entity type information for type constraint rules: {len(self.entity_types)} < {min_entities_with_types}")
            return
        
        # 对每个关系，分析头尾实体类型
        for relation in self.relations:
            # 获取具有此关系的所有三元组
            rel_triples = [(h, t) for h, r, t in self.triples if r == relation]
            if len(rel_triples) < self.support_threshold:
                continue
                
            # 统计头尾实体类型
            head_types = Counter()
            tail_types = Counter()
            
            for h, t in rel_triples:
                if h in self.entity_types:
                    for h_type in self.entity_types[h]:
                        head_types[h_type] += 1
                
                if t in self.entity_types:
                    for t_type in self.entity_types[t]:
                        tail_types[t_type] += 1
            
            # 寻找频繁的头尾实体类型组合
            for h_type, h_count in head_types.most_common(5):
                if h_count < self.support_threshold:
                    continue
                    
                for t_type, t_count in tail_types.most_common(5):
                    if t_count < self.support_threshold:
                        continue
                    
                    # 计算此类型组合的置信度
                    # 即在给定关系和头实体类型的条件下，尾实体为特定类型的概率
                    valid_heads = sum(1 for h, _ in rel_triples if h in self.entity_types and h_type in self.entity_types[h])
                    valid_tails = sum(1 for h, t in rel_triples if h in self.entity_types and t in self.entity_types 
                                      and h_type in self.entity_types[h] and t_type in self.entity_types[t])
                    
                    if valid_heads > 0:
                        confidence = valid_tails / valid_heads
                        
                        if confidence >= self.confidence_threshold:
                            rule = {
                                'type': 'type_constraint',
                                'relation': relation,
                                'head_type': h_type,
                                'tail_type': t_type,
                                'confidence': confidence,
                                'support': valid_tails
                            }
                            type_rules[relation].append(rule)
            
            # 根据置信度排序
            type_rules[relation] = sorted(
                type_rules[relation],
                key=lambda r: (r['confidence'], r['support']),
                reverse=True
            )[:self.max_rules_per_relation]
            
            logging.info(f"Found {len(type_rules[relation])} type constraint rules for {relation}")
        
        # 将类型规则添加到规则集
        for relation, rules in type_rules.items():
            self.ruleset.extend(rules)
    
    def _find_paths(self, start_node, end_node, max_length):
        """查找从start_node到end_node的所有长度为max_length的路径"""
        paths = []
        if max_length == 1:
            # 直接检查是否有边
            if self.nx_graph.has_edge(start_node, end_node):
                # 获取这条边的关系属性
                relation = self.nx_graph.get_edge_data(start_node, end_node)['relation']
                paths.append([relation])
        else:
            # 对于更长的路径，执行深度优先搜索
            def dfs(current, target, path, length):
                if length == 0:
                    if current == target:
                        paths.append(path.copy())
                    return
                
                # 限制搜索邻居的数量以提高效率
                neighbors = list(self.nx_graph.successors(current))
                if len(neighbors) > 10:  # 如果邻居太多，随机取样
                    neighbors = np.random.choice(neighbors, 10, replace=False)
                
                for neighbor in neighbors:
                    # 获取边的关系属性
                    relation = self.nx_graph.get_edge_data(current, neighbor)['relation']
                    path.append(relation)
                    dfs(neighbor, target, path, length - 1)
                    path.pop()  # 回溯
            
            dfs(start_node, end_node, [], max_length)
        
        return paths
    
    def _follow_path(self, start_node, path):
        """从start_node开始，沿着给定的路径寻找终点"""
        current_nodes = [start_node]
        for relation in path:
            next_nodes = []
            for node in current_nodes:
                for neighbor in self.nx_graph.successors(node):
                    # 检查边的关系是否匹配
                    edge_relation = self.nx_graph.get_edge_data(node, neighbor)['relation']
                    if edge_relation == relation:
                        next_nodes.append(neighbor)
            current_nodes = next_nodes
            if not current_nodes:  # 如果某一步无法找到下一个节点，提前退出
                break
        
        return current_nodes
    
    def mine_rules(self, kg_file):
        """
        从知识图谱中挖掘规则
        
        参数:
            kg_file: 知识图谱文件路径
        
        返回:
            ruleset: 规则集列表
        """
        # 加载知识图谱
        self.load_kg(kg_file)
        
       
        
        # 提取路径模式规则
        self.extract_path_patterns()
        
        # 提取类型约束规则
        self.extract_type_constraints()
        
        logging.info(f"Total rules mined: {len(self.ruleset)}")
        
        # 为每条规则分配唯一ID
        for i, rule in enumerate(self.ruleset):
            rule['id'] = i
        
        return self.ruleset
    
    def save_ruleset(self, output_file):
        """
        将规则集保存到文件
        
        参数:
            output_file: 输出文件路径
        """
        with open(output_file, 'w') as f:
            json.dump(self.ruleset, f, indent=2)
        
        logging.info(f"Saved {len(self.ruleset)} rules to {output_file}")
        
    def analyze_rule_conflicts(self):
        """分析规则之间的冲突情况"""
        conflict_pairs = []
        
        # 按关系对规则进行分组
        rules_by_relation = defaultdict(list)
        for rule in self.ruleset:
            rules_by_relation[rule['relation']].append(rule)
        
        # 对于每个关系，查找相互冲突的规则对
        for relation, rules in rules_by_relation.items():
            # 只考虑同一类型的规则之间的冲突
            path_rules = [r for r in rules if r['type'] == 'path']
            type_rules = [r for r in rules if r['type'] == 'type_constraint']
            
            # 检查路径规则的冲突
            for i, rule1 in enumerate(path_rules):
                for rule2 in path_rules[i+1:]:
                    # 如果两条路径规则预测的结果经常不一致，我们认为它们冲突
                    # 这里简化处理，只检查路径是否完全相同
                    if rule1['path'] == rule2['path'] and abs(rule1['confidence'] - rule2['confidence']) > 0.3:
                        conflict_pairs.append((rule1['id'], rule2['id']))
            
            # 检查类型规则的冲突
            for i, rule1 in enumerate(type_rules):
                for rule2 in type_rules[i+1:]:
                    # 如果头实体类型相同但尾实体类型不同，且置信度都很高，则冲突
                    if (rule1['head_type'] == rule2['head_type'] and 
                        rule1['tail_type'] != rule2['tail_type'] and
                        rule1['confidence'] > 0.8 and rule2['confidence'] > 0.8):
                        conflict_pairs.append((rule1['id'], rule2['id']))
        
        logging.info(f"Found {len(conflict_pairs)} conflicting rule pairs")
        
        # 将冲突信息添加到规则中
        conflict_dict = defaultdict(list)
        for rule1_id, rule2_id in conflict_pairs:
            conflict_dict[rule1_id].append(rule2_id)
            conflict_dict[rule2_id].append(rule1_id)
        
        for rule in self.ruleset:
            rule['conflicts'] = conflict_dict.get(rule['id'], [])
        
        return conflict_pairs 