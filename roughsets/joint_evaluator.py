import os
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm


class JointEvaluator:
    """
    联合评估器，用于评估GNN和Agent联合模型的性能
    """
    def __init__(self, params, graph_classifier, transformer_agent, rule_matcher, data):
        """
        初始化联合评估器
        
        参数:
            params: 评估参数
            graph_classifier: GNN模型
            transformer_agent: Transformer Agent模型
            rule_matcher: 规则匹配器
            data: 验证/测试数据
        """
        self.params = params
        self.graph_classifier = graph_classifier
        self.transformer_agent = transformer_agent
        self.rule_matcher = rule_matcher
        self.data = data
    
    def eval(self, save=False):
        """
        评估模型
        
        参数:
            save: 是否保存预测结果
            
        返回:
            包含评估指标的字典
        """
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        pos_gnn_scores = []
        neg_gnn_scores = []
        pos_rule_contrib = []
        neg_rule_contrib = []
        
        dataloader = DataLoader(
            self.data, 
            batch_size=self.params.batch_size, 
            shuffle=False, 
            num_workers=self.params.num_workers, 
            collate_fn=self.params.collate_fn
        )
        
        # 设置模型为评估模式
        self.graph_classifier.eval()
        self.transformer_agent.eval()
        
        # 使用tqdm创建进度条
        pbar = tqdm(dataloader, desc="评估批次", ncols=100)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                
                # GNN前向传播
                score_pos_gnn = self.graph_classifier(data_pos)
                score_neg_gnn = self.graph_classifier(data_neg)
                
                # 规则匹配（不使用dropout）
                pos_relations = data_pos[1]  # 假设关系ID位于索引1
                neg_relations = data_neg[1]
                
                pos_rule_ids, pos_rule_masks, neg_rule_ids_list, neg_rule_masks_list = self.rule_matcher.match_rule_batched(
                    data_pos, pos_relations, data_neg, neg_relations, max_rules=50, apply_dropout=False
                )
                
                # Agent前向传播
                pos_adj, _ = self.transformer_agent(pos_rule_ids, pos_rule_masks)
                
                # 处理负样本
                neg_adj_list = []
                for neg_rule_ids, neg_rule_masks in zip(neg_rule_ids_list, neg_rule_masks_list):
                    neg_adj, _ = self.transformer_agent(neg_rule_ids, neg_rule_masks)
                    neg_adj_list.append(neg_adj)
                
                # 融合得分
                pos_gate = torch.sigmoid(pos_adj)
                score_pos_final = score_pos_gnn * pos_gate
                
                score_neg_final_list = []
                for neg_idx, neg_adj in enumerate(neg_adj_list):
                    neg_gate = torch.sigmoid(neg_adj)
                    score_neg_final = score_neg_gnn[neg_idx:neg_idx+1] * neg_gate
                    score_neg_final_list.append(score_neg_final)
                
                # 收集评估指标
                batch_pos_scores = score_pos_final.squeeze(1).cpu().tolist()
                batch_neg_scores = torch.cat(score_neg_final_list, dim=0).squeeze(1).cpu().tolist()
                
                pos_scores += batch_pos_scores
                neg_scores += batch_neg_scores
                
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()
                
                # 收集GNN和规则贡献指标
                batch_pos_gnn = score_pos_gnn.squeeze(1).cpu().tolist()
                batch_neg_gnn = score_neg_gnn.squeeze(1).cpu().tolist()
                pos_gnn_scores += batch_pos_gnn
                neg_gnn_scores += batch_neg_gnn
                
                batch_pos_contrib = pos_gate.squeeze(1).cpu().tolist()
                pos_rule_contrib += batch_pos_contrib
                
                batch_neg_contrib = []
                for neg_gate in neg_adj_list:
                    batch_neg_contrib += torch.sigmoid(neg_gate).squeeze(1).cpu().tolist()
                neg_rule_contrib += batch_neg_contrib
                
                # 更新进度条信息
                if len(batch_pos_scores) > 0 and len(batch_neg_scores) > 0:
                    pbar.set_postfix({
                        'pos_score': f'{np.mean(batch_pos_scores):.4f}',
                        'neg_score': f'{np.mean(batch_neg_scores):.4f}',
                        'rule_contrib': f'{np.mean(batch_pos_contrib):.4f}'
                    })
        
        # 计算评估指标
        all_scores = pos_scores + neg_scores
        all_labels = pos_labels + neg_labels
        
        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)
        
        # 计算规则贡献率
        avg_pos_rule_contrib = np.mean(pos_rule_contrib)
        avg_neg_rule_contrib = np.mean(neg_rule_contrib)
        
        # 计算正负样本平均分数
        avg_pos_score = np.mean(pos_scores)
        avg_neg_score = np.mean(neg_scores)
        
        # 计算GNN和规则贡献的相关指标
        avg_pos_gnn = np.mean(pos_gnn_scores)
        avg_neg_gnn = np.mean(neg_gnn_scores)
        
        # 打印详细评估结果
        print("\n评估结果详情:")
        print(f"  AUC: {auc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")
        print(f"  正样本平均分数: {avg_pos_score:.4f}")
        print(f"  负样本平均分数: {avg_neg_score:.4f}")
        print(f"  正样本GNN平均分数: {avg_pos_gnn:.4f}")
        print(f"  负样本GNN平均分数: {avg_neg_gnn:.4f}")
        
        # 如果需要保存预测结果
        if save:
            self._save_predictions(pos_scores, neg_scores)
        
        return {
            'auc': auc, 
            'auc_pr': auc_pr,
            'avg_pos_score': avg_pos_score,
            'avg_neg_score': avg_neg_score,
            'avg_pos_gnn': avg_pos_gnn,
            'avg_neg_gnn': avg_neg_gnn,
            'avg_pos_rule_contrib': avg_pos_rule_contrib,
            'avg_neg_rule_contrib': avg_neg_rule_contrib
        }
    
    def _save_predictions(self, pos_scores, neg_scores):
        """
        保存预测结果到文件
        """
        # 保存正样本预测
        pos_test_triplets_path = os.path.join(
            self.params.main_dir, 
            f'data/{self.params.dataset}/{self.data.file_name}.txt'
        )
        
        with open(pos_test_triplets_path) as f:
            pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
        
        pos_file_path = os.path.join(
            self.params.main_dir, 
            f'data/{self.params.dataset}/grail_art_{self.data.file_name}_predictions.txt'
        )
        
        with open(pos_file_path, "w") as f:
            for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                f.write('\t'.join([s, r, o, str(score)]) + '\n')
        
        # 保存负样本预测
        neg_test_triplets_path = os.path.join(
            self.params.main_dir, 
            f'data/{self.params.dataset}/neg_{self.data.file_name}_0.txt'
        )
        
        with open(neg_test_triplets_path) as f:
            neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
        
        neg_file_path = os.path.join(
            self.params.main_dir, 
            f'data/{self.params.dataset}/grail_art_neg_{self.data.file_name}_{self.params.constrained_neg_prob}_predictions.txt'
        )
        
        with open(neg_file_path, "w") as f:
            for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                f.write('\t'.join([s, r, o, str(score)]) + '\n')
        
        print(f"预测结果已保存到文件")
        logging.info(f"预测结果已保存到 {pos_file_path} 和 {neg_file_path}") 