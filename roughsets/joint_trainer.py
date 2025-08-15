import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm  # 导入tqdm用于显示进度条
import os
import json


class JointTrainer:
    """
    联合训练器，负责GNN和Agent的多任务训练
    """
    def __init__(self, params, graph_classifier, transformer_agent, rule_matcher, train_data, joint_evaluator=None):
        """
        初始化联合训练器
        
        参数:
            params: 训练参数
            graph_classifier: GNN模型
            transformer_agent: Transformer Agent模型
            rule_matcher: 规则匹配器
            train_data: 训练数据
            joint_evaluator: 联合评估器
        """
        self.params = params
        self.graph_classifier = graph_classifier
        self.transformer_agent = transformer_agent
        self.rule_matcher = rule_matcher
        self.train_data = train_data
        self.joint_evaluator = joint_evaluator
        
        self.updates_counter = 0
        
        # 创建优化器
        model_params = list(self.graph_classifier.parameters()) + list(self.transformer_agent.parameters())
        logging.info(f'总参数数量: {sum(map(lambda x: x.numel(), model_params))}')
        
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=0.9, weight_decay=params.l2)
        elif params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=params.l2)
        else:
            raise ValueError(f"不支持的优化器: {params.optimizer}")
        
        # 排名损失
        self.ranking_criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        
        # 初始化训练状态
        self.reset_training_state()
    
    def reset_training_state(self):
        """重置训练状态"""
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0
    
    def train_epoch(self):
        """训练一个epoch"""
        total_loss = 0
        total_rank_loss = 0
        total_aux_loss = 0
        total_reg_loss = 0
        total_rule_contrib = 0
        total_match_rate = 0
        batch_count = 0
        
        all_labels = []
        all_scores = []
        
        dataloader = DataLoader(
            self.train_data, 
            batch_size=self.params.batch_size, 
            shuffle=True, 
            num_workers=self.params.num_workers, 
            collate_fn=self.params.collate_fn
        )
        
        # 设置模型为训练模式
        self.graph_classifier.train()
        self.transformer_agent.train()
        
        # 使用tqdm创建进度条
        pbar = tqdm(dataloader, desc="训练批次", ncols=100)
        batch_losses = []
        
        for batch_idx, batch in enumerate(pbar):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 1. GNN前向传播获取基础分数
            score_pos_gnn = self.graph_classifier(data_pos)
            score_neg_gnn = self.graph_classifier(data_neg)
            
            # 2. 规则匹配
            pos_relations = data_pos[1]  # 假设数据中关系ID位于索引1
            neg_relations = data_neg[1]
            
            # 应用规则dropout
            pos_rule_ids, pos_rule_masks, neg_rule_ids_list, neg_rule_masks_list = self.rule_matcher.match_rule_batched(
                data_pos, pos_relations, data_neg, neg_relations, 
                max_rules=50, apply_dropout=True, dropout_prob=self.params.rule_dropout
            )
            
            # 统计规则匹配率
            total_samples = len(pos_rule_masks)
            matched_samples = sum([1 for mask in pos_rule_masks if torch.any(mask).item()])
            rule_match_rate = matched_samples / total_samples if total_samples > 0 else 0
            total_match_rate += rule_match_rate
            
            # 3. Agent前向传播获取调整值
            pos_adj, pos_rule_scores = self.transformer_agent(pos_rule_ids, pos_rule_masks)
            
            # 处理负样本
            neg_adj_list = []
            neg_rule_scores_list = []
            
            for neg_rule_ids, neg_rule_masks in zip(neg_rule_ids_list, neg_rule_masks_list):
                neg_adj, neg_rule_scores = self.transformer_agent(neg_rule_ids, neg_rule_masks)
                neg_adj_list.append(neg_adj)
                neg_rule_scores_list.append(neg_rule_scores)
            
            # 4. 融合得分
            # 使用sigmoid门控机制结合GNN得分和Agent调整
            pos_gate = torch.sigmoid(pos_adj)
            score_pos_final = score_pos_gnn * pos_gate
            
            score_neg_final_list = []
            for neg_idx, neg_adj in enumerate(neg_adj_list):
                neg_gate = torch.sigmoid(neg_adj)
                score_neg_final = score_neg_gnn[neg_idx:neg_idx+1] * neg_gate
                score_neg_final_list.append(score_neg_final)
            
            # 堆叠所有负样本的最终得分
            score_neg_final = torch.cat(score_neg_final_list, dim=0)
            
            # 5. 计算损失
            # 排名损失
            ranking_loss = self.ranking_criterion(
                score_pos_final,
                score_neg_final.view(len(score_pos_final), -1).mean(dim=1),
                torch.ones(len(score_pos_final), device=self.params.device)
            )
            
            # 辅助损失（规则预测）
            aux_loss = self.transformer_agent.compute_aux_loss(pos_rule_scores, pos_rule_masks)
            for neg_rule_scores, neg_rule_masks in zip(neg_rule_scores_list, neg_rule_masks_list):
                aux_loss += self.transformer_agent.compute_aux_loss(neg_rule_scores, neg_rule_masks)
            aux_loss = aux_loss / (1 + len(neg_rule_scores_list))  # 平均辅助损失
            
            # 正则化损失
            reg_loss = self.transformer_agent.compute_reg_loss()
            
            # 总损失
            loss = ranking_loss + self.params.aux_loss_weight * aux_loss + self.params.reg_loss_weight * reg_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.graph_classifier.parameters(), self.params.clip)
            torch.nn.utils.clip_grad_norm_(self.transformer_agent.parameters(), self.params.clip)
            
            # 更新参数
            self.optimizer.step()
            self.updates_counter += 1
            
            # 统计信息
            with torch.no_grad():
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                total_loss += batch_loss
                total_rank_loss += ranking_loss.item()
                total_aux_loss += aux_loss.item()
                total_reg_loss += reg_loss.item()
                
                all_scores += score_pos_final.squeeze().detach().cpu().tolist() + score_neg_final.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                
                # 计算规则贡献率
                pos_contribution = (pos_gate - 1.0).abs().mean().item()
                rule_contribution = pos_contribution
                total_rule_contrib += rule_contribution
                
                # 更新进度条信息
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'rank_loss': f'{ranking_loss.item():.4f}',
                    'aux_loss': f'{aux_loss.item():.4f}',
                    'rule_contrib': f'{rule_contribution:.4f}',
                    'match_rate': f'{rule_match_rate:.2f}'
                })
            
            batch_count += 1
            
            # 定期评估
            if self.joint_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                print("\n正在评估模型...")
                result = self.joint_evaluator.eval()
                eval_time = time.time() - tic
                
                print(f"验证结果: AUC: {result['auc']:.4f}, AUC-PR: {result['auc_pr']:.4f}, 耗时: {eval_time:.2f}s")
                logging.info(f'\n性能: {str(result)} 耗时: {str(eval_time)}')
                
                if result['auc'] >= self.best_metric:
                    self.save_models()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0
                    print(f"发现更好的模型，已保存! 最佳AUC: {self.best_metric:.4f}")
                else:
                    self.not_improved_count += 1
                    print(f"模型性能未提升 ({self.not_improved_count}/{self.params.early_stop})")
        
        # 计算平均损失和性能指标
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_aux_loss = total_aux_loss / batch_count if batch_count > 0 else 0
        avg_rule_contrib = total_rule_contrib / batch_count if batch_count > 0 else 0
        avg_match_rate = total_match_rate / batch_count if batch_count > 0 else 0
        
        # 计算AUC
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            auc = metrics.roc_auc_score(all_labels, all_scores)
            auc_pr = metrics.average_precision_score(all_labels, all_scores)
        else:
            auc = 0
            auc_pr = 0
        
        return avg_loss, auc, auc_pr, avg_aux_loss, avg_rule_contrib, avg_match_rate
    
    def train(self):
        """训练模型"""
        self.reset_training_state()
        
        # 记录训练历史
        history = {
            'epochs': [],
            'train_losses': [],
            'aux_losses': [],
            'rule_contributions': [],
            'rule_match_rates': [],
            'valid_metrics': []
        }
        
        # 开始训练循环
        for epoch in range(1, self.params.num_epochs + 1):
            start_time = time.time()
            
            # 训练一个epoch
            epoch_loss, epoch_auc, epoch_auc_pr, epoch_aux_loss, rule_contrib, rule_match_rate = self.train_epoch()
            
            # 记录训练时间
            train_time = time.time() - start_time
            
            # 记录训练历史
            history['epochs'].append(epoch)
            history['train_losses'].append(epoch_loss)
            history['aux_losses'].append(epoch_aux_loss)
            history['rule_contributions'].append(rule_contrib)
            history['rule_match_rates'].append(rule_match_rate)
            
            # 定期评估
            if self.joint_evaluator and epoch % self.params.eval_every == 0:
                print("\n正在评估模型...")
                start_time = time.time()
                val_metrics = self.joint_evaluator.eval()
                eval_time = time.time() - start_time
                
                # 记录验证指标
                history['valid_metrics'].append(val_metrics)
                
                print(f"Epoch {epoch}/{self.params.num_epochs}, 验证结果: AUC: {val_metrics['auc']:.4f}, AUC-PR: {val_metrics['auc_pr']:.4f}")
                
                # 修复KeyError: 'mrr'错误，使用实际存在的键
                if 'mrr' in val_metrics and 'hits@1' in val_metrics:
                    logging.info(f"\nEpoch {epoch}, 验证指标: MRR: {val_metrics['mrr']:.4f}, Hits@1: {val_metrics['hits@1']:.4f}")
                else:
                    logging.info(f"\nEpoch {epoch}, 验证指标: AUC: {val_metrics['auc']:.4f}, AUC-PR: {val_metrics['auc_pr']:.4f}")
                
                # 检查是否提升
                if val_metrics['auc'] > self.best_metric:
                    self.best_metric = val_metrics['auc']
                    self.not_improved_count = 0
                    print(f"发现更好的模型，AUC: {self.best_metric:.4f}")
                    
                    # 保存最佳检查点
                    self.save_checkpoint(epoch)
                else:
                    self.not_improved_count += 1
                    print(f"模型性能未提升 ({self.not_improved_count}/{self.params.early_stop})")
                    
                    # 检查早停
                    if self.not_improved_count >= self.params.early_stop:
                        print(f"\n连续 {self.params.early_stop} 轮未提升，早停")
                        break
            
            # 打印训练信息
            print(f"\nEpoch {epoch}/{self.params.num_epochs}, 训练损失: {epoch_loss:.4f}, 辅助损失: {epoch_aux_loss:.6f}")
            print(f"规则贡献率: {rule_contrib:.6f}, 规则匹配率: {rule_match_rate:.4f}")
            print(f"训练AUC: {epoch_auc:.4f}, 训练AUC-PR: {epoch_auc_pr:.4f}")
            print(f"训练耗时: {train_time:.2f}s")
            
            logging.info(f"\nEpoch {epoch}, 训练损失: {epoch_loss:.4f}, 辅助损失: {epoch_aux_loss:.6f}")
            logging.info(f"规则贡献率: {rule_contrib:.6f}")
            
            # 定期保存检查点
            if epoch % self.params.save_every == 0:
                self.save_checkpoint(epoch)
        
        # 训练完成后，保存训练历史
        history_path = os.path.join(self.params.exp_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # 将numpy数组转换为列表以便JSON序列化
            for key in history:
                history[key] = [x if not isinstance(x, (np.ndarray, np.number)) else x.tolist() 
                              for x in history[key]]
            json.dump(history, f)
        
        print(f"\n训练历史已保存至 {history_path}")
        
        return history
    
    def save_checkpoint(self, epoch):
        """
        保存模型检查点
        """
        torch.save(self.graph_classifier, f'{self.params.exp_dir}/gnn_checkpoint_epoch_{epoch}.pth')
        torch.save(self.transformer_agent, f'{self.params.exp_dir}/agent_checkpoint_epoch_{epoch}.pth')
        logging.info(f'保存了Epoch {epoch}的检查点')
    
    def save_models(self):
        """
        保存最佳模型
        """
        torch.save(self.graph_classifier, f'{self.params.exp_dir}/best_gnn_model.pth')
        torch.save(self.transformer_agent, f'{self.params.exp_dir}/best_agent_model.pth')
        logging.info('发现更好的模型，已保存！') 