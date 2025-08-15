import os
import sys
import argparse
import logging
import torch
import json
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
from scipy.stats import truncnorm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets, process_files, save_mappings
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model
from roughsets.rule_matcher_enhanced import EnhancedRuleMatcher
from roughsets.agent_transformer import TransformerAgent
from roughsets.joint_trainer import JointTrainer
from roughsets.joint_evaluator import JointEvaluator
from roughsets.subgraph_generator import generate_and_save_subgraphs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GraIL-ART第二阶段: 联合多任务训练')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='FB237_v1', 
                       help='数据集名称: FB237_v1, FB237_v2, FB237_v3, FB237_v4')
    parser.add_argument('--train_file', type=str, default='train', 
                       help='训练数据文件名')
    parser.add_argument('--valid_file', type=str, default='valid', 
                       help='验证数据文件名')
    parser.add_argument('--test_file', type=str, default='test', 
                       help='测试数据文件名')
    parser.add_argument('--rule_file', type=str, default=None, 
                       help='规则文件路径')
    parser.add_argument('--embedding_file', type=str, default=None, 
                       help='规则嵌入文件路径')
    
    # 子图生成参数
    parser.add_argument('--hop', type=int, default=2, help='子图的最大跳数')
    parser.add_argument('--enclosing_sub_graph', action='store_true', help='是否使用封闭子图')
    parser.add_argument('--add_traspose_rels', action='store_true', help='是否添加反向关系')
    parser.add_argument('--num_neg_samples_per_link', type=int, default=1, 
                       help='每个链接的负样本数量')
    parser.add_argument('--max_links', type=int, default=1000000,
                       help='设置训练链接的最大数量（以适应内存）')
    parser.add_argument('--constrained_neg_prob', type=float, default=0.0,
                       help='约束负采样的概率')
    parser.add_argument('--max_nodes_per_hop', type=int, default=None,
                       help='每跳最大节点数（通过子采样限制）')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载的进程数')
    
    # 路径参数
    parser.add_argument('--main_dir', type=str, default='./', help='项目主目录')
    parser.add_argument('--output_dir', type=str, default='roughsets/output', help='输出目录')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='模型保存目录')
    parser.add_argument('--experiment_name', type=str, default='joint_training', help='实验名称')
    parser.add_argument('--load_model', action='store_true', help='是否加载已有模型')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='Adam', help='优化器')
    parser.add_argument('--early_stop', type=int, default=10, help='早停轮数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
    parser.add_argument('--eval_every', type=int, default=3, help='每隔多少轮评估模型')
    parser.add_argument('--eval_every_iter', type=int, default=455, help='每隔多少迭代评估模型')
    parser.add_argument('--save_every', type=int, default=10, help='每隔多少轮保存模型检查点')
    parser.add_argument('--clip', type=int, default=1000, help='最大梯度范数')
    parser.add_argument('--l2', type=float, default=5e-4, help='GNN权重的正则化常数')
    parser.add_argument('--margin', type=float, default=10, help='max-margin损失中正例和负例之间的边界')
    
    # GNN模型参数
    parser.add_argument('--num_gcn_layers', type=int, default=3, help='GCN层数')
    parser.add_argument('--gnn_agg_type', type=str, default='sum', choices=['sum', 'mlp', 'gru'], help='GNN聚合类型')
    parser.add_argument('--use_edge_features', action='store_true', help='是否使用边特征')
    parser.add_argument('--rel_emb_dim', type=int, default=32, help='关系嵌入大小')
    parser.add_argument('--attn_rel_emb_dim', type=int, default=32, help='注意力机制的关系嵌入大小')
    parser.add_argument('--emb_dim', type=int, default=32, help='实体嵌入大小')
    parser.add_argument('--num_bases', type=int, default=4, help='GCN权重的基函数数量')
    parser.add_argument('--edge_dropout', type=float, default=0.5, help='子图边的dropout率')
    parser.add_argument('--add_ht_emb', type=bool, default=True, help='是否将头/尾嵌入与池化的图表示连接起来')
    parser.add_argument('--has_attn', type=bool, default=True, help='模型中是否有注意力机制')
    
    # KGE参数
    parser.add_argument('--use_kge_embeddings', action='store_true', help='是否使用知识图谱嵌入')
    parser.add_argument('--kge_model', type=str, default='TransE', help='KGE模型类型')
    
    # 设备参数
    parser.add_argument('--disable_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    
    # 规则Agent参数
    parser.add_argument('--rule_dropout', type=float, default=0.1,
                       help='规则dropout概率')
    parser.add_argument('--transformer_layers', type=int, default=2,
                       help='Transformer层数')
    parser.add_argument('--transformer_heads', type=int, default=4,
                       help='Transformer头数')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Agent隐藏层维度')
    
    # 损失权重参数
    parser.add_argument('--aux_loss_weight', type=float, default=0.3,
                       help='辅助规则预测损失的权重')
    parser.add_argument('--reg_loss_weight', type=float, default=0.01,
                       help='Agent正则化损失的权重')
    
    # ID映射参数
    parser.add_argument('--use_saved_mappings', type=bool, default=True,
                       help='是否使用保存的ID映射')
    
    return parser.parse_args()


def load_id_mappings(args):
    """加载实体和关系的ID映射"""
    entity2id_path = os.path.join(args.output_dir, f"{args.dataset}_entity2id.json")
    relation2id_path = os.path.join(args.output_dir, f"{args.dataset}_relation2id.json")
    id2entity_path = os.path.join(args.output_dir, f"{args.dataset}_id2entity.json")
    id2relation_path = os.path.join(args.output_dir, f"{args.dataset}_id2relation.json")
    
    # 检查文件是否存在
    if not all(os.path.exists(path) for path in [entity2id_path, relation2id_path, id2entity_path, id2relation_path]):
        logging.warning("ID映射文件不完整，将使用默认映射")
        return None, None, None, None
    
    # 加载映射
    with open(entity2id_path, 'r') as f:
        entity2id = json.load(f)
    
    with open(relation2id_path, 'r') as f:
        relation2id = json.load(f)
    
    with open(id2entity_path, 'r') as f:
        id2entity_str = json.load(f)
        # 将字符串键转换回整数
        id2entity = {int(k): v for k, v in id2entity_str.items()}
    
    with open(id2relation_path, 'r') as f:
        id2relation_str = json.load(f)
        # 将字符串键转换回整数
        id2relation = {int(k): v for k, v in id2relation_str.items()}
    
    logging.info(f"成功加载ID映射: 实体数量 {len(entity2id)}, 关系数量 {len(relation2id)}")
    
    return entity2id, relation2id, id2entity, id2relation


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 如果规则文件和嵌入文件没有指定，使用默认路径
    if args.rule_file is None:
        args.rule_file = os.path.join(args.output_dir, f"{args.dataset}_ruleset.json")
    
    if args.embedding_file is None:
        args.embedding_file = os.path.join(args.output_dir, f"{args.dataset}_rule_embeddings.json")
    
    # 检查必要文件是否存在
    if not os.path.exists(args.rule_file):
        logging.error(f"规则文件不存在: {args.rule_file}")
        return
        
    if not os.path.exists(args.embedding_file):
        logging.error(f"嵌入文件不存在: {args.embedding_file}")
        return
    
    # 初始化实验
    initialize_experiment(args, __file__)
    
    # 设置路径 - 支持大小写不敏感的数据集名称
    dataset_folder = args.dataset
    if args.dataset.lower() == 'wn18rr':
        dataset_folder = 'WN18RR'  # 确保目录名称正确
        logging.info(f"使用WN18RR数据集，路径调整为: data/{dataset_folder}/")
    elif args.dataset.lower() in ['fb237_v1', 'fb237_v2', 'fb237_v3', 'fb237_v4']:
        dataset_folder = args.dataset.lower()
        logging.info(f"使用FB237数据集，路径调整为: data/{dataset_folder}/")
    
    args.file_paths = {
        'train': os.path.join(args.main_dir, f'data/{dataset_folder}/{args.train_file}.txt'),
        'valid': os.path.join(args.main_dir, f'data/{dataset_folder}/{args.valid_file}.txt')
    }
    
    # 检查文件路径是否存在
    for key, path in args.file_paths.items():
        if not os.path.exists(path):
            logging.error(f"{key}文件不存在: {path}")
            return
    
    # 设置设备
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
        
    # 设置数据处理函数
    args.collate_fn = collate_dgl
    args.move_batch_to_device = move_batch_to_device_dgl
    
    # 数据库路径
    args.db_path = os.path.join(args.main_dir, f'data/{args.dataset}/subgraphs_en_{args.enclosing_sub_graph}_neg_{args.num_neg_samples_per_link}_hop_{args.hop}')
    
    # 加载ID映射
    entity2id, relation2id, id2entity, id2relation = None, None, None, None
    if args.use_saved_mappings:
        entity2id, relation2id, id2entity, id2relation = load_id_mappings(args)
    
    # 如果子图数据集不存在，则生成
    if not os.path.isdir(args.db_path):
        # 如果有保存的映射，使用它们生成子图
        if args.use_saved_mappings and relation2id is not None:
            logging.info("使用保存的关系ID映射生成子图数据集")
            generate_subgraph_datasets(args, saved_relation2id=relation2id)
        else:
            logging.info("使用默认映射生成子图数据集")
            generate_subgraph_datasets(args)
            
            # 生成后保存映射，以便后续使用
            if args.use_saved_mappings:
                # 重新处理文件以获取映射
                adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(args.file_paths)
                save_mappings(entity2id, relation2id, id2entity, id2relation, args.output_dir, args.dataset)
    
    # 加载训练和验证数据集
    train_data = SubgraphDataset(args.db_path, 'train_pos', 'train_neg', args.file_paths,
                             add_traspose_rels=args.add_traspose_rels,
                             num_neg_samples_per_link=args.num_neg_samples_per_link,
                             use_kge_embeddings=args.use_kge_embeddings, dataset=args.dataset,
                             kge_model=args.kge_model, file_name=args.train_file)
                             
    valid_data = SubgraphDataset(args.db_path, 'valid_pos', 'valid_neg', args.file_paths,
                             add_traspose_rels=args.add_traspose_rels,
                             num_neg_samples_per_link=args.num_neg_samples_per_link,
                             use_kge_embeddings=args.use_kge_embeddings, dataset=args.dataset,
                             kge_model=args.kge_model, file_name=args.valid_file)
    
    # 设置模型参数
    args.num_rels = train_data.num_rels
    args.aug_num_rels = train_data.aug_num_rels
    args.inp_dim = train_data.n_feat_dim
    args.max_label_value = train_data.max_n_label
    
    # 初始化GNN模型
    graph_classifier = initialize_model(args, dgl_model, args.load_model)
    
    # 加载规则嵌入
    with open(args.embedding_file, 'r') as f:
        embed_data = json.load(f)
    
    rule_embeds = np.array(embed_data['rule_embeds'])
    relation_embeds = np.array(embed_data['relation_embeds'])
    no_rule_idx = embed_data['no_rule_idx']
    
    # 加载规则集
    with open(args.rule_file, 'r') as f:
        ruleset = json.load(f)
    
    # 加载关系ID映射
    relation2id_path = os.path.join(args.output_dir, f"{args.dataset}_relation2id.json")
    id2relation_path = os.path.join(args.output_dir, f"{args.dataset}_id2relation.json")
    
    relation2id = None
    id2relation = None
    
    if os.path.exists(relation2id_path) and os.path.exists(id2relation_path):
        with open(relation2id_path, 'r') as f:
            relation2id = json.load(f)
        
        with open(id2relation_path, 'r') as f:
            id2relation_str = json.load(f)
            id2relation = {int(k): v for k, v in id2relation_str.items()}
            
        logging.info(f"成功加载关系ID映射: 共{len(relation2id)}个关系")
    else:
        logging.warning("未找到关系ID映射文件，将创建临时映射")
        # 尝试从规则文件中提取关系路径
        rel_paths = set()
        for rule in ruleset:
            if 'relation' in rule:
                rel_paths.add(rule['relation'])
        
        if rel_paths:
            logging.info(f"从规则文件中提取了{len(rel_paths)}个关系路径")
            relation2id = {path: i for i, path in enumerate(rel_paths)}
            id2relation = {i: path for i, path in enumerate(rel_paths)}
        else:
            logging.error("无法创建关系映射，规则匹配将受到严重影响")
    
    # 调整关键参数，确保规则贡献率提高
    if args.rule_dropout > 0.2:
        logging.warning(f"规则dropout过高({args.rule_dropout})，可能导致规则贡献率低")
        logging.info("将规则dropout调整为0.1")
        args.rule_dropout = 0.1
    
    if args.aux_loss_weight < 0.2:
        logging.warning(f"辅助损失权重过低({args.aux_loss_weight})，可能导致规则学习不足")
        logging.info("将辅助损失权重调整为0.3")
        args.aux_loss_weight = 0.3
    
    # 初始化增强版规则匹配器
    logging.info("初始化增强版规则匹配器...")
    rule_matcher = EnhancedRuleMatcher(ruleset, args.device, no_rule_idx, relation2id, id2relation)
    
    # 测试规则匹配效果
    logging.info("测试规则匹配效果...")
    test_rel_ids = list(id2relation.keys())[:5] if id2relation else []
    if test_rel_ids:
        rel_tensor = torch.tensor(test_rel_ids, device=args.device)
        dummy_graphs = [None] * len(test_rel_ids)
        _, rule_masks = rule_matcher.match_rules(dummy_graphs, rel_tensor, max_rules=5)
        match_count = sum(1 for mask in rule_masks if torch.any(mask).item())
        match_rate = match_count / len(test_rel_ids) if test_rel_ids else 0
        logging.info(f"规则匹配测试: {match_count}/{len(test_rel_ids)} 匹配成功, 匹配率: {match_rate:.2f}")
    
    # 初始化Transformer Agent
    transformer_agent = TransformerAgent(
        input_dim=rule_embeds.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=1,  # 输出为标量调整值
        num_layers=args.transformer_layers,
        num_heads=args.transformer_heads,
        dropout=args.dropout,
        max_seq_len=50,  # 假设每个关系最多50条规则
        no_rule_idx=no_rule_idx,
        device=args.device
    )
    
    # 将规则嵌入矩阵注册为模型参数但不参与梯度更新
    rule_embeds_tensor = torch.FloatTensor(rule_embeds).to(args.device)
    transformer_agent.register_rule_embeddings(rule_embeds_tensor)
    
    # 初始化联合评估器
    joint_evaluator = JointEvaluator(
        args,
        graph_classifier,
        transformer_agent,
        rule_matcher,
        valid_data
    )
    
    # 初始化联合训练器
    joint_trainer = JointTrainer(
        args,
        graph_classifier,
        transformer_agent,
        rule_matcher,
        train_data,
        joint_evaluator
    )
    
    # 开始训练
    logging.info('开始联合训练...')
    print("\n" + "="*50)
    print("GraIL-ART 第二阶段: 联合多任务训练")
    print("="*50)
    print(f"数据集: {args.dataset}")
    print(f"实验名称: {args.experiment_name}")
    print(f"设备: {args.device}")
    print(f"批处理大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"优化器: {args.optimizer}")
    print(f"GNN层数: {args.num_gcn_layers}")
    print(f"Transformer层数: {args.transformer_layers}")
    print(f"Transformer头数: {args.transformer_heads}")
    print(f"规则dropout概率: {args.rule_dropout}")
    print(f"辅助损失权重: {args.aux_loss_weight}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"早停轮数: {args.early_stop}")
    print("="*50 + "\n")
    joint_trainer.train()
    
    logging.info('训练完成。保存最终模型...')
    print("\n训练完成！保存最终模型...")
    
    # 保存最终模型
    torch.save(graph_classifier, os.path.join(args.exp_dir, 'final_gnn_model.pth'))
    torch.save(transformer_agent, os.path.join(args.exp_dir, 'final_agent_model.pth'))
    
    logging.info('模型已保存。')
    print(f"模型已保存到: {args.exp_dir}")
    print("="*50)


if __name__ == "__main__":
    main() 