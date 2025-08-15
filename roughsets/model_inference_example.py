import os
import sys
import torch
import json
import numpy as np
import dgl
import argparse
import scipy.sparse as ssp
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roughsets.rule_matcher_enhanced import EnhancedRuleMatcher
from utils.graph_utils import ssp_multigraph_to_dgl
from subgraph_extraction.datasets import SubgraphDataset

def parse_args():
    parser = argparse.ArgumentParser(description='GraIL-ART模型推理示例')
    parser.add_argument('--experiment_name', type=str, default='fb237_v1_fixed',
                       help='实验名称，用于加载保存的模型')
    parser.add_argument('--dataset', type=str, default='fb237_v1',
                       help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='roughsets/output',
                       help='规则和ID映射的输出目录')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--disable_cuda', action='store_true',
                       help='禁用CUDA')
    parser.add_argument('--enclosing_sub_graph', action='store_true',
                       help='是否使用封闭子图')
    parser.add_argument('--hop', type=int, default=2,
                       help='提取子图时跳跃的次数')
    return parser.parse_args()

def load_id_mappings(args):
    """加载实体和关系的ID映射"""
    entity2id_path = os.path.join(args.output_dir, f"{args.dataset}_entity2id.json")
    relation2id_path = os.path.join(args.output_dir, f"{args.dataset}_relation2id.json")
    id2entity_path = os.path.join(args.output_dir, f"{args.dataset}_id2entity.json")
    id2relation_path = os.path.join(args.output_dir, f"{args.dataset}_id2relation.json")
    
    # 检查文件是否存在
    if not all(os.path.exists(path) for path in [entity2id_path, relation2id_path, id2entity_path, id2relation_path]):
        print("错误：ID映射文件不完整，请先运行联合训练获取映射文件")
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
    
    print(f"成功加载ID映射: 实体数量 {len(entity2id)}, 关系数量 {len(relation2id)}")
    
    return entity2id, relation2id, id2entity, id2relation

def process_triplet(head, relation, tail, entity2id, relation2id):
    """将三元组转换为ID形式"""
    if head not in entity2id or tail not in entity2id or relation not in relation2id:
        return None
    
    return [entity2id[head], entity2id[tail], relation2id[relation]]

def create_example_subgraph(triplet, adj_list, dgl_adj_list, args, max_node_label_value):
    """为一个三元组创建示例子图"""
    from roughsets.evaluate_inductive import subgraph_extraction_labeling, prepare_features
    
    head, tail, rel = triplet
    
    # 提取子图
    nodes, node_labels = subgraph_extraction_labeling(
        (head, tail), rel, adj_list, h=args.hop,
        enclosing_sub_graph=args.enclosing_sub_graph)
    
    if len(nodes) == 0:
        print("警告：提取的子图为空")
        return None
    
    # 创建DGL子图
    subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
    subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
    subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
    
    # 检查根节点间是否有边，如果没有则添加
    edges_btw_roots = subgraph.edge_id(0, 1)
    rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)
    
    if rel_link.squeeze().nelement() == 0:
        subgraph.add_edge(0, 1)
        subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
        subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)
    
    # 准备特征
    subgraph = prepare_features(subgraph, node_labels, max_node_label_value)
    
    return subgraph

def main():
    args = parse_args()
    
    # 设置设备
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 加载最佳模型
    model_dir = os.path.join('experiments', args.experiment_name)
    gnn_model_path = os.path.join(model_dir, 'best_gnn_model.pth')
    agent_model_path = os.path.join(model_dir, 'best_agent_model.pth')
    
    print(f"加载GNN模型: {gnn_model_path}")
    gnn_model = torch.load(gnn_model_path, map_location=device)
    gnn_model.eval()
    
    print(f"加载Agent模型: {agent_model_path}")
    agent_model = torch.load(agent_model_path, map_location=device)
    agent_model.eval()
    
    # 加载规则集和ID映射
    rule_file = os.path.join(args.output_dir, f"{args.dataset}_ruleset.json")
    relation2id_path = os.path.join(args.output_dir, f"{args.dataset}_relation2id.json")
    id2relation_path = os.path.join(args.output_dir, f"{args.dataset}_id2relation.json")
    embed_file = os.path.join(args.output_dir, f"{args.dataset}_rule_embeddings.json")
    
    # 加载ID映射
    entity2id, relation2id, id2entity, id2relation = load_id_mappings(args)
    if entity2id is None:
        return
    
    # 加载规则集
    with open(rule_file, 'r') as f:
        ruleset = json.load(f)
    print(f"加载了 {len(ruleset)} 条规则")
    
    # 加载规则嵌入
    with open(embed_file, 'r') as f:
        embed_data = json.load(f)
    
    no_rule_idx = embed_data['no_rule_idx']
    
    # 初始化规则匹配器
    rule_matcher = EnhancedRuleMatcher(ruleset, device, no_rule_idx, relation2id, id2relation)
    
    # 加载知识图谱数据并构建邻接矩阵
    print("加载知识图谱数据...")
    train_path = os.path.join('data', args.dataset, 'train.txt')
    valid_path = os.path.join('data', args.dataset, 'valid.txt')
    
    if not os.path.exists(train_path):
        print(f"错误：找不到训练文件 {train_path}")
        return
    
    # 读取图数据
    triplets = []
    for file_path in [train_path, valid_path]:
        if os.path.exists(file_path):
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]
                for h, r, t in file_data:
                    if h in entity2id and t in entity2id and r in relation2id:
                        triplets.append([entity2id[h], entity2id[t], relation2id[r]])
    
    triplets = np.array(triplets)
    
    # 构建邻接矩阵列表
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets[:, 2] == i)
        adj_list.append(
            ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), 
                          (triplets[:, 0][idx].squeeze(1), triplets[:, 1][idx].squeeze(1))), 
                          shape=(len(entity2id), len(entity2id)))
        )
    
    # 转换为DGL图
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list)
    
    print("知识图谱构建完成，准备进行推理演示...")
    
    # 示例三元组用于推理
    # 这里使用人工创建的三元组，实际应用中可以从测试集中选择
    example_triplets = [
        ("/m/02mjmr", "/www.freebase.com/film/film/genre", "/m/02kdv5l"),  # 电影-类型关系
        ("/m/01mpn_s", "/www.freebase.com/music/artist/origin", "/m/0d05q6"),  # 艺术家-出生地关系
        ("/m/01c9vj", "/www.freebase.com/people/person/profession", "/m/02hrh1q")  # 人物-职业关系
    ]
    
    # 转换为ID形式
    processed_triplets = []
    for h, r, t in example_triplets:
        triplet = process_triplet(h, r, t, entity2id, relation2id)
        if triplet:
            processed_triplets.append(triplet)
    
    if not processed_triplets:
        print("没有找到有效的示例三元组，使用自动生成的三元组")
        # 从知识图谱中随机选择三元组
        if len(triplets) > 0:
            indices = np.random.choice(len(triplets), min(3, len(triplets)), replace=False)
            processed_triplets = triplets[indices].tolist()
    
    if not processed_triplets:
        print("无法生成示例三元组，退出")
        return
    
    print(f"使用 {len(processed_triplets)} 个示例三元组进行推理")
    
    # 推理
    for i, triplet in enumerate(processed_triplets):
        head, tail, rel = triplet
        
        # 显示人类可读的三元组
        head_name = id2entity.get(head, f"实体{head}")
        tail_name = id2entity.get(tail, f"实体{tail}")
        rel_name = id2relation.get(rel, f"关系{rel}")
        
        print(f"\n示例 {i+1}: ({head_name}) --[{rel_name}]--> ({tail_name})")
        
        # 创建子图
        max_node_label_value = np.array([2, 2])  # 默认最大标签值
        subgraph = create_example_subgraph(triplet, adj_list, dgl_adj_list, args, max_node_label_value)
        
        if subgraph is None:
            print("无法为此三元组创建子图，跳过")
            continue
        
        # 批处理
        batched_graph = dgl.batch([subgraph])
        r_label = torch.LongTensor([rel])
        
        # 移动到设备
        batched_graph = batched_graph.to(device)
        r_label = r_label.to(device)
        
        # 不计算梯度
        with torch.no_grad():
            # GNN前向传播
            gnn_score = gnn_model((batched_graph, r_label))
            
            # 规则匹配
            rule_ids, rule_masks = rule_matcher.match_rules(
                batched_graph, r_label, max_rules=50, apply_dropout=False)
            
            # 规则匹配结果
            matched = torch.any(rule_masks[0]).item()
            matched_count = rule_masks[0].sum().item()
            
            # Agent前向传播
            adj_values, _ = agent_model(rule_ids, rule_masks)
            
            # 应用门控机制
            gate = torch.sigmoid(adj_values)
            final_score = gnn_score * gate
            
            # 计算规则贡献率
            rule_contrib = (gate - 1.0).abs().item()
        
        # 显示结果
        print(f"  GNN分数: {gnn_score.item():.4f}")
        print(f"  门控值: {gate.item():.4f}")
        print(f"  最终分数: {final_score.item():.4f}")
        print(f"  规则匹配: {'成功' if matched else '失败'}")
        print(f"  匹配规则数: {matched_count}")
        print(f"  规则贡献率: {rule_contrib:.4f}")
        
        if matched_count > 0:
            print("  匹配的规则:")
            rule_indices = torch.where(rule_masks[0])[0].cpu().numpy()
            for idx in rule_indices:
                rule_id = rule_ids[0][idx].item()
                if rule_id != no_rule_idx:
                    rule = ruleset[rule_id]
                    if 'relation' in rule:
                        print(f"    * {rule['relation']} (置信度: {rule.get('confidence', 'N/A')})")
                    else:
                        print(f"    * 规则 #{rule_id}")

if __name__ == "__main__":
    main() 