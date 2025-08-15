import os
import sys
import random
import argparse
import logging
import json
import time
import torch
import numpy as np
import dgl
import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import ssp_multigraph_to_dgl
from roughsets.rule_matcher_enhanced import EnhancedRuleMatcher

def parse_args():
    parser = argparse.ArgumentParser(description='GraIL-ART第三阶段: 归纳关系预测评估')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='fb237_v1_ind', 
                        help='归纳测试数据集名称，通常使用_ind后缀版本')
    parser.add_argument('--model_dataset', type=str, default='fb237_v1', 
                        help='模型训练时使用的数据集名称')
    parser.add_argument('--experiment_name', type=str, default='fb237_v1_fixed', 
                        help='实验名称，用于加载保存的模型')
    
    # 模型参数
    parser.add_argument('--use_kge_embeddings', '-kge', action='store_true',
                        help='是否使用预训练的KGE嵌入')
    parser.add_argument('--kge_model', type=str, default='TransE',
                        help='使用哪种KGE模型加载实体嵌入')
    parser.add_argument('--enclosing_sub_graph', '-en', action='store_true',
                        help='是否只考虑封闭子图')
    parser.add_argument('--hop', type=int, default=2,
                        help='提取子图时跳跃的次数')
    parser.add_argument('--add_traspose_rels', '-tr', action='store_true',
                        help='是否添加对称关系的邻接矩阵')
    parser.add_argument('--num_neg_samples_per_link', '-neg', type=int, default=49,
                        help='每个链接的负采样数量')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='roughsets/output',
                        help='输出目录，用于存储规则和ID映射')
    
    # 设备参数
    parser.add_argument('--disable_cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的GPU编号')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载的进程数量')
    
    return parser.parse_args()

def process_files(files, saved_relation2id=None, add_traspose_rels=False):
    """
    处理三元组文件并构建图结构
    
    参数:
        files: 文件路径字典
        saved_relation2id: 保存的关系ID映射
        add_traspose_rels: 是否添加转置关系
        
    返回:
        adj_list: 邻接矩阵列表
        dgl_adj_list: DGL邻接矩阵列表
        triplets: 三元组字典
        entity2id: 实体ID映射
        relation2id: 关系ID映射
        id2entity: ID到实体映射
        id2relation: ID到关系映射
    """
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id
    
    triplets = {}
    
    ent = 0
    rel = 0
    
    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]
        
        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
                
            # 仅保存已知关系的三元组
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])
        
        triplets[file_type] = np.array(data)
    
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    
    # 构造每个关系对应的邻接矩阵列表
    adj_list = []
    for i in range(len(saved_relation2id)):
        if 'graph' in triplets:  # 确保有graph文件
            idx = np.argwhere(triplets['graph'][:, 2] == i)
            if len(idx) > 0:
                adj_list.append(ssp.csc_matrix(
                    (np.ones(len(idx), dtype=np.uint8), 
                    (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), 
                    shape=(len(entity2id), len(entity2id))))
            else:
                adj_list.append(ssp.csc_matrix((len(entity2id), len(entity2id))))
        else:
            adj_list.append(ssp.csc_matrix((len(entity2id), len(entity2id))))
    
    # 添加转置矩阵以处理关系的两个方向
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t
    
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)
    
    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation

def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):
    """
    通过替换头尾实体生成负样本
    """
    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]
    
    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        
        # 添加原始正样本作为头部替换的第一个样本
        neg_triplet['head'][0].append([head, tail, rel])
        
        # 为头部替换生成负样本
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)
            
            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])
        
        # 添加原始正样本作为尾部替换的第一个样本
        neg_triplet['tail'][0].append([head, tail, rel])
        
        # 为尾部替换生成负样本
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            
            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])
        
        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])
        
        neg_triplets.append(neg_triplet)
    
    return neg_triplets

def incidence_matrix(adj_list):
    """
    计算关联矩阵
    
    参数:
        adj_list: 邻接矩阵列表
    """
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)

def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    多边类型图的BFS。返回层级集列表。
    """
    visited = set()
    current_lvl = set(roots)
    
    next_lvl = set()
    
    while current_lvl:
        for v in current_lvl:
            visited.add(v)
        
        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # 集合差
        
        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))
        
        yield next_lvl
        
        current_lvl = set.union(next_lvl)

def _get_neighbors(adj, nodes):
    """
    获取节点集合在图中的邻居集合
    """
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # 转换为索引集合
    return neighbors

def _sp_row_vec_from_idx_list(idx_list, dim):
    """
    从索引列表创建稀疏向量
    """
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    """
    获取节点的邻居节点
    """
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls) if lvls else set()

def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None):
    """
    提取并标记子图
    """
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T
    
    # 获取两个根节点的h-hop邻居
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)
    
    # 计算交集和并集
    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    
    # 提取子图
    if enclosing_sub_graph:
        # 使用封闭子图（两根节点的共同邻居）
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        # 使用完整子图（两根节点的所有邻居）
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
    
    # 提取子图邻接矩阵
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    
    # 节点标签计算
    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)
    
    # 裁剪子图
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    
    return pruned_subgraph_nodes, pruned_labels

def remove_nodes(A_incidence, nodes):
    """
    从关联矩阵中移除节点
    """
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]

def node_label_new(subgraph, max_distance=1):
    """
    双半径节点标记(DRNL)的实现
    """
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
    
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    """
    准备节点特征
    """
    # 对节点标签特征进行one-hot编码，并与n_feats连接
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
    
    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)
    
    return subgraph

def get_kge_embeddings(dataset, kge_model):
    """
    获取KGE嵌入
    """
    path = os.path.join('./experiments/kge_baselines', f'{kge_model}_{dataset}')
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}
    
    return node_features, kge_entity2id

def evaluate_model(args):
    """
    评估模型在归纳性任务上的性能
    """
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
    rule_file = os.path.join(args.output_dir, f"{args.model_dataset}_ruleset.json")
    relation2id_path = os.path.join(args.output_dir, f"{args.model_dataset}_relation2id.json")
    id2relation_path = os.path.join(args.output_dir, f"{args.model_dataset}_id2relation.json")
    embed_file = os.path.join(args.output_dir, f"{args.model_dataset}_rule_embeddings.json")
    
    # 检查文件是否存在
    for file_path in [rule_file, relation2id_path, id2relation_path, embed_file]:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return
    
    # 加载规则集
    with open(rule_file, 'r') as f:
        ruleset = json.load(f)
    print(f"加载了 {len(ruleset)} 条规则")
    
    # 加载关系映射
    with open(relation2id_path, 'r') as f:
        relation2id = json.load(f)
    
    with open(id2relation_path, 'r') as f:
        id2relation_str = json.load(f)
        id2relation = {int(k): v for k, v in id2relation_str.items()}
    
    print(f"加载了 {len(relation2id)} 个关系映射")
    
    # 加载规则嵌入
    with open(embed_file, 'r') as f:
        embed_data = json.load(f)
    
    no_rule_idx = embed_data['no_rule_idx']
    
    # 初始化规则匹配器
    print("初始化增强版规则匹配器...")
    rule_matcher = EnhancedRuleMatcher(ruleset, device, no_rule_idx, relation2id, id2relation)
    
    # 加载KGE嵌入(如果使用)
    node_features = None
    kge_entity2id = None
    if args.use_kge_embeddings:
        print(f"加载KGE嵌入: {args.kge_model}")
        node_features, kge_entity2id = get_kge_embeddings(args.model_dataset, args.kge_model)
    
    # 设置文件路径 - 支持不同的数据集格式
    dataset_folder = args.dataset
    if args.dataset.lower() == 'wn18rr':
        dataset_folder = 'WN18RR'
        print(f"使用WN18RR数据集，路径调整为: ./data/{dataset_folder}/")
    elif args.dataset.lower() in ['fb237_v1', 'fb237_v2', 'fb237_v3', 'fb237_v4']:
        dataset_folder = args.dataset.lower()
        print(f"使用FB237数据集，路径调整为: ./data/{dataset_folder}/")
    
    file_paths = {
        'graph': os.path.join('./data', dataset_folder, 'train.txt'),
        'links': os.path.join('./data', dataset_folder, 'test.txt')
    }
    
    # 检查文件是否存在
    for key, path in file_paths.items():
        if not os.path.exists(path):
            print(f"错误: {key}文件不存在 {path}")
            return
    
    # 处理文件
    print("处理文件并构建图结构...")
    adj_list, dgl_adj_list, triplets, entity2id, _, id2entity, _ = process_files(
        file_paths, relation2id, args.add_traspose_rels)
    
    print(f"处理了 {len(triplets['links'])} 个测试链接")
    
    # 生成负样本
    print(f"生成每个链接 {args.num_neg_samples_per_link} 个负样本...")
    neg_triplets = get_neg_samples_replacing_head_tail(
        triplets['links'], adj_list, args.num_neg_samples_per_link)
    
    # 设置模型参数
    gnn_model.eval()
    agent_model.eval()
    
    # 初始化评估指标
    all_ranks = []
    all_rec_ranks = []
    all_rule_contribs = []
    all_rule_matches = 0
    total_samples = 0
    
    print("\n开始评估...")
    
    # 批量处理负样本
    for i, neg_triplet in tqdm(enumerate(neg_triplets), total=len(neg_triplets), desc="处理测试三元组"):
        head_neg_links = neg_triplet['head'][0]#把正样本放在第一位
        tail_neg_links = neg_triplet['tail'][0]#负样本第一位
        
        # 处理头部替换负样本
        head_scores, head_rule_contrib, head_match_rate = process_batch(
            head_neg_links, adj_list, dgl_adj_list, gnn_model, agent_model, rule_matcher,
            args, id2entity, node_features, kge_entity2id)
        
        # 处理尾部替换负样本
        tail_scores, tail_rule_contrib, tail_match_rate = process_batch(
            tail_neg_links, adj_list, dgl_adj_list, gnn_model, agent_model, rule_matcher,
            args, id2entity, node_features, kge_entity2id)
        
        # 计算排名
        head_rank = np.where(np.argsort(-head_scores) == 0)[0][0] + 1
        tail_rank = np.where(np.argsort(-tail_scores) == 0)[0][0] + 1
        
        # 计算倒数排名(MRR计算用)
        head_rec_rank = 1.0 / head_rank
        tail_rec_rank = 1.0 / tail_rank
        
        # 收集结果
        all_ranks.append(head_rank)
        all_ranks.append(tail_rank)
        all_rec_ranks.append(head_rec_rank)
        all_rec_ranks.append(tail_rec_rank)
        all_rule_contribs.append((head_rule_contrib + tail_rule_contrib) / 2)
        all_rule_matches += head_match_rate[0]
        all_rule_matches += tail_match_rate[0]
        total_samples += head_match_rate[1] + tail_match_rate[1]
    
    # 计算最终指标
    hits_1 = np.mean([1.0 if r <= 1 else 0.0 for r in all_ranks])
    hits_3 = np.mean([1.0 if r <= 3 else 0.0 for r in all_ranks])
    hits_5 = np.mean([1.0 if r <= 5 else 0.0 for r in all_ranks])
    hits_10 = np.mean([1.0 if r <= 10 else 0.0 for r in all_ranks])
    mrr = np.mean(all_rec_ranks)
    
    # 计算规则相关指标
    rule_match_rate = all_rule_matches / total_samples if total_samples > 0 else 0
    avg_rule_contrib = np.mean(all_rule_contribs)
    
    # 打印结果
    print("\n" + "="*50)
    print("GraIL-ART 归纳评估结果")
    print("="*50)
    print(f"训练数据集: {args.model_dataset}")
    print(f"测试数据集: {args.dataset}")
    print(f"封闭子图: {args.enclosing_sub_graph}")
    print(f"测试样本数: {len(all_ranks)}")
    print(f"平均规则匹配率: {rule_match_rate:.4f}")
    print(f"平均规则贡献率: {avg_rule_contrib:.4f}")
    print("-"*50)
    print(f"MRR: {mrr:.4f}")
    print(f"Hits@1: {hits_1:.4f}")
    print(f"Hits@3: {hits_3:.4f}")
    print(f"Hits@5: {hits_5:.4f}")
    print(f"Hits@10: {hits_10:.4f}")
    print("="*50)
    
    # 保存结果
    results = {
        'mrr': mrr,
        'hits@1': hits_1,
        'hits@3': hits_3,
        'hits@5': hits_5,
        'hits@10': hits_10,
        'rule_match_rate': rule_match_rate,
        'rule_contrib': avg_rule_contrib
    }
    
    result_path = os.path.join(model_dir, f'inductive_results_{args.dataset}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {result_path}")

def process_batch(links, adj_list, dgl_adj_list, gnn_model, agent_model, rule_matcher, 
                 args, id2entity, node_features, kge_entity2id):
    """
    处理一批链接
    
    返回:
        scores: 所有链接的得分
        rule_contrib: 规则贡献率
        match_rate: (匹配数量, 总样本数)
    """
    max_n_label = np.array([2, 2])  # 默认最大标签值
    subgraphs = []
    r_labels = []
    
    # 为每个链接提取子图
    for link in links:
        head, tail, rel = link[0], link[1], link[2]
        
        try:
            nodes, node_labels = subgraph_extraction_labeling(
                (head, tail), rel, adj_list, h=args.hop,
                enclosing_sub_graph=args.enclosing_sub_graph)
            
            # 确保子图不为空
            if len(nodes) > 0:
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
                
                # 准备KGE特征
                kge_nodes = None
                if node_features is not None and kge_entity2id is not None:
                    try:
                        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes if n in id2entity and id2entity[n] in kge_entity2id]
                        # 只有当所有节点都有KGE嵌入时才使用
                        if len(kge_nodes) == len(nodes):
                            n_feats = node_features[kge_nodes]
                        else:
                            n_feats = None
                    except:
                        n_feats = None
                else:
                    n_feats = None
                
                # 准备特征
                subgraph = prepare_features(subgraph, node_labels, max_n_label, n_feats)
                
                subgraphs.append(subgraph)
                r_labels.append(rel)
        except Exception as e:
            print(f"处理链接 {head}->{rel}->{tail} 时出错: {e}")
            continue
    
    # 如果没有有效的子图，返回零分
    if not subgraphs:
        return np.zeros(len(links)), 0, (0, 0)
    
    # 批量处理子图
    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)
    
    # 移动到适当的设备
    batched_graph = batched_graph.to(agent_model.device)
    r_labels = r_labels.to(agent_model.device)
    
    with torch.no_grad():
        # GNN前向传播
        gnn_scores = gnn_model((batched_graph, r_labels))
        
        # 规则匹配
        rels = r_labels.cpu().numpy()
        rule_ids, rule_masks = rule_matcher.match_rules(
            batched_graph, torch.tensor(rels, device=agent_model.device),
            max_rules=50, apply_dropout=False)
        
        # 规则匹配率统计
        matched_count = sum(torch.any(mask).item() for mask in rule_masks)
        
        # Agent前向传播
        adj_values, _ = agent_model(rule_ids, rule_masks)
        
        # 应用门控机制
        gates = torch.sigmoid(adj_values)
        final_scores = gnn_scores * gates
        
        # 计算规则贡献率
        rule_contrib = (gates - 1.0).abs().mean().item()
    
    # 如果子图数量少于链接数量，补足剩余得分
    scores = final_scores.squeeze(1).cpu().numpy()
    if len(scores) < len(links):
        padding = np.ones(len(links) - len(scores)) * float('-inf')  # 使用负无穷作为填充值
        scores = np.concatenate([scores, padding])
    
    return scores, rule_contrib, (matched_count, len(rule_masks))

if __name__ == "__main__":
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 打印参数
    print("\n" + "="*50)
    print("GraIL-ART 归纳评估")
    print("="*50)
    print(f"训练数据集: {args.model_dataset}")
    print(f"测试数据集: {args.dataset}")
    print(f"实验名称: {args.experiment_name}")
    print(f"规则匹配和ID映射目录: {args.output_dir}")
    print(f"子图跳数: {args.hop}")
    print(f"封闭子图: {args.enclosing_sub_graph}")
    print(f"每链接负样本数: {args.num_neg_samples_per_link}")
    print("="*50 + "\n")
    
    # 运行评估
    evaluate_model(args) 