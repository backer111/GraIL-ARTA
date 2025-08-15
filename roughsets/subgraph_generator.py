import os
import sys
import logging
import argparse
import json
import torch
import numpy as np
import lmdb
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subgraph_extraction.datasets import generate_subgraph_datasets
from utils.data_utils import process_files
from utils.initialization_utils import initialize_experiment


def parse_args():
    parser = argparse.ArgumentParser(description='GraIL-ART: 子图数据集生成')
    
    parser.add_argument("--dataset", "-d", type=str, default="fb237_v1",
                        help="数据集名称 (default: fb237_v1)")
    parser.add_argument("--output_dir", type=str, default="./roughsets/output",
                        help="输出目录")
    parser.add_argument("--hop", type=int, default=2,
                        help="子图提取的跳数")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="每跳最大节点数（通过子采样限制）")
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="设置训练链接的最大数量（以适应内存）")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='约束负采样的概率')
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="每个正链接采样的负例数量")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='是否只考虑包含的子图')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="训练三元组文件名")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="验证三元组文件名")
    parser.add_argument("--test_file", "-ttf", type=str, default="test",
                        help="测试三元组文件名")
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用哪个GPU')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='禁用CUDA')
    
    return parser.parse_args()


def save_mappings(entity2id, relation2id, id2entity, id2relation, output_dir, dataset):
    """保存实体和关系的ID映射"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存entity2id
    entity2id_path = os.path.join(output_dir, f"{dataset}_entity2id.json")
    with open(entity2id_path, 'w') as f:
        json.dump(entity2id, f)
    
    # 保存relation2id
    relation2id_path = os.path.join(output_dir, f"{dataset}_relation2id.json")
    with open(relation2id_path, 'w') as f:
        json.dump(relation2id, f)
    
    # 保存id2entity
    id2entity_path = os.path.join(output_dir, f"{dataset}_id2entity.json")
    with open(id2entity_path, 'w') as f:
        # 将整数键转换为字符串，因为JSON不支持整数键
        id2entity_str = {str(k): v for k, v in id2entity.items()}
        json.dump(id2entity_str, f)
        
    relation2id_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'data/{dataset}')
    os.makedirs(relation2id_data_dir, exist_ok=True)
    relation2id_data_path = os.path.join(relation2id_data_dir, 'relation2id.json')
    with open(relation2id_data_path, 'w') as f:
        json.dump(relation2id, f)
    
    # 保存id2relation
    id2relation_path = os.path.join(output_dir, f"{dataset}_id2relation.json")
    with open(id2relation_path, 'w') as f:
        # 将整数键转换为字符串，因为JSON不支持整数键
        id2relation_str = {str(k): v for k, v in id2relation.items()}
        json.dump(id2relation_str, f)
    
    logging.info(f"实体和关系映射已保存到 {output_dir}")
    logging.info(f"实体数量: {len(entity2id)}, 关系数量: {len(relation2id)}")


def generate_and_save_subgraphs(args):
    """生成子图数据集并保存映射"""
    # 设置文件路径
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.main_dir = main_dir
    
    args.file_paths = {
        'train': os.path.join(main_dir, f'data/{args.dataset}/{args.train_file}.txt'),
        'valid': os.path.join(main_dir, f'data/{args.dataset}/{args.valid_file}.txt')
    }
    
    # 如果指定了测试文件，也添加到文件路径中
    if args.test_file:
        args.file_paths['test'] = os.path.join(main_dir, f'data/{args.dataset}/{args.test_file}.txt')
    
    # 设置设备
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
    
    # 设置子图数据库路径
    args.db_path = os.path.join(main_dir, f'data/{args.dataset}/subgraphs_en_{args.enclosing_sub_graph}_neg_{args.num_neg_samples_per_link}_hop_{args.hop}')
    
    # 处理数据文件，获取映射
    splits = ['train', 'valid']
    if 'test' in args.file_paths:
        splits.append('test')
    
    # 先处理文件获取映射
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(args.file_paths)
    
    # 保存映射
    save_mappings(entity2id, relation2id, id2entity, id2relation, args.output_dir, args.dataset)
    
    # 生成子图数据集
    logging.info(f"开始生成子图数据集，保存到 {args.db_path}")
    generate_subgraph_datasets(args, splits=splits)
    logging.info(f"子图数据集生成完成")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'subgraph_generator.log')),
            logging.StreamHandler()
        ]
    )
    
    # 生成子图数据集并保存映射
    generate_and_save_subgraphs(args)


if __name__ == "__main__":
    main() 