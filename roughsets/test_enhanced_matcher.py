import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

# 导入规则匹配器
from roughsets.rule_matcher_enhanced import EnhancedRuleMatcher

def parse_args():
    parser = argparse.ArgumentParser(description="测试增强版规则匹配器")
    
    # 数据集与规则文件参数
    parser.add_argument('--dataset', type=str, default='fb237_v1', help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='roughsets/output', help='输出目录')
    parser.add_argument('--rule_file', type=str, default=None, help='规则文件路径')
    parser.add_argument('--relation2id_file', type=str, default=None, help='关系ID映射文件路径')
    
    # 设备参数
    parser.add_argument('--disable_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 设置路径
    main_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果规则文件和关系映射文件没有指定，使用默认路径
    if args.rule_file is None:
        args.rule_file = os.path.join(args.output_dir, f"{args.dataset}_ruleset.json")
    
    if args.relation2id_file is None:
        args.relation2id_file = os.path.join(args.output_dir, f"{args.dataset}_relation2id.json")
    
    # 检查文件存在
    if not os.path.exists(args.rule_file):
        logging.error(f"规则文件不存在: {args.rule_file}")
        return
        
    if not os.path.exists(args.relation2id_file):
        logging.error(f"关系ID映射文件不存在: {args.relation2id_file}")
        return
    
    # 设置设备
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    logging.info(f"使用设备: {device}")
    
    # 加载规则集
    with open(args.rule_file, 'r') as f:
        ruleset = json.load(f)
    
    logging.info(f"加载了 {len(ruleset)} 条规则")
    
    # 加载关系ID映射
    with open(args.relation2id_file, 'r') as f:
        relation2id = json.load(f)
    
    id2relation = {int(v): k for k, v in relation2id.items()}
    logging.info(f"加载了 {len(relation2id)} 种关系的ID映射")
    
    # 创建常规规则匹配器（不使用映射）
    no_rule_idx = len(ruleset)
    logging.info("\n===== 使用常规规则匹配器（不使用ID映射） =====")
    vanilla_matcher = EnhancedRuleMatcher(ruleset, device, no_rule_idx)
    
    # 创建增强版规则匹配器（使用ID映射）
    logging.info("\n===== 使用增强版规则匹配器（使用ID映射） =====")
    enhanced_matcher = EnhancedRuleMatcher(ruleset, device, no_rule_idx, relation2id, id2relation)
    
    # 测试一批模拟的关系ID
    test_size = 100
    logging.info(f"\n执行匹配测试，生成 {test_size} 个随机关系ID...")
    
    # 生成随机关系ID
    import random
    max_relation_id = max(relation2id.values())
    relation_ids = [random.randint(0, max_relation_id) for _ in range(test_size)]
    
    # 创建模拟输入数据
    relation_tensor = torch.tensor(relation_ids, device=device)
    dummy_subgraphs = [None] * test_size
    
    # 使用常规匹配器匹配规则
    logging.info("使用常规匹配器匹配...")
    vanilla_rule_ids, vanilla_rule_masks = vanilla_matcher.match_rules(
        dummy_subgraphs, relation_tensor, max_rules=10
    )
    
    # 使用增强版匹配器匹配规则
    logging.info("使用增强版匹配器匹配...")
    enhanced_rule_ids, enhanced_rule_masks = enhanced_matcher.match_rules(
        dummy_subgraphs, relation_tensor, max_rules=10
    )
    
    # 统计结果
    vanilla_matched = sum([1 for mask in vanilla_rule_masks if mask.any()])
    enhanced_matched = sum([1 for mask in enhanced_rule_masks if mask.any()])
    
    print("\n=== 匹配结果比较 ===")
    print(f"总样本数: {test_size}")
    print(f"常规匹配器匹配数量: {vanilla_matched} ({vanilla_matched/test_size*100:.2f}%)")
    print(f"增强版匹配器匹配数量: {enhanced_matched} ({enhanced_matched/test_size*100:.2f}%)")
    print(f"改进数量: {enhanced_matched - vanilla_matched} ({(enhanced_matched - vanilla_matched)/test_size*100:.2f}%)")
    
    # 分析匹配情况
    relation_stats = defaultdict(lambda: {'vanilla': False, 'enhanced': False})
    
    for i, rel_id in enumerate(relation_ids):
        relation_stats[rel_id]['vanilla'] = vanilla_rule_masks[i].any().item()
        relation_stats[rel_id]['enhanced'] = enhanced_rule_masks[i].any().item()
    
    # 打印部分示例
    print("\n=== 部分关系的匹配情况示例 ===")
    print("关系ID\t关系路径\t\t普通匹配器\t增强匹配器")
    print("-" * 80)
    
    # 选择一些有差异的关系进行展示
    improved_relations = [rel_id for rel_id, stats in relation_stats.items() 
                        if stats['enhanced'] and not stats['vanilla']]
    
    # 如果有改进的关系，展示前5个
    for rel_id in improved_relations[:5]:
        rel_path = id2relation.get(rel_id, "未知")
        print(f"{rel_id}\t{rel_path[:30]}\t{'✓' if relation_stats[rel_id]['vanilla'] else '×'}\t\t{'✓' if relation_stats[rel_id]['enhanced'] else '×'}")

if __name__ == "__main__":
    main()