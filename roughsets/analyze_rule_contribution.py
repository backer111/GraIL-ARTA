import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
from collections import defaultdict
from sklearn import metrics

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入规则匹配器
from roughsets.rule_matcher_enhanced import EnhancedRuleMatcher

def parse_args():
    parser = argparse.ArgumentParser(description="分析规则贡献率")
    
    # 数据集与规则文件参数
    parser.add_argument('--dataset', type=str, default='fb237_v1', help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='roughsets/output', help='输出目录')
    parser.add_argument('--rule_file', type=str, default=None, help='规则文件路径')
    parser.add_argument('--relation2id_file', type=str, default=None, help='关系ID映射文件路径')
    parser.add_argument('--model_dir', type=str, default=None, help='模型目录路径')
    
    # 设备参数
    parser.add_argument('--disable_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    
    return parser.parse_args()

def test_rule_matcher(args, device):
    """测试规则匹配器的匹配效果"""
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
    
    # 测试实际的关系ID
    unique_relations = list(relation2id.values())
    test_size = len(unique_relations)
    logging.info(f"\n执行匹配测试，使用 {test_size} 个实际关系ID...")
    
    # 创建模拟输入数据
    relation_tensor = torch.tensor(unique_relations, device=device)
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
    print(f"总关系数量: {test_size}")
    print(f"常规匹配器匹配关系数量: {vanilla_matched} ({vanilla_matched/test_size*100:.2f}%)")
    print(f"增强版匹配器匹配关系数量: {enhanced_matched} ({enhanced_matched/test_size*100:.2f}%)")
    print(f"改进关系数量: {enhanced_matched - vanilla_matched} ({(enhanced_matched - vanilla_matched)/test_size*100:.2f}%)")
    
    # 分析匹配情况
    relation_stats = {}
    
    for i, rel_id in enumerate(unique_relations):
        rel_path = id2relation.get(rel_id, f"ID={rel_id}")
        relation_stats[rel_id] = {
            'path': rel_path,
            'vanilla': vanilla_rule_masks[i].any().item(),
            'enhanced': enhanced_rule_masks[i].any().item()
        }
    
    # 打印部分示例
    print("\n=== 部分关系的匹配情况示例 ===")
    print("关系ID\t关系路径\t\t普通匹配器\t增强匹配器")
    print("-" * 80)
    
    # 选择一些有差异的关系进行展示
    improved_relations = [rel_id for rel_id, stats in relation_stats.items() 
                        if stats['enhanced'] and not stats['vanilla']]
    
    # 如果有改进的关系，展示前10个
    for rel_id in improved_relations[:10]:
        rel_path = relation_stats[rel_id]['path']
        print(f"{rel_id}\t{rel_path[:30]}\t{'✓' if relation_stats[rel_id]['vanilla'] else '×'}\t\t{'✓' if relation_stats[rel_id]['enhanced'] else '×'}")
    
    return relation_stats

def analyze_model_outputs(args, device):
    """分析模型输出中规则的贡献情况"""
    if args.model_dir is None:
        logging.warning("未指定模型目录，跳过模型输出分析")
        return
    
    if not os.path.exists(args.model_dir):
        logging.error(f"模型目录不存在: {args.model_dir}")
        return
    
    # 检查最终模型文件
    agent_model_path = os.path.join(args.model_dir, 'final_agent_model.pth')
    gnn_model_path = os.path.join(args.model_dir, 'final_gnn_model.pth')
    
    if not os.path.exists(agent_model_path) or not os.path.exists(gnn_model_path):
        logging.error(f"模型文件不完整，agent_model: {os.path.exists(agent_model_path)}, gnn_model: {os.path.exists(gnn_model_path)}")
        return
    
    # 加载模型
    logging.info("加载训练好的模型...")
    try:
        agent_model = torch.load(agent_model_path, map_location=device)
        gnn_model = torch.load(gnn_model_path, map_location=device)
        logging.info("模型加载成功")
    except Exception as e:
        logging.error(f"加载模型出错: {str(e)}")
        return
    
    # 尝试提取辅助损失历史（如果有）
    history_file = os.path.join(args.model_dir, 'training_history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if 'aux_losses' in history:
                aux_losses = history['aux_losses']
                logging.info(f"辅助损失历史: {aux_losses}")
                
                print("\n=== 辅助损失趋势 ===")
                print(f"初始辅助损失: {aux_losses[0]:.6f}")
                print(f"最终辅助损失: {aux_losses[-1]:.6f}")
                print(f"辅助损失变化: {aux_losses[-1] - aux_losses[0]:.6f}")
        except Exception as e:
            logging.warning(f"读取训练历史出错: {str(e)}")
    
    # 分析Agent模型参数
    print("\n=== Agent模型参数分析 ===")
    total_params = sum(p.numel() for p in agent_model.parameters())
    print(f"总参数数量: {total_params}")
    
    # 尝试分析规则调整幅度
    if hasattr(agent_model, 'rule_embeds') and agent_model.rule_embeds is not None:
        num_rules = agent_model.rule_embeds.shape[0]
        print(f"规则数量: {num_rules}")
        
        # 可能的决策参数
        if hasattr(agent_model, 'output_layer'):
            try:
                weights = agent_model.output_layer.weight.data
                bias = agent_model.output_layer.bias.data if agent_model.output_layer.bias is not None else 0.0
                
                print(f"输出层权重平均值: {weights.mean().item():.6f}")
                print(f"输出层权重标准差: {weights.std().item():.6f}")
                print(f"输出层偏置项: {bias.item() if isinstance(bias, torch.Tensor) else bias:.6f}")
            except:
                pass

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
    
    # 设置设备
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    logging.info(f"使用设备: {device}")
    
    # 测试规则匹配器
    print("\n" + "="*50)
    print(" 规则匹配分析")
    print("="*50)
    relation_stats = test_rule_matcher(args, device)
    
    # 分析模型输出
    print("\n" + "="*50)
    print(" 模型输出分析")
    print("="*50)
    analyze_model_outputs(args, device)

if __name__ == "__main__":
    main() 