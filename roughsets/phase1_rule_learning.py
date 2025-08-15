import os
import argparse
import logging
import torch
import json
from rule_miner import RoughSetRuleMiner
from rule_encoder import RuleEncoder
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='GraIL-ART Phase 1: Rule Learning & Embedding')
    
    parser.add_argument('--dataset', type=str, default='fb237_v1',
                        help='Dataset name (default: fb237_v1)')
    parser.add_argument('--output_dir', type=str, default='./roughsets/output',
                        help='Output directory for rules and embeddings')
    parser.add_argument('--confidence_threshold', type=float, default=0.4,
                        help='Confidence threshold for rules (default: 0.7)')
    parser.add_argument('--support_threshold', type=int, default=4,
                        help='Support threshold for rules (default: 5)')
    parser.add_argument('--max_path_length', type=int, default=3,
                        help='Maximum path length for path patterns (default: 3)')
    parser.add_argument('--max_rules_per_relation', type=int, default=100,
                        help='Maximum number of rules per relation (default: 100)')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Rule embedding dimension (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for embedding training (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs for embedding training (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for embedding training (default: 64)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU even if GPU is available')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志记录器
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'phase1.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting Phase 1 for dataset: {args.dataset}")
    logging.info(f"Arguments: {args}")
    
    # 设置数据文件路径
    train_file = f"data/{args.dataset}/train.txt"  
    if not os.path.exists(train_file):
        logging.error(f"Train file not found: {train_file}")
        return
    
    # 设置输出文件路径
    ruleset_file = os.path.join(args.output_dir, f"{args.dataset}_ruleset.json")
    embedding_file = os.path.join(args.output_dir, f"{args.dataset}_rule_embeddings.json")
    
    # 配置规则挖掘器
    rule_miner_config = {
        'confidence_threshold': args.confidence_threshold,
        'support_threshold': args.support_threshold,
        'max_path_length': args.max_path_length,
        'max_rules_per_relation': args.max_rules_per_relation
    }
    
    
    
    # 配置规则编码器
    encoder_config = {
        'embedding_dim': args.embedding_dim,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'device': 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    try:
        # 步骤1：规则挖掘
        logging.info("Step 1: Mining rules from knowledge graph")
        miner = RoughSetRuleMiner(rule_miner_config)
        ruleset = miner.mine_rules(train_file)
        miner.analyze_rule_conflicts()  # 添加规则冲突信息
        miner.save_ruleset(ruleset_file)
        
        # 打印规则统计信息
        rule_types = {}
        for rule in ruleset:
            rule_type = rule.get('type', 'unknown')
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        
        logging.info(f"Rule statistics by type: {rule_types}")
        
        # 步骤2：嵌入学习
        logging.info(f"Step 2: Learning rule embeddings (device: {encoder_config['device']})")
        encoder = RuleEncoder(encoder_config)
        encoder.encode_ruleset(ruleset_file, embedding_file)
        
        # 创建规则和嵌入的摘要统计
        with open(embedding_file, 'r') as f:
            embed_data = json.load(f)
        
        logging.info(f"Generated {len(embed_data['rule_embeds'])} rule embeddings")
        logging.info(f"Generated {len(embed_data['relation_embeds'])} relation embeddings")
        logging.info(f"Special token: NO_RULE_IDX = {embed_data['no_rule_idx']}")
        
        logging.info("Phase 1 completed successfully")
        
    except Exception as e:
        logging.error(f"Error in Phase 1: {str(e)}", exc_info=True)
        return


if __name__ == "__main__":
    main() 