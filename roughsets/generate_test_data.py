import os
import json
import random
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="生成测试规则集和关系ID映射")
    
    parser.add_argument('--dataset', type=str, default='fb237_v1', help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='roughsets/output', help='输出目录')
    parser.add_argument('--num_rules', type=int, default=100, help='生成的规则数量')
    parser.add_argument('--num_relations', type=int, default=30, help='关系数量')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成测试关系
    relations = []
    # 一些示例关系路径
    fb_relations = [
        "/film/film/genre",
        "/people/person/nationality",
        "/people/person/profession",
        "/location/location/contains",
        "/film/director/film",
        "/organization/organization/founders",
        "/people/person/place_of_birth",
        "/business/company/industry",
        "/people/person/education",
        "/book/author/works_written"
    ]
    
    # 补充更多关系路径
    for i in range(max(0, args.num_relations - len(fb_relations))):
        relations.append(f"/domain_{i}/type_{i}/relation_{i}")
    
    # 添加预设的关系路径
    relations.extend(fb_relations[:min(len(fb_relations), args.num_relations)])
    
    # 确保不超过要求的关系数量
    relations = relations[:args.num_relations]
    
    # 创建关系ID映射
    relation2id = {rel: i for i, rel in enumerate(relations)}
    id2relation = {i: rel for i, rel in enumerate(relations)}
    
    # 生成测试规则集
    ruleset = []
    rule_types = ["path", "type_constraint"]
    
    for i in range(args.num_rules):
        rule_type = random.choice(rule_types)
        relation = random.choice(relations)
        
        if rule_type == "path":
            # 生成路径规则
            path_length = random.randint(1, 3)
            path = [random.choice(relations) for _ in range(path_length)]
            
            rule = {
                "id": i,
                "type": "path",
                "relation": relation,
                "path": path,
                "confidence": random.uniform(0.6, 1.0),
                "support": random.randint(5, 100)
            }
        else:
            # 生成类型约束规则
            rule = {
                "id": i,
                "type": "type_constraint",
                "relation": relation,
                "head_type": f"/type/entity_{random.randint(1, 20)}",
                "tail_type": f"/type/entity_{random.randint(1, 20)}",
                "confidence": random.uniform(0.6, 1.0),
                "support": random.randint(5, 100)
            }
        
        ruleset.append(rule)
    
    # 保存规则集
    ruleset_file = os.path.join(args.output_dir, f"{args.dataset}_ruleset.json")
    with open(ruleset_file, 'w') as f:
        json.dump(ruleset, f, indent=2)
    
    # 保存关系ID映射
    relation2id_file = os.path.join(args.output_dir, f"{args.dataset}_relation2id.json")
    with open(relation2id_file, 'w') as f:
        json.dump(relation2id, f, indent=2)
    
    id2relation_file = os.path.join(args.output_dir, f"{args.dataset}_id2relation.json")
    with open(id2relation_file, 'w') as f:
        json.dump({str(k): v for k, v in id2relation.items()}, f, indent=2)
    
    entity2id_file = os.path.join(args.output_dir, f"{args.dataset}_entity2id.json")
    with open(entity2id_file, 'w') as f:
        # 简单的实体ID映射
        entity2id = {f"entity_{i}": i for i in range(100)}
        json.dump(entity2id, f, indent=2)
    
    id2entity_file = os.path.join(args.output_dir, f"{args.dataset}_id2entity.json")
    with open(id2entity_file, 'w') as f:
        # 简单的ID到实体映射
        id2entity = {str(i): f"entity_{i}" for i in range(100)}
        json.dump(id2entity, f, indent=2)
    
    # 生成规则嵌入
    rule_embeds_file = os.path.join(args.output_dir, f"{args.dataset}_rule_embeddings.json")
    
    embed_dim = 64
    rule_embeds = [[random.uniform(-0.5, 0.5) for _ in range(embed_dim)] for _ in range(args.num_rules + 1)]
    relation_embeds = [[random.uniform(-0.5, 0.5) for _ in range(embed_dim)] for _ in range(args.num_relations)]
    
    embed_data = {
        "rule_embeds": rule_embeds,
        "relation_embeds": relation_embeds,
        "no_rule_idx": args.num_rules
    }
    
    with open(rule_embeds_file, 'w') as f:
        json.dump(embed_data, f)
    
    print(f"已生成测试数据:")
    print(f"- 规则集: {ruleset_file} ({len(ruleset)} 条规则)")
    print(f"- 关系ID映射: {relation2id_file} ({len(relation2id)} 种关系)")
    print(f"- ID到关系映射: {id2relation_file}")
    print(f"- 实体ID映射: {entity2id_file}")
    print(f"- ID到实体映射: {id2entity_file}")
    print(f"- 规则嵌入: {rule_embeds_file}")

if __name__ == "__main__":
    main() 