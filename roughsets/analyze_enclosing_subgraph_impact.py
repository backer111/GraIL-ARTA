import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='分析封闭子图选项对规则贡献率的影响')
    parser.add_argument('--dataset', type=str, default='fb237_v1',
                       help='数据集名称')
    parser.add_argument('--ind_dataset', type=str, default='fb237_v1_ind',
                       help='归纳测试数据集名称')
    parser.add_argument('--experiment_name', type=str, default='fb237_v1_fixed',
                       help='实验名称')
    parser.add_argument('--output_dir', type=str, default='roughsets/analysis',
                       help='输出分析文件的目录')
    parser.add_argument('--model_dir', type=str, default='experiments',
                       help='模型目录')
    return parser.parse_args()

def load_results(model_dir, experiment_name, dataset, with_enclosing=False):
    """加载训练或评估结果"""
    # 训练结果路径
    logs_dir = os.path.join(model_dir, experiment_name, 'logs')
    # 归纳评估结果路径
    ind_result_path = os.path.join(model_dir, experiment_name, f'inductive_results_{dataset}{"_enclosing" if with_enclosing else ""}.json')
    
    results = {}
    
    # 尝试加载归纳评估结果
    if os.path.exists(ind_result_path):
        with open(ind_result_path, 'r') as f:
            results['inductive'] = json.load(f)
            results['inductive']['source'] = 'inductive_evaluation'
    
    # 尝试从日志文件中提取训练过程中的规则贡献率
    rule_contribs = []
    rule_matches = []
    epochs = []
    
    # 处理训练日志
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        for log_file in log_files:
            enclosing_marker = 'enclosing' if with_enclosing else 'no_enclosing'
            # 只处理与我们要找的封闭子图设置相符的日志
            if enclosing_marker in log_file:
                log_path = os.path.join(logs_dir, log_file)
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    
                    # 解析日志中的规则贡献率信息
                    for line in log_content.split('\n'):
                        if '规则贡献率' in line:
                            try:
                                # 提取数值
                                parts = line.split('规则贡献率:')
                                if len(parts) > 1:
                                    value = float(parts[1].strip())
                                    rule_contribs.append(value)
                            except:
                                pass
                        elif '规则匹配率' in line:
                            try:
                                # 提取数值
                                parts = line.split('规则匹配率:')
                                if len(parts) > 1:
                                    value = float(parts[1].strip())
                                    rule_matches.append(value)
                            except:
                                pass
                        elif 'Epoch ' in line:
                            try:
                                # 提取轮次
                                parts = line.split('Epoch ')
                                if len(parts) > 1:
                                    epoch = int(parts[1].split('/')[0].strip())
                                    if epoch not in epochs:
                                        epochs.append(epoch)
                            except:
                                pass
    
    if rule_contribs:
        results['training'] = {
            'rule_contribs': rule_contribs,
            'rule_matches': rule_matches,
            'epochs': epochs,
            'source': 'training_logs'
        }
    
    return results

def analyze_results(with_enclosing_results, without_enclosing_results, output_dir, args):
    """分析比较两种设置的结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并结果
    combined_results = {
        'with_enclosing': with_enclosing_results,
        'without_enclosing': without_enclosing_results
    }
    
    # 写入分析结果
    output_file = os.path.join(output_dir, f'{args.dataset}_enclosing_impact_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # 生成分析报告
    report_file = os.path.join(output_dir, f'{args.dataset}_enclosing_impact_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GraIL-ART 封闭子图影响分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("数据集: " + args.dataset + "\n")
        f.write("归纳测试数据集: " + args.ind_dataset + "\n")
        f.write("实验名称: " + args.experiment_name + "\n\n")
        
        # 分析归纳评估结果
        f.write("-" * 40 + "\n")
        f.write("归纳评估结果比较\n")
        f.write("-" * 40 + "\n")
        
        if 'inductive' in with_enclosing_results and 'inductive' in without_enclosing_results:
            encl_res = with_enclosing_results['inductive']
            no_encl_res = without_enclosing_results['inductive']
            
            f.write(f"指标\t\t有封闭子图\t无封闭子图\t差异\n")
            for metric in ['mrr', 'hits@1', 'hits@3', 'hits@5', 'hits@10']:
                if metric in encl_res and metric in no_encl_res:
                    diff = encl_res[metric] - no_encl_res[metric]
                    f.write(f"{metric}\t\t{encl_res[metric]:.4f}\t{no_encl_res[metric]:.4f}\t{diff:+.4f}\n")
            
            # 规则相关指标
            for metric in ['rule_match_rate', 'rule_contrib']:
                if metric in encl_res and metric in no_encl_res:
                    diff = encl_res[metric] - no_encl_res[metric]
                    f.write(f"{metric}\t{encl_res[metric]:.4f}\t{no_encl_res[metric]:.4f}\t{diff:+.4f}\n")
        else:
            f.write("缺少归纳评估结果，无法比较\n")
        
        f.write("\n")
        
        # 分析训练过程
        f.write("-" * 40 + "\n")
        f.write("训练过程比较\n")
        f.write("-" * 40 + "\n")
        
        if 'training' in with_enclosing_results and 'training' in without_enclosing_results:
            encl_train = with_enclosing_results['training']
            no_encl_train = without_enclosing_results['training']
            
            # 规则贡献率
            f.write("规则贡献率比较：\n")
            if encl_train['rule_contribs'] and no_encl_train['rule_contribs']:
                encl_avg = np.mean(encl_train['rule_contribs'])
                no_encl_avg = np.mean(no_encl_train['rule_contribs'])
                diff = encl_avg - no_encl_avg
                f.write(f"平均规则贡献率: 有封闭子图 {encl_avg:.4f} vs 无封闭子图 {no_encl_avg:.4f}, 差异: {diff:+.4f}\n")
            
                if len(encl_train['rule_contribs']) >= 3 and len(no_encl_train['rule_contribs']) >= 3:
                    early_diff = np.mean(encl_train['rule_contribs'][:3]) - np.mean(no_encl_train['rule_contribs'][:3])
                    late_diff = np.mean(encl_train['rule_contribs'][-3:]) - np.mean(no_encl_train['rule_contribs'][-3:])
                    f.write(f"训练早期差异: {early_diff:+.4f}, 训练后期差异: {late_diff:+.4f}\n")
            else:
                f.write("缺少规则贡献率数据\n")
            
            # 规则匹配率
            f.write("\n规则匹配率比较：\n")
            if encl_train['rule_matches'] and no_encl_train['rule_matches']:
                encl_avg = np.mean(encl_train['rule_matches'])
                no_encl_avg = np.mean(no_encl_train['rule_matches'])
                diff = encl_avg - no_encl_avg
                f.write(f"平均规则匹配率: 有封闭子图 {encl_avg:.4f} vs 无封闭子图 {no_encl_avg:.4f}, 差异: {diff:+.4f}\n")
            else:
                f.write("缺少规则匹配率数据\n")
        else:
            f.write("缺少训练过程数据，无法比较\n")
        
        # 总结
        f.write("\n" + "-" * 40 + "\n")
        f.write("分析总结\n")
        f.write("-" * 40 + "\n")
        
        # 汇总观察结果
        has_training_data = 'training' in with_enclosing_results and 'training' in without_enclosing_results
        has_inductive_data = 'inductive' in with_enclosing_results and 'inductive' in without_enclosing_results
        
        if has_training_data:
            encl_train = with_enclosing_results['training']
            no_encl_train = without_enclosing_results['training']
            if encl_train['rule_contribs'] and no_encl_train['rule_contribs']:
                encl_avg = np.mean(encl_train['rule_contribs'])
                no_encl_avg = np.mean(no_encl_train['rule_contribs'])
                
                if encl_avg < no_encl_avg:
                    f.write("1. 训练过程中，使用封闭子图导致规则贡献率降低，可能是因为封闭子图限制了可用的结构信息。\n")
                else:
                    f.write("1. 训练过程中，使用封闭子图并未导致规则贡献率显著降低。\n")
        
        if has_inductive_data:
            encl_res = with_enclosing_results['inductive']
            no_encl_res = without_enclosing_results['inductive']
            
            if 'rule_contrib' in encl_res and 'rule_contrib' in no_encl_res:
                if encl_res['rule_contrib'] < no_encl_res['rule_contrib']:
                    f.write("2. 在归纳评估中，使用封闭子图降低了规则贡献率。\n")
                else:
                    f.write("2. 在归纳评估中，使用封闭子图并未降低规则贡献率。\n")
            
            # 比较性能指标
            if 'mrr' in encl_res and 'mrr' in no_encl_res:
                if encl_res['mrr'] > no_encl_res['mrr']:
                    f.write("3. 尽管可能降低规则贡献率，封闭子图在归纳设置下仍然提高了模型的整体性能（MRR）。\n")
                else:
                    f.write("3. 封闭子图在归纳设置下降低了模型的整体性能（MRR）。\n")
        
        # 最终建议
        f.write("\n**结论与建议**:\n")
        
        if has_inductive_data and 'mrr' in with_enclosing_results.get('inductive', {}) and 'mrr' in without_enclosing_results.get('inductive', {}):
            encl_mrr = with_enclosing_results['inductive']['mrr']
            no_encl_mrr = without_enclosing_results['inductive']['mrr']
            
            if encl_mrr > no_encl_mrr:
                f.write("- 对于归纳关系预测任务，建议使用封闭子图，因为它提高了模型性能，即使可能减少规则贡献。\n")
            else:
                f.write("- 对于归纳关系预测任务，建议不使用封闭子图，这样可以保持更高的规则贡献率且不损失性能。\n")
        elif has_training_data:
            encl_train = with_enclosing_results.get('training', {})
            no_encl_train = without_enclosing_results.get('training', {})
            
            if encl_train.get('rule_contribs') and no_encl_train.get('rule_contribs'):
                encl_avg = np.mean(encl_train['rule_contribs'])
                no_encl_avg = np.mean(no_encl_train['rule_contribs'])
                
                if encl_avg < no_encl_avg:
                    f.write("- 基于训练数据，建议不使用封闭子图以保持更高的规则贡献率。\n")
                else:
                    f.write("- 基于训练数据，封闭子图选项对规则贡献率影响不大，可根据具体任务需求选择。\n")
        else:
            f.write("- 数据不足以提供明确建议，建议进行更多对比实验。\n")

    # 生成图表(如果有训练过程数据)
    if 'training' in with_enclosing_results and 'training' in without_enclosing_results:
        encl_train = with_enclosing_results['training']
        no_encl_train = without_enclosing_results['training']
        
        if encl_train['rule_contribs'] and no_encl_train['rule_contribs']:
            plt.figure(figsize=(10, 6))
            
            # 如果有epoch信息，使用它作为x轴
            if encl_train['epochs'] and len(encl_train['epochs']) == len(encl_train['rule_contribs']):
                plt.plot(encl_train['epochs'], encl_train['rule_contribs'], 'b-', label='有封闭子图')
            else:
                plt.plot(encl_train['rule_contribs'], 'b-', label='有封闭子图')
                
            if no_encl_train['epochs'] and len(no_encl_train['epochs']) == len(no_encl_train['rule_contribs']):
                plt.plot(no_encl_train['epochs'], no_encl_train['rule_contribs'], 'r-', label='无封闭子图')
            else:
                plt.plot(no_encl_train['rule_contribs'], 'r-', label='无封闭子图')
            
            plt.title('封闭子图对规则贡献率的影响')
            plt.xlabel('训练轮次')
            plt.ylabel('规则贡献率')
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            chart_file = os.path.join(output_dir, f'{args.dataset}_rule_contrib_comparison.png')
            plt.savefig(chart_file)
            plt.close()
            
            print(f"生成图表: {chart_file}")
    
    print(f"分析报告已保存至: {report_file}")
    print(f"分析数据已保存至: {output_file}")
    
    return combined_results

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"正在分析封闭子图对规则贡献率的影响...")
    
    # 加载带封闭子图的结果
    print("加载带封闭子图的结果...")
    with_enclosing_results = load_results(args.model_dir, f"{args.experiment_name}_enclosing", args.ind_dataset, True)
    
    # 加载不带封闭子图的结果
    print("加载不带封闭子图的结果...")
    without_enclosing_results = load_results(args.model_dir, args.experiment_name, args.ind_dataset, False)
    
    # 分析结果
    analyze_results(with_enclosing_results, without_enclosing_results, args.output_dir, args)
    
    print("分析完成!")

if __name__ == "__main__":
    main() 