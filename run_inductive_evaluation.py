import os
import sys
import argparse
import subprocess
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='启动GraIL-ART归纳关系预测评估')
    
    parser.add_argument('--dataset', type=str, default='fb237_v1_ind',
                       help='归纳测试数据集名称，通常使用_ind后缀版本')
    parser.add_argument('--model_dataset', type=str, default='fb237_v1',
                       help='模型训练时使用的数据集名称')
    parser.add_argument('--experiment_name', type=str, default='fb237_v1_fixed',
                       help='实验名称，用于加载保存的模型')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--hop', type=int, default=2,
                       help='提取子图时跳跃的次数')
    parser.add_argument('--enclosing_sub_graph', '-en', action='store_true',
                       help='是否只考虑封闭子图')
    parser.add_argument('--output_dir', type=str, default='roughsets/output',
                       help='规则和ID映射的输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小')
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 构建命令
    cmd = [
        'python', '-m', 'grail-master.roughsets.evaluate_inductive',
        f'--dataset={args.dataset}',
        f'--model_dataset={args.model_dataset}',
        f'--experiment_name={args.experiment_name}',
        f'--gpu={args.gpu}',
        f'--hop={args.hop}',
        f'--output_dir={args.output_dir}',
        f'--batch_size={args.batch_size}'
    ]
    
    # 添加封闭子图参数（如果指定）
    if args.enclosing_sub_graph:
        cmd.append('--enclosing_sub_graph')
    
    # 输出将要执行的命令
    logging.info("=" * 80)
    logging.info("启动GraIL-ART归纳关系预测评估")
    logging.info("=" * 80)
    logging.info("训练数据集: %s", args.model_dataset)
    logging.info("测试数据集: %s", args.dataset)
    logging.info("实验名称: %s", args.experiment_name)
    logging.info("封闭子图: %s", "是" if args.enclosing_sub_graph else "否")
    logging.info("执行命令: %s", " ".join(cmd))
    logging.info("=" * 80)
    
    # 执行命令
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        logging.info("=" * 80)
        logging.info("归纳评估完成!")
        logging.info("=" * 80)
        
        # 计算结果路径
        result_path = os.path.join('experiments', args.experiment_name, f'inductive_results_{args.dataset}.json')
        if os.path.exists(result_path):
            logging.info("结果已保存到: %s", result_path)
            logging.info("您可以使用以下命令查看结果:")
            logging.info(f"cat {result_path}")
        
    except subprocess.CalledProcessError as e:
        logging.error("评估过程中出现错误: %s", str(e))
    except KeyboardInterrupt:
        logging.info("评估被用户中断")
    except Exception as e:
        logging.error("发生未知错误: %s", str(e))

if __name__ == "__main__":
    main() 