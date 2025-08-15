import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='启动GraIL-ART归纳测试')
    parser.add_argument('--dataset', type=str, default='fb237_v1',
                       help='数据集名称')
    parser.add_argument('--experiment_name', type=str, default='joint_training_fixed',
                       help='训练实验名称')
    parser.add_argument('--model_dir', type=str, default='experiments',
                       help='模型目录')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 找到最新的模型文件
    model_root_dir = os.path.join(args.model_dir, args.experiment_name)
    
    # 构建命令
    cmd = [
        'python', '-m', 'roughsets.test_phase2',
        f'--dataset={args.dataset}',
        f'--experiment_name={args.experiment_name}_eval',
        f'--gpu={args.gpu}',
        f'--batch_size={args.batch_size}',
    ]
    
    # 输出将要执行的命令
    print("=" * 80)
    print("启动GraIL-ART归纳测试评估")
    print("=" * 80)
    print("执行命令:", " ".join(cmd))
    print("=" * 80)
    
    # 执行命令
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
    
    print("=" * 80)
    print("评估完成!")
    print("=" * 80)
    
    # 运行规则贡献率分析
    print("\n分析规则贡献率...\n")
    analyze_cmd = [
        'python', '-m', 'roughsets.analyze_rule_contribution',
        f'--experiment_name={args.experiment_name}',
        f'--log_dir={args.model_dir}',
        '--save_plots'
    ]
    
    try:
        subprocess.run(analyze_cmd, check=True)
    except subprocess.CalledProcessError:
        print("规则贡献率分析失败，请手动运行分析脚本。")

if __name__ == "__main__":
    main() 