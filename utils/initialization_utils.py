import os
import logging
import json
import torch


def initialize_experiment(params, file_name):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    # 清除已有的日志处理器，避免重复
    logger = logging.getLogger()
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)
    
    # 根据文件类型创建文件处理器
    if file_name == 'test_auc.py':
        params.test_exp_dir = os.path.join(params.exp_dir, f"test_{params.dataset}_{params.constrained_neg_prob}")
        if not os.path.exists(params.test_exp_dir):
            os.makedirs(params.test_exp_dir)
        log_file = os.path.join(params.test_exp_dir, f"log_test.txt")
    else:
        log_file = os.path.join(params.exp_dir, "log_train.txt")
    
    # 确保文件可写
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f'日志文件路径: {log_file}')
    except Exception as e:
        logger.error(f'无法创建日志文件 {log_file}: {str(e)}')
        # 尝试使用备用位置
        try:
            backup_log = os.path.join(os.path.dirname(params.exp_dir), f"log_backup_{params.experiment_name}.txt")
            file_handler = logging.FileHandler(backup_log, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(file_handler)
            logger.info(f'使用备用日志文件: {backup_log}')
        except:
            logger.error('无法创建备用日志文件，仅使用控制台输出')
    
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, model, load_model=False):
    '''
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    '''

    if load_model and os.path.exists(os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
        graph_classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth')).to(device=params.device)
    else:
        relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
        with open(relation2id_path) as f:
            relation2id = json.load(f)

        logging.info('No existing model found. Initializing new model..')
        graph_classifier = model(params, relation2id).to(device=params.device)

    return graph_classifier
