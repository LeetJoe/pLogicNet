#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model_test import KGEModel

from dataloader_test import TrainDataset
from dataloader_test import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    # 若指定，则为 true
    parser.add_argument('--cuda', action='store_true', help='use GPU')

    # 在 train 阶段实际上 train/valid/test 都做
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    # 这个似乎没有在任何地方使用
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    # 这三个不是主要数据集
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    # 这个在运行时设定，不能在命令行里指定
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--workspace_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)

    # 这两个是额外增加 embedding 的维数的意思
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    # 反向取样就是故意找一些不符合目标的样本作为排斥样例
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    # 隐状态维度
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    # 一个系数 todo 了解下具体作用的位置
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    # todo 负对抗采样？
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    # 对抗温度
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    # batch size
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    # todo 规格化？
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    # valid/test batch size
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    # todo 子采样权重？貌似是在 word2vec 里的东西
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')

    # lr
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)

    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    # 这是一个路径，如果其中有 config.json 文件，则会尝试从中读取配置
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    #
    parser.add_argument('-save', '--save_path', default=None, type=str)
    # todo 什么的 step ？
    parser.add_argument('--max_steps', default=100000, type=int)
    # todo warm up?
    parser.add_argument('--warm_up_steps', default=None, type=int)

    #
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    #
    parser.add_argument('--valid_steps', default=10000, type=int)
    #
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    #
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    # number of entities，这两个参数在运行时设定，应该只是为了记录
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    # number of relations
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    #
    parser.add_argument('--record', action='store_true')
    # top k
    parser.add_argument('--topk', default=100, type=int)
    
    return parser.parse_args(args)


# 如果指定的 checkpoint 目录中有配置信息，则使用其中的配置覆盖现有的配置
def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    # 感觉这里应该缩进一层，先不动它
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


# 保存模型，分三个文件，一个包含模型和优化器的参数，一个是实体嵌入，一个是关系嵌入
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    # 这里保存的是模型状态（参数）和优化器状态（参数）
    torch.save({
        **save_variable_list,   # 这里 ** 的意思应该跟函数参数里的 ** 类似，表示不定数量的任意展开，这里就是把字典内容展开放在前面。
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    # 实体嵌入分离出来，然后放在 cpu 里转化成 numpy 格式？
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )

    # 关系嵌入部分
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )


# 这个是根据实体和关系与 id 的映射，将文件中的三元组转化成 id 形式的三元组
def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


# 使用 logging 模块设置的一些 log 格式
def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# 格式化输出一组数据
def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


#
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def main(args):
    # 一些参数间的组合约束
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)

    # 实体 - id 关系的双向映射
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    # 关系 - id 之间的双向映射
    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    # Read regions for Countries S* datasets 这部分数据没给
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    # --------------------------------------------------
    # Comments by Meng:
    # During training, pLogicNet will augment the training triplets,
    # so here we load both the augmented triplets (train.txt) for training and
    # the original triplets (train_kge.txt) for evaluation.
    # Also, the hidden triplets (hidden.txt) are also loaded for annotation.
    # --------------------------------------------------

    # train_kge 是从 train_augmented 复制过来的。这里把所有数据以 id 的形式加载
    train_triples = read_triple(os.path.join(args.workspace_path, 'train_kge.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    train_original_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train original: %d' % len(train_original_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    # hidden 是潜在的未经验证计算的关系
    hidden_triples = read_triple(os.path.join(args.workspace_path, 'hidden.txt'), entity2id, relation2id)
    logging.info('#hidden: %d' % len(hidden_triples))
    
    #All true triples， true 是指实际存在的，hidden 不算。
    all_true_triples = train_original_triples + valid_triples + test_triples

    # 实例化 model todo 需要进一步深入 KGEModel 的定义
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    logging.info('Model Parameter Configuration:')
    # todo: 这个是在记录模型的规模？记录所有参数名与参数尺寸
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # data loader 分为 head 和 tail，是基于同一个数据集(train_triples)进行采样生成

        # Set training dataloader iterator todo 深入了解一下 Dataloader 的实现
        train_dataloader_head = DataLoader(
            # todo 注意这里 Dataset 的实现
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        # todo 还有这里的实现
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam( # adaptive moment optimizer
            filter(lambda p: p.requires_grad, kge_model.parameters()),   # todo
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    # 到这里 kge_model 和 optimizer 都刚初始化，如果有 checkpoint, 则从 checkpoint 加载
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        # todo 这个 step 为什么还要从 checkpoint 里取？
        init_step = checkpoint['step']
        # checkpoint 里的状态词典
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train: # 下面三个只有在训练模式下才覆盖
            current_learning_rate = checkpoint['current_learning_rate']  # lr
            warm_up_steps = checkpoint['warm_up_steps']  # 热身步数
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 优化器状态
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # 参数全部保存在 opt.txt 文件里
    if args.record:
        local_path = args.workspace_path
        ensure_dir(local_path)

        opt = vars(args)
        with open(local_path + '/opt.txt', 'w') as fo:
            for key, val in opt.items():
                fo.write('{} {}\n'.format(key, val))

    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        training_logs = []

        #Training Loop todo 这里的 max_steps 肯定跟之前对应着 kge.sh 里找出来的不一样，需要再审查一下
        # todo 这里的 steps 跟之前的代码里的 epochs 是不是同一个概念？ max_steps 是一个很大的数, 如150000
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                # 如果超过了热身步骤，就每个步骤里都把 lr /= 10 todo 第一次见到动态 lr
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                # 这里是直接重新定义 optimizer 了 todo 不能单改 lr 吗?
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3      # todo 直接乘 3 ？默认 warm_up_steps 取 max_steps // 2，这里乘以 3 ？

            # 每 x 步保存一下 model 等数据
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            # 定期清空 training_logs 并将记录内容求均值记录下来
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            # 每隔一定 steps 进行一次 valid
            if args.do_valid and (step + 1) % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics, preds = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)

        # 所有 steps 完成后，保存模型和相关状态
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    # 专门的 valid 步骤
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics, preds = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
        
        # --------------------------------------------------
        # Comments by Meng:
        # Save the prediction results of KGE on validation set.
        # --------------------------------------------------

        if args.record:   # 保存
            # Save the final results
            with open(local_path + '/result_kge_valid.txt', 'w') as fo:
                for metric in metrics:
                    fo.write('{} : {}\n'.format(metric, metrics[metric]))

            # Save the predictions on test data
            with open(local_path + '/pred_kge_valid.txt', 'w') as fo:
                for h, r, t, f, rk, l in preds:
                    fo.write('{}\t{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], f, rk))
                    for e, val in l:
                        fo.write('{}:{:.4f} '.format(id2entity[e], val))
                    fo.write('\n')

    # 专门的 test 步骤
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, preds = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
        # --------------------------------------------------
        # Comments by Meng:
        # Save the prediction results of KGE on test set.
        # --------------------------------------------------

        if args.record:   # 保存
            # Save the final results
            with open(local_path + '/result_kge.txt', 'w') as fo:
                for metric in metrics:
                    fo.write('{} : {}\n'.format(metric, metrics[metric]))

            # Save the predictions on test data
            with open(local_path + '/pred_kge.txt', 'w') as fo:
                for h, r, t, f, rk, l in preds:
                    fo.write('{}\t{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], f, rk))
                    for e, val in l:
                        fo.write('{}:{:.4f} '.format(id2entity[e], val))
                    fo.write('\n')

    # --------------------------------------------------
    # Comments by Meng:
    # Save the annotations on hidden triplets.
    # --------------------------------------------------

    # todo 这里执行的是 infer step，记录的是一个推理的结果？
    if args.record:
        # Annotate hidden triplets， id 转化为名称，便于阅读
        scores = kge_model.infer_step(kge_model, hidden_triples, args)
        with open(local_path + '/annotation.txt', 'w') as fo:
            for (h, r, t), s in zip(hidden_triples, scores):
                fo.write('{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], s))

    # todo 这个 evaluate_train 与 train 同时进行的 valid 有什么区别，为什么要单独设置一个参数来控制这部分评估工作？
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics, preds = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
