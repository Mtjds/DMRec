#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author:
# @Date  : 2021/11/1 15:25
# @Desc  :
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from data_set import DataSet
from model import DMRec

from trainer import Trainer


# seed = 2021
def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # True can improve train speed
        torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':

    random_number = random.randint(2020, 2040)
    # 429106
    # 2021
    set_seed(2021)
    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--node_dropout', type=float, default=0.05)
    parser.add_argument('--message_dropout', type=float, default=0.25)
    parser.add_argument('--dim_qk', type=int, default=32)
    parser.add_argument('--dim_v', type=int, default=64)
    parser.add_argument('--omega', type=float, default=1)

    parser.add_argument('--data_name', type=str, default='tmall', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')

    parser.add_argument('--decay', type=float, default=0.0001, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=300, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='a_tmall_base.pth', help='')
    parser.add_argument('--model_name', type=str, default='model_weight_att', help='')
    parser.add_argument('--pt_loop', type=int, default=50, help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')


    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
        args.layers = [4,1,1,2]
    elif args.data_name == 'tmall_cold':
        args.data_path = './data/Tmall_cold_all'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
        args.layers = [2, 2, 2, 2]
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['click', 'cart', 'buy']
        args.layers = [2, 2, 3]
    elif args.data_name == 'beibei_cold':
        args.data_path = './data/beibei_cold_all'
        args.behaviors = ['click', 'cart', 'buy']
        args.layers = [2, 2, 2]
    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
        args.layers = [2, 2, 2, 2]
    elif args.data_name == 'jdata_cold':
        args.data_path = './data/jdata_cold_all'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
        args.layers = [2, 2, 2, 2]
    else:
        raise Exception('data_name cannot be None')


    # args = parser.parse_args()
    # if args.data_name == 'tmall':
    #     args.data_path = './data/Tmall'
    #     args.behaviors = ['click', 'collect', 'cart', 'buy']
    #     args.layers = [1, 1, 1, 1]
    #     args.model_name = 'Tmall'
    # elif args.data_name == 'tmall_cold':
    #     args.data_path = 'data/Tmall_cold_all'
    #     args.behaviors = ['click', 'cart', 'collect', 'buy']
    #     args.model_name = 'Tmall_cold_all'
    # elif args.data_name == 'beibei':
    #     args.data_path = './data/beibei'
    #     args.behaviors = ['view', 'cart', 'buy']
    #     args.layers = [1, 1, 1]
    #     args.model_name = 'beibei'
    # elif args.data_name == 'beibei_cold':
    #     args.data_path = './data/beibei_cold_all'
    #     args.behaviors = ['view', 'cart', 'buy']
    #     args.layers = [1, 1, 1]
    #     args.model_name = 'beibei'
    # elif args.data_name == 'jdata':
    #     args.data_path = './data/jdata'
    #     args.behaviors = ['view', 'collect', 'cart', 'buy']
    #     args.layers = [1, 2, 2, 3]
    #     args.model_name = 'jdata'
    # else:
    #     raise Exception('data_name cannot be None')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device


    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = DMRec(args, dataset)
    logger.info('随机种子为：')
    logger.info(random_number)
    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    # trainer.evaluate(0, 1, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



