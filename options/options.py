#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # 保存位置
    parser.add_argument('--save', type=str, default='save', help="dic to save results (ending without /)")   # 失效！
    # 读取保存模型（无初始值）
    parser.add_argument('--init', type=str, default='None', help="location of init model")
    # 总迭代数
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    # 总客户端数
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    # 采样客户端数
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    # 恶意客户端数
    parser.add_argument('--malicious', type=float, default=0, help="proportion of mailicious clients")

    # ***** badnet labelflip layerattack updateflip get_weight  adaptive****
    # 攻击方式
    parser.add_argument('--attack', type=str, default='badnet', help='attack method')
    # 不懂？
    parser.add_argument('--ada_mode', type=int, default=1, help='adaptive attack mode')
    # 投毒比例，如果是层级投毒，则设置为1
    parser.add_argument('--poison_frac', type=float, default=0.2,help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")

    # *****local_ep = 3, local_bs=50, lr=0.1*******
    # 本地客户端内部迭代数
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    # 客户端每处理完{arg.local_bs}条样本数据后才会执行一次梯度下降更新
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    # 测试数据批处理大小
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    # 学习率
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    # model arguments
    # *************************model******************************#
    # resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist
    # 模型选择
    parser.add_argument('--model', type=str, default='Mnist_CNN', help='model name')

    # other arguments
    # *************************dataset*******************************#
    # fashion_mnist mnist cifar
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")

    # ****0-avg, 1-fltrust 2-tr-mean 3-median 4-krum 5-muli_krum 6-RLR fltrust_bn fltrust_bn_lr****#
    # 防御方案选择
    parser.add_argument('--defence', type=str, default='avg', help="strategy of defence")
    # krum防御方案的参数
    parser.add_argument('--k', type=int, default=2, help="parameter of krum")
    # parser.add_argument('--iid', action='store_true',
    #                     help='whether i.i.d or not')
    # 客户端是否独立同分布
    parser.add_argument('--iid', type=int, default=1,help='whether i.i.d or not')

    # ************************atttack_label********************************#
    # 触发标签
    parser.add_argument('--attack_label', type=int, default=5,help="trigger for which label")
    # 没懂？
    parser.add_argument('--single', type=int, default=0,help="single shot or repeated")
    # 设置为-1表示投毒任何标签的数据，设置为其他，只投毒这种标签的数据集
    parser.add_argument('--attack_goal', type=int, default=7,help="trigger to which label")
    # 只有模型ACC达到某个值时开始进行攻击
    parser.add_argument('--attack_begin', type=int, default=0,help="the accuracy begin to attack")
    # 没懂？
    parser.add_argument('--search_times', type=int, default=20,help="binary search times")
    # 设备选择
    parser.add_argument('--gpu', type=int, default=0,help="GPU ID, -1 for CPU")
    parser.add_argument('--robustLR_threshold', type=int, default=4,help="break ties when votes sum to 0")

    parser.add_argument('--server_dataset', type=int, default=200, help="number of dataset in server")

    parser.add_argument('--server_lr', type=float, default=1, help="number of dataset in server using in fltrust")

    parser.add_argument('--momentum', type=float, default=0.9,help="SGD momentum (default: 0.9)")

    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    # *********trigger info*********
    #  square  apple  watermark  
    parser.add_argument('--trigger', type=str, default='square',help="Kind of trigger")
    # mnist 28*28  cifar10 32*32
    parser.add_argument('--triggerX', type=int, default='0',help="position of trigger x-aix")
    parser.add_argument('--triggerY', type=int, default='0',help="position of trigger y-aix")

    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1,help='random seed (default: 1)')
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)

    parser.add_argument('--mal_score', type=float, default=0)
    parser.add_argument('--ben_score', type=float, default=0)

    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)
    parser.add_argument('--all_clients', action='store_true',help='aggregation over all clients')
    parser.add_argument('--tau', type=float, default=0.8,help="threshold of LPA_ER")
    parser.add_argument('--debug', type=int, default=0, help="log debug info or not")
    parser.add_argument('--local_dataset', type=int, default=1, help="use local dataset for layer identification")
    parser.add_argument('--debug_fld', type=int, default=0, help="#1 save, #2 load")
    parser.add_argument('--decrease', type=float, default=0.3,help="proportion of dropped layers in robust experiments (used in mode11)")
    parser.add_argument('--increase', type=float, default=0.3,help="proportion of added layers in robust experiments (used in mode12)")
    parser.add_argument('--mode10_tau', type=float, default=0.95, help="threshold of mode 10")
    parser.add_argument('--cnn_scale', type=float, default=0.5, help="scale of cnn")
    parser.add_argument('--cifar_scale', type=float, default=1.0, help="scale of larger model")

    parser.add_argument('--num_layer', type=int, default=3, help="fixed number of layer attacks")

    parser.add_argument('--num_identification', type=int, default=1, help="fixed number of round to identify")
    parser.add_argument('--beta', type=float, default=0.5,help="weight of regularization loss in distance awareness attacks")
    parser.add_argument('--log_distance', type=bool, default=False, help="output krum distance")
    parser.add_argument('--scaling_attack_round', type=int, default=1, help="rounds of attack implements")
    parser.add_argument('--scaling_param', type=float, default=5, help="scaling up how many times")
    parser.add_argument('--p', type=float, default=0.5, help="level of non-iid")
    args = parser.parse_args()
    return args
