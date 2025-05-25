#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import logging

import utils.logUtils
from models.test import test_img
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, get_model, vgg11
from models.resnet20 import resnet20
from models.Update import LocalUpdate
from options.options_lp_attack import args_parser_lp_attack
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from options.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.defense import fltrust, multi_krum, get_update, RLR, flame, get_update2, fld_distance, detection, detection1, parameters_dict_to_vector_flt, lbfgs_torch, layer_krum, flare
from utils.text_helper import TextHelper
from models.Attacker import attacker
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
import yaml
import datetime
from utils.text_load import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')



def write_file(filename, accu_list, back_list, args, analyse=False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    if args.defence == "krum":
        krum_file = filename + "_krum_dis"
        torch.save(args.krum_distance, krum_file)
    if args.defence == "flare":
        benign_file = filename + "_benign_dis.torch"
        malicious_file = filename + "_malicious_dis.torch"
        torch.save(args.flare_benign_list, benign_file)
        torch.save(args.flare_malicious_list, malicious_file)
        f.write('\n')
        f.write("avg_benign_list=")
        f.write(str(np.mean(args.flare_benign_list)))
        f.write('\n')
        f.write("avg_malicious_list=")
        f.write(str(np.mean(args.flare_malicious_list)))
    if analyse == True:
        need_length = len(accu_list) // 10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc), 2)
        average_back = round(np.mean(back), 2)
        best_back = round(max(back), 2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

# lp_attack
optionChoose = 'lp_attack'

if __name__ == '__main__':

    # parse args
    if optionChoose == 'lp_attack':
        args = args_parser_lp_attack()
    else:
        raise NotImplementedError


    if args.attack == 'lp_attack':
        args.attack = 'adaptive'  # adaptively control the number of attacking layers
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if not os.path.isdir('./' + args.save):
        os.makedirs('./' + args.save)
    log = utils.logUtils.init_logger(logging.DEBUG,args.save)

    log.debug(f"运行设备: {args.device}")
    print_exp_details(log,args)
    # ----------------------------------------------------------------------------------- Load dataset and split users ----------------------------------------------------------------------------------- #
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            # dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
            dict_users = cifar_noniid([x[1] for x in dataset_train], args.num_users, 10, args.p)
            print('main_fed.py line 137 len(dict_users):', len(dict_users))
    elif args.dataset == 'reddit':
        with open(f'./utils/words.yaml', 'r') as f:
            params_loaded = yaml.safe_load(f)
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        helper = TextHelper(
            current_time=current_time,
            params=params_loaded,
            name=params_loaded.get('name', 'text'),
        )
        helper.load_data()
        dataset_train = helper.train_data
        dict_users = cifar_iid(dataset_train, args.num_users)
        dataset_test = helper.test_data
        args.helper = helper
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    # dict_users {
    #     0: {1, 3, 5, ...},  # 客户端0的数据索引
    #     1: {2, 4, 6, ...},  # 客户端1的数据索引
    #     ...
    # }

    # ----------------------------------------------------------------------------------- Load dataset and split users ----------------------------------------------------------------------------------- #



    # ------------------------------------------------------------------------------------------- Load model -------------------------------------------------------------------------------------------- #
    log.debug(f"模型选择: {args.model}")
    if args.model == 'VGG' and args.dataset == 'cifar':
        global_model = vgg19_bn().to(args.device)
    elif args.model == 'VGG11' and args.dataset == 'cifar':
        global_model = vgg11().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        global_model = ResNet18().to(args.device)
    elif args.model == "resnet20" and args.dataset == 'cifar':
        global_model = resnet20().to(args.device)
    elif args.model == "rlr_mnist" or args.model == "cnn":
        global_model = get_model('fmnist').to(args.device)
    elif args.model == 'lstm':
        helper.create_model()
        global_model = helper.local_model.to(args.device)
    else:
        exit('Error: unrecognized model')
    # ------------------------------------------------------------------------------------------- Load model -------------------------------------------------------------------------------------------- #

    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    # training
    loss_train = []

    args.flare_benign_list=[]
    args.flare_malicious_list=[]
    if args.defence == 'fld':
        old_update_list = []
        weight_record = []
        update_record = []
        args.frac = 1
        malicious_score = torch.zeros((1, 100))

    if math.isclose(args.malicious, 0):
        # 模型acc达到100才会投毒，其实就是不投毒了
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin

    log.debug(f"模型acc达到后{backdoor_begin_acc}开始投毒")
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)  # get root dataset for FLTrust
    base_info = get_base_info(args)
    filename = './' + args.save + '/accuracy_file_{}.txt'.format( base_info)  # log hyperparameters
    log.debug(f"文件最终保存位置:{filename}")
    if args.init != 'None':  # continue from checkpoint
        param = torch.load(args.init)
        global_model.load_state_dict(param)
        log.info(f"读取模型来自{param}")

    val_acc_list = [0.0001]  # Acc list
    backdoor_acculist = [0]  # BSR list

    args.attack_layers = []  # keep LSA

    if args.attack == "dba":
        args.dba_sign = 0  # control the index of group to attack
    if args.log_distance:
        args.krum_distance = []
        args.krum_layer_distance = []
    malicious_list = []  # list of the index of malicious clients
    for i in range(int(args.num_users * args.malicious)):
        malicious_list.append(i)
    log.info(f"恶意客户端设置为:{malicious_list}")

    if args.all_clients:
        log.info("Aggregation over all clients")
        # w_locals存储每个客户端的本地模型参数
        w_locals = [global_weights for i in range(args.num_users)]
    for iter in range(args.epochs):
        log.info(f"---------------------------------训练开始:{iter}--------------------------------------------")
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)  # number of clients in each round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select the clients for a single round
        log.debug(f"本轮选中的客户端:{idxs_users}")
        if args.defence == 'fld':
            idxs_users = np.arange(args.num_users)
            if iter == 350:  # decrease the lr in the specific round to improve Acc
                args.lr *= 0.1

        if backdoor_begin_acc < val_acc_list[-1]:  # start attack only when Acc overtakes backdoor_begin_acc
            backdoor_begin_acc = 0
            attack_number = int(args.malicious * m)  # number of malicious clients in a single round
        else:
            attack_number = 0
        if args.scaling_attack_round != 1:
            # scaling attack begin 100-th round and perform each args.attack_round round
            if iter > 100 and iter%args.scaling_attack_round == 0:
                attack_number = attack_number
            else:
                attack_number = 0
        log.info(f"恶意客户端数量:{attack_number}")
        mal_weight=[]
        mal_loss=[]
        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:
                # upload models for malicious clients
                log.info(f"恶意客户端:{idx}开始攻击")
                args.iter = iter
                if args.defence == 'fld':
                    args.old_update_list = old_update_list[0:int(args.malicious * m)]
                    m_idx = idx
                else:
                    m_idx = None
                mal_weight, loss, args.attack_layers = attacker(malicious_list, attack_number, args.attack, dataset_train, dataset_test, dict_users, global_model, args, idx = m_idx)
                attack_number -= 1
                w = mal_weight[0]
            else:
                # upload models for benign clients
                log.info(f"良性客户端:{idx}训练")
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(global_model).to(args.device))
            if args.defence == 'fld':
                w_updates.append(get_update2(w, global_weights)) # ignore num_batches_tracked, running_mean, running_var
            else:
                w_updates.append(get_update(w, global_weights))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        if args.defence == 'avg':  # no defence
            global_weights = FedAvg(w_locals)
        elif args.defence == 'krum':  # single krum
            selected_client = multi_krum(w_updates, 1, args)
            global_weights = w_locals[selected_client[0]]
        elif args.defence == 'multikrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            global_weights = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'RLR':
            global_weights = RLR(copy.deepcopy(global_model), w_updates, args)
        elif args.defence == 'fltrust':
            local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
            fltrust_norm, loss = local.train(
                net=copy.deepcopy(global_model).to(args.device))
            fltrust_norm = get_update(fltrust_norm, global_weights)
            global_weights = fltrust(w_updates, fltrust_norm, global_weights, args)
        elif args.defence == 'flare':
            global_weights = flare(w_updates, w_locals, global_model, central_dataset, dataset_test, global_weights, args)
        elif args.defence == 'flame':
            global_weights = flame(w_locals, w_updates, global_weights, args, debug=args.debug)
        elif args.defence == 'flip':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            global_weights = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'flip_multikrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            global_weights = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'layer_krum':
            w_glob_update = layer_krum(w_updates, args.k, args, multi_k=True)
            for key, val in global_weights.items():
                global_weights[key] += w_glob_update[key]


        elif args.defence == 'fld':
            # ignore key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var'
            N = 5
            args.N = N
            weight = parameters_dict_to_vector_flt(global_weights)
            local_update_list = []
            for local in w_updates:
                local_update_list.append(-1*parameters_dict_to_vector_flt(local).cpu()) # change to 1 dimension

            if iter > N+1:
                args.hvp = lbfgs_torch(args, weight_record, update_record, weight - last_weight)
                hvp = args.hvp

                attack_number = int(args.malicious * m)
                distance = fld_distance(old_update_list, local_update_list, global_model, attack_number, hvp)
                distance = distance.view(1,-1)
                malicious_score = torch.cat((malicious_score, distance), dim=0)
                if malicious_score.shape[0] > N+1:
                    if detection1(np.sum(malicious_score[-N:].numpy(), axis=0)):

                        label = detection(np.sum(malicious_score[-N:].numpy(), axis=0), int(args.malicious * m))
                    else:
                        label = np.ones(100)
                    selected_client = []
                    for client in range(100):
                        if label[client] == 1:
                            selected_client.append(client)
                    new_w_glob = FedAvg([w_locals[client] for client in selected_client])
                else:
                    new_w_glob = FedAvg(w_locals)  # avg
            else:
                hvp = None
                new_w_glob = FedAvg(w_locals)  # avg

            update = get_update2(global_weights, new_w_glob)  # w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
            update = parameters_dict_to_vector_flt(update)
            if iter > 0:
                weight_record.append(weight.cpu() - last_weight.cpu())
                update_record.append(update.cpu() - last_update.cpu())
            if iter > N:
                del weight_record[0]
                del update_record[0]
            last_weight = weight
            last_update = update
            old_update_list = local_update_list
            global_weights = new_w_glob
        else:
            print("Wrong Defense Method")
            raise NotImplementedError("Wrong Defense Method")

        # copy weight to net_glob
        global_model.load_state_dict(global_weights)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        log.info('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            acc_test, _, back_acc = test_img(global_model, dataset_test, args, test_backdoor=True)
            log.info("Main accuracy: {:.2f}".format(acc_test))
            log.info("Backdoor accuracy: {:.2f}".format(back_acc))
            if args.model == 'lstm':
                val_acc_list.append(acc_test)
            else:
                val_acc_list.append(acc_test.item())

            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, args)

        log.info(f"---------------------------------训练结束:{iter}--------------------------------------------")
    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)

    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label='main task(acc:' + str(best_acc) + '%)')
    plt.plot(backdoor_acculist, label='backdoor task(BBSR:' + str(bbsr) + '%, ABSR:' + str(absr) + '%)')
    plt.legend()
    title = base_info
    plt.title(title)
    plt.savefig('./' + args.save + '/' + title + '.pdf', format='pdf', bbox_inches='tight')

    # testing
    global_model.eval()
    acc_train, loss_train = test_img(global_model, dataset_train, args)
    acc_test, loss_test = test_img(global_model, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    torch.save(global_model.state_dict(), './' + args.save + '/model' + '.pth')
