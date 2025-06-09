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
from options.config import read_config
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from options.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid,tinyimagenet_iid,tinyimagenet_noniid
from utils.defense import fltrust, multi_krum, get_update, RLR, flame, get_update2, fld_distance, detection, detection1, \
    parameters_dict_to_vector_flt, lbfgs_torch, layer_krum, flare, median_aggregation
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
import wandb
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
        torch.save(args.log_distance, krum_file)   # 这里把krum_distance改成了log_distance
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


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # 是否启用wandb
    wandb_enable = True
    set_seed(0)
    # parse args
    args = read_config()

    # 修改配置
    # badnet/lp_attack/opt
    args.attack = 'badnet'
    # avg/medium/krum/muli_krum/RLR/flame
    args.defence = 'avg'
    # opt/square
    args.trigger = 'square'

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.attack == 'lp_attack':
        args.attack = 'adaptive'  # adaptively control the number of attacking layers
        args.poison_frac = 1.0

    args.save = './save/' + f"{args.attack}-{args.defence}-{args.trigger}-{args.dataset}-{args.model}"
    # 保存路径创建
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    # 日志系统启动
    log = utils.logUtils.init_logger(logging.DEBUG,args.save)
    if wandb_enable:
        run = wandb.init(project="optBC-exp",name=args.save.replace("./save/","").replace("adaptive","BC"),group="对照")
    args.log = log

    # 初始化自适应触发器
    if args.trigger == 'opt':
        trigger_size = 5
        args.optTrigger = torch.ones((1,3, 32, 32), requires_grad=False, device='cuda') * 0.5
        args.mask = torch.zeros_like(args.optTrigger)
        args.mask[:, :, args.triggerX:args.triggerX + trigger_size,
        args.triggerY:args.triggerY + trigger_size] = 1
        args.mask = args.mask.to('cuda')

    log.debug(f"运行设备: {args.device}")
    print_exp_details(log,args)


    # ----------------------------------------------------------------------------------- Load dataset and split users ----------------------------------------------------------------------------------- #
    log.debug(f"数据集: {args.dataset}")
    log.debug(f"独立同分布: {args.iid}")
    log.debug(f"开始加载数据集:{args.dataset}")
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
            log.debug('main_fed.py line 137 len(dict_users):', len(dict_users))
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
    elif args.dataset == 'tinyimagenet':
        from data.tinyImageNet import TinyImageNetDataset  # 假设你将自定义类放在这个文件中

        data_dir = '../data/tiny-imagenet-200'
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 可微调
        ])

        dataset_train = TinyImageNetDataset(data_dir, split='train', transform=transform)
        dataset_test = TinyImageNetDataset(data_dir, split='val', transform=transform)

        # 联邦划分（需要你定义 tinyimagenet_iid 和 tinyimagenet_noniid 函数）
        if args.iid:
            dict_users = tinyimagenet_iid(dataset_train, args.num_users)
        else:
            dict_users = tinyimagenet_noniid(dataset_train, args.num_users, args.p)
    else:
        raise NotImplementedError("Error: unrecognized dataset")
    log.debug(f"训练集加载成功,数据集大小: {len(dataset_train)}")
    # img_size = dataset_train[0][0].shape
    # dict_users {
    #     0: {1, 3, 5, ...},  # 客户端0的数据索引
    #     1: {2, 4, 6, ...},  # 客户端1的数据索引
    #     ...
    # }

    # ----------------------------------------------------------------------------------- Load dataset and split users ----------------------------------------------------------------------------------- #



    # ------------------------------------------------------------------------------------------- Load model -------------------------------------------------------------------------------------------- #
    log.debug(f"全局模型: {args.model}")
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
    filename =args.save+ '/accuracy_file_{}.txt'.format( base_info)  # log hyperparameters
    log.debug(f"文件最终保存位置:{filename}")
    if args.init != 'None':  # continue from checkpoint
        param = torch.load(args.init)
        global_model.load_state_dict(param)
        log.debug(f"读取模型来自{param}")

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
    log.debug(f"恶意客户端设置为:{malicious_list}")

    if args.all_clients:
        log.debug("所有客户端都参与聚合")
        # w_locals存储每个客户端的本地模型参数
        w_locals = [global_weights for i in range(args.num_users)]     # [{},{},{}] 每个客户一个字典
    for iter in range(args.epochs):
        log_dict = {}
        log.info(f"---------------------------------训练开始:{iter}--------------------------------------------")
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)  # number of clients in each round
        selected_clients_ids = np.random.choice(range(args.num_users), m, replace=False)  # select the clients for a single round
        log.info(f"本轮选中的客户端:{selected_clients_ids}")
        if args.defence == 'fld':
            selected_clients_ids = np.arange(args.num_users)
            if iter == 350:  # decrease the lr in the specific round to improve Acc
                args.lr *= 0.1

        if backdoor_begin_acc < val_acc_list[-1]:  # start attack only when Acc overtakes backdoor_begin_acc
            backdoor_begin_acc = 0
            attack_client_num = int(args.malicious * m)  # number of malicious clients in a single round
        else:
            attack_client_num = 0
        if args.scaling_attack_round != 1:
            # scaling attack begin 100-th round and perform each args.attack_round round
            if iter > 100 and iter%args.scaling_attack_round == 0:
                attack_client_num = attack_client_num
            else:
                attack_client_num = 0
        log.info(f"恶意客户端数量:{attack_client_num}")
        mal_weight=[]
        mal_loss=[]
        for num_turn, selected_client_id in enumerate(selected_clients_ids):
            if attack_client_num > 0:
                # upload models for malicious clients
                args.iter = iter
                if args.defence == 'fld':
                    args.old_update_list = old_update_list[0:int(args.malicious * m)] # ?
                    m_idx = selected_client_id # ?
                else:
                    m_idx = None
                mal_weight, loss, args.attack_layers = attacker(malicious_list, attack_client_num, args.attack, dataset_train, dataset_test, dict_users, global_model, args, idx = m_idx)
                attack_client_num -= 1
                w = mal_weight[0]
            else:
                # upload models for benign clients
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[selected_client_id])
                w, loss = local.train(net=copy.deepcopy(global_model).to(args.device))
                log.info(f"良性客户端:{selected_client_id}训练")
            if args.defence == 'fld':
                w_updates.append(get_update2(w, global_weights)) # ignore num_batches_tracked, running_mean, running_var
            else:
                w_updates.append(get_update(w, global_weights))
            if args.all_clients:
                w_locals[selected_client_id] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # ------------------------------------------------------------------------------------------- Defence -------------------------------------------------------------------------------------------- #
        log.info(f"开始{args.defence}聚合")
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
        elif args.defence == 'medium':
            global_weights = median_aggregation(w_updates, global_weights)

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

                attack_client_num = int(args.malicious * m)
                distance = fld_distance(old_update_list, local_update_list, global_model, attack_client_num, hvp)
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
            raise NotImplementedError("Wrong Defense Method")
        # ------------------------------------------------------------------------------------------- Defence -------------------------------------------------------------------------------------------- #

        # ------------------------------------------------------------------------------------------- INFO ----------------------------------------------------------------------------------------------- #

        # copy weight to net_glob
        global_model.load_state_dict(global_weights)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        log.info('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        acc_test, _, back_acc = test_img(global_model, dataset_test, args, test_backdoor=True)
        log.info("Main accuracy: {:.2f}".format(acc_test))
        log.info("Backdoor accuracy: {:.2f}".format(back_acc))
        if args.model == 'lstm':
            val_acc_list.append(acc_test)
        else:
            val_acc_list.append(acc_test.item())

        backdoor_acculist.append(back_acc)
        write_file(filename, val_acc_list, backdoor_acculist, args)
        if wandb_enable:
            log_dict = ({"epoch":iter, 'avg_loss': loss_avg, 'acc': acc_test, 'BSR': back_acc})
            wandb.log(log_dict)
        log.info(f"---------------------------------训练结束:{iter}--------------------------------------------")
        # ------------------------------------------------------------------------------------------- INFO ----------------------------------------------------------------------------------------------- #

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)
    print("log save")
    print(args)
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
