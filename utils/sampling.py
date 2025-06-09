#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
def tinyimagenet_noniid(dataset, num_users, num_shards_per_user=2):
    """
    Sample non-I.I.D. client data from TinyImageNet by shard-label method
    类似于 MNIST 的非 IID 分布，每个 client 只看到几个 label
    :param dataset: TinyImageNetDataset 对象（必须支持索引访问和 label 获取）
    :param num_users: 客户端数量
    :param num_shards_per_user: 每个 client 拿几个 label shard（默认 2）
    :return: dict_users: {client_id: set(image indices)}
    """
    num_shards = num_users * num_shards_per_user
    num_classes = 200
    assert num_shards <= num_classes * 10, "shard 数量太大，超出类分布能力"

    # 创建每个 label → 样本 index 的映射
    label2idx = {i: [] for i in range(num_classes)}
    for i, (_, label) in enumerate(dataset):
        label2idx[label].append(i)

    # 按 label 划分 shard，每个 shard 包含固定数量样本（如 50 张）
    shards = []
    shard_size = 50  # 每个 shard 含多少样本，可调节
    for label in label2idx:
        idxs = label2idx[label]
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), shard_size):
            shard = idxs[i:i+shard_size]
            if len(shard) == shard_size:
                shards.append(shard)

    assert len(shards) >= num_shards, "shard 总数不足"

    # 分配 shard 给用户
    np.random.shuffle(shards)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    shard_idx = 0
    for user in range(num_users):
        for _ in range(num_shards_per_user):
            dict_users[user] = np.concatenate((dict_users[user], shards[shard_idx]), axis=0)
            shard_idx += 1

    return dict_users
def tinyimagenet_iid(dataset, num_users):
    """
    Sample I.I.D. client data from TinyImageNet dataset
    :param dataset: TinyImageNetDataset 对象
    :param num_users: 客户端数量
    :return: dict_users: {client_id: set(image indices)}
    """
    num_items = int(len(dataset) / num_users)
    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)
    dict_users = {i: set(all_idxs[i * num_items:(i + 1) * num_items]) for i in range(num_users)}
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset_label, num_clients, num_classes, q):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    return dict_users

def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(np.array(dataset_label) == k)[0]
        num_idx_k = len(idx_k)
        
        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k*q) , replace=False))
        dict_users[k] = dict_users[k]|selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set(np.random.choice(idx_k, int(num_idx_k*(1-q)/(num_classes-1)) , replace=False))
            dict_users[other_group] = dict_users[other_group]|selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs),' samples are remained')
    print('random put those samples into groups')
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i]|selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs),' samples are remained after relocating')
    return dict_users


def non_iid_distribution_client(group_proportion, num_clients, num_classes):
    num_each_group = num_clients // num_classes
    num_data_each_client = len(group_proportion[0]) // num_each_group
    dict_users, all_idxs = {}, [i for i in range(num_data_each_client*num_clients)]
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        for j in range(num_each_group):
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
            dict_users[i*10+j] = selected_data
            group_data = list(set(group_data) - selected_data)
            all_idxs = list(set(all_idxs) - selected_data)
    print(len(all_idxs),' samples are remained')
    return dict_users


def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
    for client in client_data_proportion.keys():
        client_data = dataset_label[list(client_data_proportion[client])]
        print('client', client, 'distribution information:')
        for i in range(num_classes):
            print('class ', i, ':', len(client_data[client_data==i])/len(client_data))


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
