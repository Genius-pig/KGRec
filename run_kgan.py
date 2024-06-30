import collections
import os
from time import time
from logging import getLogger
import random

import numpy as np
import torch
from tqdm import tqdm

from modules.KGAN.KGAN import KGAN
from utils.data_loader import load_data_kgan
from utils.helper import init_logger
from utils.parser import parse_args_kgan as parse_args


def get_aggregate_set(aggregate_args, kg, user_history_dict, relation_set, n_entity):
    aggregate_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict, ascii=True):
        for hop in range(aggregate_args.context_hops):

            if hop == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_aggregation_R = aggregate_set[user][-1]
                tails_of_last_hop = []
                for relation in list(relation_set):
                    relation = relation - 1
                    t = tails_of_aggregation_R[relation][2]
                    tails_of_last_hop = t + tails_of_last_hop

            _aggregation_R = []
            for relation in list(relation_set):
                h = []
                r = []
                t = []
                for entity in tails_of_last_hop:
                    if entity == n_entity:
                        continue
                    for tail_and_relation in kg[entity]:
                        if tail_and_relation[0] == relation:
                            h.append(entity)
                            r.append(tail_and_relation[0])
                            t.append(tail_and_relation[1])
                if len(r) == 0:
                    _aggregation_R.append([[n_entity], [relation], [n_entity]])
                else:
                    _aggregation_R.append([h, r, t])

            for relation in list(relation_set):
                relation = relation - 1
                replace = len(_aggregation_R[relation][0]) < aggregate_args.n_memory
                indices = np.random.choice(len(_aggregation_R[relation][0]), size=aggregate_args.n_memory,
                                           replace=replace)
                _aggregation_R[relation][0] = [_aggregation_R[relation][0][i] for i in indices]
                _aggregation_R[relation][1] = [_aggregation_R[relation][1][i] for i in indices]
                _aggregation_R[relation][2] = [_aggregation_R[relation][2][i] for i in indices]
            aggregate_set[user].append(_aggregation_R)

    return aggregate_set


def get_feed_dict(args, data, aggregate_set, relation_set, start, end):
    feed_dict = dict()
    # feed_dict["labels"] = data[start:end, 2]
    feed_dict["items"] = torch.LongTensor(data[start:end, 1]).to(device)
    memories_h = []
    memories_r = []
    memories_t = []

    for i in range(args.context_hops):
        m_h = []
        m_r = []
        m_t = []
        for user in data[start:end, 0]:
            h = []
            r = []
            t = []
            for relation in list(relation_set):
                relation = relation - 1
                h.append(aggregate_set[user][i][relation][0])
                r.append(aggregate_set[user][i][relation][1])
                t.append(aggregate_set[user][i][relation][2])
            m_h.append(h)
            m_r.append(r)
            m_t.append(t)
        memories_h.append(torch.LongTensor(m_h).to(device))
        memories_r.append(torch.LongTensor(m_r).to(device))
        memories_t.append(torch.LongTensor(m_t).to(device))

    feed_dict["memories_h"] = memories_h
    feed_dict["memories_r"] = memories_r
    feed_dict["memories_t"] = memories_t
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """log"""
    log_fn = init_logger(args)
    logger = getLogger()

    logger.info('PID: %d', os.getpid())
    logger.info(f"DESC: {args.desc}\n")

    train_cf, test_cf, relation_set, triplets, user_dict, n_params = load_data_kgan(args)

    """define model"""
    model = KGAN(n_params, args).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    aggregate_set = get_aggregate_set(args, triplets, user_dict['train_user_set'], relation_set, n_params['n_entities'])

    cur_best_pre_0 = 0
    ndcg = 0
    stopping_step = 0
    should_stop = False

    logger.info("start training ...")

    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_shuffle = train_cf[index]
        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        with tqdm(total=len(train_cf) // args.batch_size) as pbar:
            while s + args.batch_size <= len(train_cf):
                batch = get_feed_dict(args, train_cf_shuffle, aggregate_set, relation_set, s, s + args.batch_size)
                batch_loss, _, _, batch_cor = model(batch)
                batch_loss = batch_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                cor_loss += batch_cor
                s += args.batch_size
                pbar.update(1)

        train_e_t = time()
