import collections
import os
from logging import getLogger
from random import random

import numpy as np
import torch
from tqdm import tqdm

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
    feed_dict["items"] = data[start:end, 1]
    # feed_dict["labels"] = data[start:end, 2]

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

        feed_dict["memories_h"] = m_h
        feed_dict["memories_r"] = m_r
        feed_dict["memories_t"] = m_t

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
    aggregate_set = get_aggregate_set(args, triplets, user_dict['train_user_set'], relation_set, n_params['n_entities'])
    get_feed_dict(args, train_cf, aggregate_set, relation_set, 0, 1024)
