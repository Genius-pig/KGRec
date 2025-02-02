import collections
import os
from time import time
from logging import getLogger
import random

import numpy as np
import torch
from prettytable import PrettyTable
from tqdm import tqdm

from modules.KGAN.KGAN import KGAN
from utils.data_loader import load_data_kgan
from utils.evaluate import test
from utils.helper import init_logger, early_stopping
from utils.parser import parse_args_kgan as parse_args


def get_aggregate_set(aggregate_args, kg, user_history_dict, relation_set, n_entity):
    aggregate_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict, ascii=True):
        for hop in range(aggregate_args.context_hops):

            ### 第一个hop
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
                    if entity in kg:
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


def get_feed_dict(args, data, aggregate_set, relation_set, start, end, n_items, train_user_set):
    feed_dict = dict()
    # feed_dict["labels"] = data[start:end, 2]
    feed_dict["pos_items"] = torch.LongTensor(data[start:end, 1]).to(device)

    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item:
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(data[start:end], train_user_set)).to(device)
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
    torch.autograd.set_detect_anomaly(True)

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
                batch = get_feed_dict(args, train_cf_shuffle, aggregate_set, relation_set, s, s + args.batch_size,
                                      n_params['n_items'], user_dict['train_user_set'])
                batch_loss = model(batch)
                optimizer.zero_grad()
                batch_loss.backward(retain_graph=True)
                optimizer.step()

                loss += batch_loss
                s += args.batch_size
                pbar.update(1)
        train_e_t = time()

        if epoch >= 1:
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision",
                                     "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'],
                 ret['precision'], ret['hit_ratio']]
            )
            logger.info(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            ndcg = ret['ndcg'][0]
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '-KGIN.ckpt')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            logger.info('using time %.4f, training loss at epoch %d: %.4f' % (
            train_e_t - train_s_t, epoch, loss.item()))

    logger.info('early stopping at %d, recall@20:%.4f, ngcg@20:%.4f' % (epoch, cur_best_pre_0, ndcg))
