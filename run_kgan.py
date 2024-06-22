import collections

import numpy as np
from utils.data_loader import load_data_kgan
from utils.parser import parse_args_kgan as parse_args


def get_aggregate_set(args, kg, user_history_dict, relation_set, n_entity):
    aggregate_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.context_hops):

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_aggregation_R = aggregate_set[user][-1]
                tails_of_last_hop = []
                for relation in list(relation_set):
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
                        if tail_and_relation[1] == relation:
                            h.append(entity)
                            r.append(tail_and_relation[1])
                            t.append(tail_and_relation[0])
                if len(r) == 0:
                    _aggregation_R.append([[n_entity], [relation], [n_entity]])
                else:
                    _aggregation_R.append([h, r, t])

            for relation in list(relation_set):
                replace = len(_aggregation_R[relation][0]) < args.n_memory
                indices = np.random.choice(len(_aggregation_R[relation][0]), size=args.n_memory, replace=replace)
                _aggregation_R[relation][0] = [_aggregation_R[relation][0][i] for i in indices]
                _aggregation_R[relation][1] = [_aggregation_R[relation][1][i] for i in indices]
                _aggregation_R[relation][2] = [_aggregation_R[relation][2][i] for i in indices]
            aggregate_set[user].append(_aggregation_R)

    return aggregate_set


if __name__ == '__main__':
    global args, device
    args = parse_args()
    load_data_kgan(args)

