import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
from logging import getLogger


class AttnHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_relations, n_factors, ind, interact_mat,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.ind = ind
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.n_factors = n_factors

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def non_attn_agg_m(self, user_emb, entity_emb, edge_index, edge_type, latent_emb, interact_mat,
                       weight, disen_weight_att):
        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        """cul user->latent factor attention"""
        "equation 8"
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]

        "equation 1，这里我猜是为了后面计算用了expand"
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                weight).expand(n_users, n_factors, channel)
        "有可能是equation 7，但是为什么要加user_agg"
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, interact_mat, mess_dropout=True, item_attn=None, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(self.n_hops):
            entity_emb, user_emb = self.non_attn_agg_m(user_emb, entity_emb, edge_index,
                                                       edge_type, latent_emb, interact_mat,
                                                       self.weight, self.disen_weight_att)
            # self, user_emb, entity_emb, edge_index, edge_type, latent_emb, interact_mat,
            # weight, disen_weight_att
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        user_res_emb = user_emb
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        return user_emb

    # def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
    #     entity_res_emb = entity_emb
    #     for i in range(self.n_hops):
    #         entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
    #         """message dropout"""
    #         if mess_dropout:
    #             entity_emb = self.dropout(entity_emb)
    #         entity_emb = F.normalize(entity_emb)
    #
    #         """result emb"""
    #         entity_res_emb = torch.add(entity_res_emb, entity_emb)
    #     return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # print attn score
        if print:
            self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))