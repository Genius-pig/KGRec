import torch
from torch import nn


class KGAN(nn.Module):
    def __init__(self, data_config, args_config):
        super(KGAN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']

        self.dim = args_config.dim
        self.n_hop = args_config.context_hops
        self.lr = args_config.lr
        self.n_memory = args_config.n_memory
        self.build_embeddings()
        self.entity_emb_matrix = nn.Parameter(self.entity_emb_matrix)
        self.last_entity_init = nn.Parameter(self.last_entity_init)
        self.entity_emb_matrix = torch.cat([self.entity_emb_matrix, self.last_entity_init], 0)
        self.relation_emb_matrix = nn.Parameter(self.relation_emb_matrix)
        self.ent_transfer = nn.Parameter(self.ent_transfer)
        self.ent_transfer = torch.concat([self.ent_transfer, self.last_entity_init], 0)
        self.rel_transfer = nn.Parameter(self.rel_transfer)
        self.transform_matrix = nn.Parameter(self.transform_matrix)
        self.oR_matrix = nn.Parameter(self.oR_matrix)
        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []

        self.h_t_emb_list = []
        self.r_t_emb_list = []
        self.t_t_emb_list = []

        self.h_transfer_list = []
        self.t_transfer_list = []

        self.attention_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, self.dim, bias=False),
                nn.Dropout(0.5),
                nn.Linear(self.dim, 1, bias=False),
                nn.ReLU()
            ) for _ in range(self.context_hops)
        ])

    def forward(self, batch=None):
        pos_items = batch['pos_items']
        neg_items = batch['neg_items']
        memories_h = batch['memories_h']
        memories_r = batch['memories_r']
        memories_t = batch['memories_t']
        items_emb = self.entity_emb_matrix[pos_items]
        neg_items_emb = self.entity_emb_matrix[neg_items]
        for i in range(self.context_hops):
            memories_h = torch.reshape(memories_h, [-1, self.n_memory])
            memories_r = torch.reshape(memories_r, [-1, self.n_memory])
            memories_t = torch.reshape(memories_t, [-1, self.n_memory])
            self.h_emb_list.append(self.entity_emb_matrix[memories_h])
            self.r_emb_list.append(self.relation_emb_matrix[memories_r])
            self.t_emb_list.append(self.entity_emb_matrix[memories_t])
        o_list = self.intra_inter_group_attention(items_emb)
        scores = self.rating(items_emb, o_list)
        _scores = self.rating(neg_items_emb, o_list)
        return self.build_loss(scores, _scores)

    def intra_inter_group_attention(self, items_emb):
        o_list = []
        for hop in range(self.context_hops):
            h_expanded = torch.expand_dims(self.h_emb_list[hop], axis=3)
            Rh = torch.squeeze(torch.matmul(self.r_emb_list[hop], h_expanded), 3)
            Rh = torch.reshape(Rh, shape=[-1, self.n_relations, self.n_memory, self.embedding_size])
            v = torch.expand_dims(items_emb, axis=1)
            v = torch.expand_dims(v, axis=-1)
            probs = torch.squeeze(torch.matmul(Rh, v), 3)
            probs = torch.reshape(probs, shape=[-1, self.n_memory])
            probs_normalized = torch.softmax(probs, dim=-1)
            probs_expanded = torch.expand_dims(probs_normalized, axis=2)
            o = torch.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
            o = torch.reshape(o, shape=[-1, self.n_relations, self.embedding_size])
            attention = self.attention_layer[hop](o)

            attention = torch.squeeze(attention, 2)

            attention_weight_norm = torch.softmax(attention, dim=-1)

            attention_weight_expand = torch.expand_dims(attention_weight_norm, axis=-1)

            o = torch.reduce_sum(o * attention_weight_expand, axis=1)

            items_emb = self.update_item_embedding(items_emb, o)
            o_list.append(o)

        return o_list

    def update_item_embedding(self, item_embeddings, o):

        item_embeddings = torch.matmul(item_embeddings + o, self.transform_matrix)

        return item_embeddings

    def rating(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = torch.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def build_embeddings(self):
        initializer = nn.init.xavier_uniform_
        self.entity_emb_matrix = initializer(torch.empty(self.n_entities, self.dim))
        self.last_entity_init = initializer(torch.empty(1, self.dim))
        self.relation_emb_matrix = initializer(torch.empty(self.n_relations, self.dim))
        self.ent_transfer = initializer(torch.empty(self.n_entities, self.dim))
        self.rel_transfer = initializer(torch.empty(self.n_relations, self.dim))
        self.transform_matrix = initializer(torch.empty(self.dim, self.dim))
        self.oR_matrix = initializer(torch.empty(self.dim, self.dim))

    def build_loss(self, pos_scores, neg_scores):
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        kge_loss = 0
        for hop in range(self.context_hops):
            h_expanded = torch.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = torch.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = torch.squeeze(torch.matmul(torch.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += torch.reduce_mean(torch.sigmoid(hRt))
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += torch.reduce_mean(torch.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            l2_loss += torch.reduce_mean(torch.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            l2_loss += torch.reduce_mean(torch.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            l2_loss += torch.l2_loss(self.transform_matrix)
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return loss
