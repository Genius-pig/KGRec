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

    def forward(self, batch=None):
        items = batch['items']
        memories_h = batch['memories_h']
        memories_r = batch['memories_r']
        memories_t = batch['memories_t']
        items_emb = self.entity_emb_matrix[items]
        for i in range(self.context_hops):
            memories_h = torch.reshape(memories_h, [-1, self.n_memory])
            memories_r = torch.reshape(memories_r, [-1, self.n_memory])
            memories_t = torch.reshape(memories_t, [-1, self.n_memory])
            self.h_emb_list.append(self.entity_emb_matrix[memories_h])
            self.r_emb_list.append(self.relation_emb_matrix[memories_r])
            self.t_emb_list.append(self.entity_emb_matrix[memories_t])

    def intra_inter_group_attention(self):
        o_list = []
        for hop in range(self.n_hop):
            h_expanded = torch.expand_dims(self.h_emb_list[hop], axis=3)
            Rh = torch.squeeze(torch.matmul(self.r_emb_list[hop], h_expanded), 3)
            Rh = torch.reshape(Rh, shape=[-1, self.n_relations, self.n_memory, self.embedding_size])
            v = torch.expand_dims(self.item_embeddings, axis=1)
            v = torch.expand_dims(v, axis=-1)
            probs = torch.squeeze(torch.matmul(Rh, v), 3)
            probs = torch.reshape(probs, shape=[-1, self.n_memory])
            probs_normalized = torch.softmax(probs, dim=-1)
            probs_expanded = torch.expand_dims(probs_normalized, axis=2)
            o = torch.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
            o = torch.reshape(o, shape=[-1, self.n_relations, self.embedding_size])

        return o_list

    def build_embeddings(self):
        initializer = nn.init.xavier_uniform_
        self.entity_emb_matrix = initializer(torch.empty(self.n_entities, self.emb_size))
        self.last_entity_init = initializer(torch.empty(1, self.emb_size))
        self.relation_emb_matrix = initializer(torch.empty(self.n_relations, self.emb_size))
        self.ent_transfer = initializer(torch.empty(self.n_entities, self.emb_size))
        self.rel_transfer = initializer(torch.empty(self.n_relations, self.emb_size))
        self.transform_matrix = initializer(torch.empty(self.emb_size, self.emb_size))
        self.oR_matrix = initializer(torch.empty(self.emb_size, self.emb_size))
