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
        print("hello")

    def build_embeddings(self):
        initializer = nn.init.xavier_uniform_
        self.entity_emb_matrix = initializer(torch.empty(self.n_entities, self.emb_size))
        self.last_entity_init = initializer(torch.empty(1, self.emb_size))
        self.relation_emb_matrix = initializer(torch.empty(self.n_relations, self.emb_size))
        self.ent_transfer = initializer(torch.empty(self.n_entities, self.emb_size))
        self.rel_transfer = initializer(torch.empty(self.n_relations, self.emb_size))
        self.transform_matrix = initializer(torch.empty(self.emb_size, self.emb_size))
        self.oR_matrix = initializer(torch.empty(self.emb_size, self.emb_size))
