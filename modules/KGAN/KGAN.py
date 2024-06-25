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

    def forward(self, batch=None):
        print("hello")
