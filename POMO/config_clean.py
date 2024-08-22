import numpy as np
import random

class Config:
    def __init__(self):
        settings = 1
        self.sscale_l3_resmin, self.sscale_l3_resmax = 3, 6
        self.sscale_l2_resmin, self.sscale_l2_resmax = 30, 50
        self.sscale_l1_resmin, self.sscale_l1_resmax = 90, 110
        self.lscale_l3_resmin, self.lscale_l3_resmax = 2, 5
        self.lscale_l2_resmin, self.lscale_l2_resmax = 30, 50
        self.lscale_l1_resmin, self.lscale_l1_resmax = 70, 90
        self.initialize_dimensions(settings)
        self.initialize_scopes(settings)
        self.initialize_racks()
        self.initialize_resources()
        self.initialize_demands(settings)
        self.initialize_constraints(settings)
        self.position_cols = ['pos_id', 'status', 'scope']
        self.rack_type_cols = ['rt_id', 'group'] + [f'res_{i}' for i in range(len(self.resource_weights))]
        self.scope_rack_res_array = np.zeros((self.num_scopes, len(self.resource_weights)))
        self.scope_rack_res_spread_array = np.zeros((self.num_scopes, self.num_groups, len(self.resource_weights)))

    def __repr__(self):
        attr_list = {'sscale_l3_resmin', 'sscale_l3_resmax',\
                'sscale_l2_resmin', 'sscale_l2_resmax',\
                'sscale_l1_resmin', 'sscale_l1_resmax',\
                'lscale_l3_resmin', 'lscale_l3_resmax',\
                'lscale_l2_resmin', 'lscale_l2_resmax',\
                'lscale_l1_resmin', 'lscale_l1_resmax',\
                'num_positions', 'num_rack_types', 'num_groups', 'num_scopes',\
                'num_level2_scopes', 'num_level1_scopes'
                }
        ret = 'In config.py:\n'
        for attr in attr_list:
            ret += f'{attr}: {getattr(self, attr)}, '
        return ret

    def initialize_dimensions(self, setting):
        """Initialize basic dimensions for positions, racks, and scope groups."""
        if setting==1:
            self.num_positions = 1000
            self.num_rack_types = 10
            self.num_groups = 4
            self.num_scopes = 50
            self.num_level2_scopes = 10
            self.num_level1_scopes = 2
        else:
            self.num_positions = 9000
            self.num_rack_types = 10
            self.num_groups = 4
            self.num_scopes = 600
            self.num_level2_scopes = 30
            self.num_level1_scopes = 15

    def initialize_scopes(self, setting):
        """Configure scopes, levels, and their mappings."""
        # self.level2_scope = list(range(self.num_scopes, self.num_scopes + self.num_level2_scopes))
        # self.level1_scope = list(range(self.num_scopes + self.num_level2_scopes, self.num_scopes + self.num_level2_scopes + self.num_level1_scopes))
        if setting==1:
            self.level23_map = np.array([n // 5 for n in range(self.num_scopes)])
            self.level13_map = np.array([n // 25 for n in range(self.num_scopes)])
            self.scopes = np.array([n // 20 for n in range(self.num_positions)])
        else:
            self.level23_map = np.array([n // 20 for n in range(self.num_scopes)])
            self.level13_map = np.array([n // 40 for n in range(self.num_scopes)])
            self.scopes = np.array([n // 15 for n in range(self.num_positions)])
        np.random.shuffle(self.level23_map)
        np.random.shuffle(self.level13_map)
        np.random.shuffle(self.scopes)

    def initialize_racks(self):
        """Define rack type weights and probability mappings."""
        self.rack_group_weights = [0.25] * self.num_groups
        random_numbers = np.random.rand(10)
        normalized_numbers = random_numbers / random_numbers.sum()
        normalized_numbers *= 0.2
        self.rack_pos_mapping_prob = normalized_numbers.tolist()
        self.rack_pos_mapping_prob.append(0.8)
        self.rt_groups = np.random.choice(self.num_groups, (self.num_rack_types, 1), p=self.rack_group_weights)

    def initialize_resources(self):
        """Generate resource tables and related metrics based on random choices and weights."""
        self.resource_weights = [.25, .4, .25, 0.3, .5, .25, .25, .25, .4, 0.3]
        self.resource_table = []
        self.resource_int = []
        for rack_id in range(self.num_rack_types):
            random_numbers = np.random.random(size=len(self.resource_weights))
            resources = (random_numbers < self.resource_weights).astype(int)
            res_int = sum(res * 2**i for i, res in enumerate(resources))
            self.resource_table.append(resources)
            self.resource_int.append(res_int)
        self.resource_int = np.array(self.resource_int).reshape(self.num_rack_types, 1)
        self.resource_table = np.array(self.resource_table)
        self.max_res_int = 2**len(self.resource_weights) - 1
        self.shift = 10**len(str(self.max_res_int))

    def initialize_demands(self, setting):
        """Set ranges for demand and action limits."""
        if setting==1:
            self.demand_range = [20, 60]
            self.action_limit_range = [80 * self.num_rack_types, 100 * self.num_rack_types]
        else:
            self.demand_range = [200, 600]
            self.action_limit_range = [80 * self.num_rack_types, 100 * self.num_rack_types]

    def initialize_constraints(self, setting):
        """Setup constraints for resources at different scope levels."""
        if setting==1:
            self.L = np.random.randint(self.sscale_l3_resmin, self.sscale_l3_resmax, size=(self.num_scopes, self.num_rack_types))
            self.level2_L = np.random.randint(self.sscale_l2_resmin, self.sscale_l2_resmax, size=(self.num_level2_scopes, self.num_rack_types))
            self.level1_L = np.random.randint(self.sscale_l1_resmin, self.sscale_l1_resmax, size=(self.num_level1_scopes, self.num_rack_types))
        else:
            self.L = np.random.randint(self.lscale_l3_resmin, self.lscale_l3_resmax, size=(self.num_scopes, self.num_rack_types))
            self.level2_L = np.random.randint(self.lscale_l2_resmin, self.lscale_l2_resmax, size=(self.num_level2_scopes, self.num_rack_types))
            self.level1_L = np.random.randint(self.lscale_l1_resmin, self.lscale_l1_resmax, size=(self.num_level1_scopes, self.num_rack_types))
            #self.L = np.random.randint(2, 5, size=(self.num_scopes, self.num_rack_types))
            #self.level2_L = np.random.randint(30, 50, size=(self.num_level2_scopes, self.num_rack_types))
            #self.level1_L = np.random.randint(70, 90, size=(self.num_level1_scopes, self.num_rack_types))

# Usage
config = Config()
