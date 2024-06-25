import numpy as np
import random

class Config:
    def __init__(self):
        self.seed = 0
        self.initialize_dimensions()
        self.initialize_scopes()
        self.initialize_racks()
        self.initialize_resources()
        self.initialize_demands()
        self.initialize_constraints()
        self.position_cols = ['pos_id', 'status', 'scope']
        self.rack_type_cols = ['rt_id', 'group'] + [f'res_{i}' for i in range(len(self.resource_weights))]
        self.scope_rack_res_array = np.zeros((self.num_scopes, len(self.resource_weights)))
        self.scope_rack_res_spread_array = np.zeros((self.num_scopes, self.num_groups, len(self.resource_weights)))

    def initialize_dimensions(self):
        """Initialize basic dimensions for positions, racks, and scope groups."""
        self.num_positions = 9000
        self.num_rack_types = 10
        self.num_groups = 4
        self.num_scopes = 600
        self.num_level2_scopes = 30
        self.num_level1_scopes = 15

    def initialize_scopes(self):
        """Configure scopes, levels, and their mappings."""
        # self.level2_scope = list(range(self.num_scopes, self.num_scopes + self.num_level2_scopes))
        # self.level1_scope = list(range(self.num_scopes + self.num_level2_scopes, self.num_scopes + self.num_level2_scopes + self.num_level1_scopes))
        self.level23_map = np.array([n // 20 for n in range(self.num_scopes)])
        self.level13_map = np.array([n // 40 for n in range(self.num_scopes)])
        self.scopes = np.array([n // 10 for n in range(self.num_positions)])

    def initialize_racks(self):
        """Define rack type weights and probability mappings."""
        self.rack_group_weights = [0.25] * self.num_groups
        self.rack_pos_mapping_prob = [0.2 / self.num_rack_types] * (self.num_rack_types + 1)
        self.rack_pos_mapping_prob[-1] = 0.8
        self.rt_groups = np.random.choice(self.num_groups, (self.num_rack_types, 1), p=self.rack_group_weights)

    def initialize_resources(self):
        """Generate resource tables and related metrics based on random choices and weights."""
        self.resource_weights = [1, 0.5, 0.25, 0.5, 0.75, 0.25, 0.25, 0.25, 0.75, 0.5]
        self.resource_table = []
        self.resource_int = []
        for rack_id in range(self.num_rack_types):
            resources = [1 if random.random() < prob else 0 for prob in self.resource_weights]
            res_int = sum(res * 2**i for i, res in enumerate(resources))
            self.resource_table.append(resources)
            self.resource_int.append(res_int)
        self.resource_int = np.array(self.resource_int).reshape(self.num_rack_types, 1)
        self.resource_table = np.array(self.resource_table)
        self.max_res_int = 2**len(self.resource_weights) - 1
        self.shift = 10**len(str(self.max_res_int))

    def initialize_demands(self):
        """Set ranges for demand and action limits."""
        self.demand_range = [100, 400]
        self.action_limit_range = [80 * self.num_rack_types, 100 * self.num_rack_types]

    def initialize_constraints(self):
        """Setup constraints for resources at different scope levels."""
        self.L = np.array([[100] * 10] * self.num_scopes)
        self.level2_L = np.array([[300] * 10] * self.num_level2_scopes)
        self.level1_L = np.array([[500] * 10] * self.num_level1_scopes)
        self.spread_metrics = [(set(range(100)), set(range(4)), i) for i in range(len(self.resource_weights))]

# Usage
config = Config()
