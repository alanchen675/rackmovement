import torch
import numpy as np
import random

class ProblemGenerator:
    def __init__(self, config):
        self.config = config

    def get_random_problems(self, batch_size, problem_size, time_steps=5):
        demand, rest_demand, demand_pool = self.generate_demand(batch_size, time_steps)
        action_limit, rest_action_limit, action_limit_pool = self.generate_action_limit(batch_size, time_steps)

        rack_vals = list(range(self.config.num_rack_types + 1))
        problem = []
        pos_rack_map = []

        for batch_id in range(batch_size):
            sub_pos_rack_map, num_prev_pos_assigned = self.generate_pos_rack_mapping(rack_vals)
            x_coord, y_coord = self.encode_attributes_to_coords(batch_id, demand, num_prev_pos_assigned)
            sub_problem = np.hstack((x_coord, y_coord))
            problem.append(sub_problem)
            pos_rack_map.append(sub_pos_rack_map)

        problem = torch.from_numpy(np.array(problem)).float().to('cuda')
        pos_rack_map = torch.from_numpy(np.array(pos_rack_map)).to('cuda')

        return problem, pos_rack_map, torch.tensor(demand), torch.tensor(action_limit), demand_pool, action_limit_pool

    def generate_coord(self, batch_size, pos_rack_map, demand):
        pos_rack_map = pos_rack_map.to('cuda')  # Assume pos_rack_map is a Tensor and move to GPU
        demand = demand.to('cuda')  # Move demand to GPU if not already
        
        x_coords = torch.zeros(batch_size, 1, device='cuda')
        y_coords = torch.zeros(batch_size, 1, device='cuda')

        for batch_id in range(batch_size):
            print(f'[Generate_Coord]Iteration {batch_id}')
            num_prev_pos_assigned = self.generate_num_prev_assigned_v2(pos_rack_map[batch_id])
            x, y = self.encode_attributes_to_coords_v2(demand[batch_id], num_prev_pos_assigned)
            x_coords[batch_id] = x.mean()  # Assuming x_coord needs to be averaged
            y_coords[batch_id] = y.mean()  # Assuming y_coord needs to be averaged

        problem = torch.cat((x_coords, y_coords), dim=1)
        return problem

    def generate_coord_old(self, batch_size, pos_rack_map, demand):
        """
        Generate coordinate for each initial problem instances. Called after the first step.
        pos_rack_map - shape = (batch_size, num_positions, 1)
        demand = 
        """
        problem = []
        for batch_id in range(batch_size):
            x_coord, y_coord = 0, 0
            for pomo_id in range(pos_rack_map.shape[1]):
                num_prev_pos_assigned = self.generate_num_prev_assigned(pos_rack_map[batch_id, pomo_id])
                x, y = self.encode_attributes_to_coords(batch_id, demand, num_prev_pos_assigned)
                x_coord += x
                y_coord += y
            x_coord /= pos_rack_map.shape[1]
            y_coord /= pos_rack_map.shape[1]
            sub_problem = np.hstack((x_coord, y_coord))
            problem.append(sub_problem)
        problem = torch.from_numpy(np.array(problem)).float().to('cuda')
        return problem

    def generate_demand(self, batch_size, time_steps):
        dm_low, dm_high = self.config.demand_range
        demand = np.random.randint(dm_low, dm_high + 1, (batch_size, self.config.num_rack_types))
        rest_demand = np.random.randint(dm_low, dm_high + 1, (time_steps - 1, batch_size, self.config.num_rack_types))
        demand_pool = np.concatenate([demand[None, :, :], rest_demand], axis=0)
        return demand, rest_demand, demand_pool

    def generate_action_limit(self, batch_size, time_steps):
        act_low, act_high = self.config.action_limit_range
        action_limit = np.random.randint(act_low, act_high + 1, (batch_size, 1))
        rest_action_limit = np.random.randint(act_low, act_high + 1, (time_steps - 1, batch_size, 1))
        action_limit_pool = np.concatenate([action_limit[None, :, :], rest_action_limit], axis=0)
        action_limit = np.tile(action_limit, (1, self.config.num_rack_types))
        return action_limit, rest_action_limit, action_limit_pool

    def generate_pos_rack_mapping(self, rack_vals):
        sub_pos_rack_map = np.random.choice(np.array(rack_vals),
                                            (self.config.num_positions, 1),
                                            p=np.array(self.config.rack_pos_mapping_prob))
        num_prev_pos_assigned = self.generate_num_prev_assigned(sub_pos_rack_map)
        return sub_pos_rack_map, num_prev_pos_assigned

    def generate_num_prev_assigned(self, sub_pos_rack_map):
        pos_rack_map_flat = sub_pos_rack_map.flatten()
        counts_dict = {value: pos_rack_map_flat.tolist().count(value) for value in set(pos_rack_map_flat) if value != self.config.num_rack_types}
        num_prev_pos_assigned = np.zeros((self.config.num_rack_types, 1), dtype=int)
        for key, value in counts_dict.items():
            num_prev_pos_assigned[key][0] = value
        return num_prev_pos_assigned

    def encode_attributes_to_coords(self, batch_id, demand, num_prev_pos_assigned):
        x_coord = (self.config.rt_groups * self.config.shift + self.config.resource_int) / ((self.config.num_groups - 1) * self.config.shift + self.config.max_res_int)
        y_coord = (np.reshape(demand[batch_id], (self.config.num_rack_types, 1)) * self.config.num_positions + num_prev_pos_assigned) / ((self.config.demand_range[1] + 1) * self.config.num_positions)
        return x_coord, y_coord

    def generate_num_prev_assigned_v2(self, sub_pos_rack_map):
        # Assuming sub_pos_rack_map is already on the GPU
        num_rack_types = self.config.num_rack_types
        num_prev_pos_assigned = torch.zeros(num_rack_types, 1, device=sub_pos_rack_map.device, dtype=torch.int)

        # Calculate counts for each type except for `num_rack_types` which is assumed to be the "null" type
        for value in range(num_rack_types):
            if value != num_rack_types:  # Assuming the "null" type is equal to num_rack_types
                num_prev_pos_assigned[value] = (sub_pos_rack_map == value).sum()

        return num_prev_pos_assigned

    def encode_attributes_to_coords_v2(self, demand, num_prev_pos_assigned):
        num_rack_types = self.config.num_rack_types
        rt_groups = torch.tensor(self.config.rt_groups, device=demand.device, dtype=torch.float32)
        shift = torch.tensor(self.config.shift, device=demand.device, dtype=torch.float32)
        resource_int = torch.tensor(self.config.resource_int, device=demand.device, dtype=torch.float32)
        num_groups = torch.tensor(self.config.num_groups, device=demand.device, dtype=torch.float32)
        max_res_int = torch.tensor(self.config.max_res_int, device=demand.device, dtype=torch.float32)
        num_positions = torch.tensor(self.config.num_positions, device=demand.device, dtype=torch.float32)
        demand_range = torch.tensor(self.config.demand_range, device=demand.device, dtype=torch.float32)

        x_coord = (rt_groups * shift + resource_int) / ((num_groups - 1) * shift + max_res_int)
        y_coord = (demand.view(num_rack_types, 1) * num_positions + num_prev_pos_assigned) / ((demand_range[1] + 1) * num_positions)

        return x_coord, y_coord
