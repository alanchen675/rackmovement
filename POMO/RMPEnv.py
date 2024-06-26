
from dataclasses import dataclass
import torch
import random

from RMProblemDef import get_random_problems, augment_xy_data_by_8_fold
from config_complicated import Config
#from config import Config


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class RMPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # The following attributes are fixed for all trajectory instances.
        self.num_positions = Config.num_positions
        self.num_rack_types = Config.num_racks
        self.num_groups = Config.num_groups
        self.num_scopes = Config.num_scopes
        self.num_level2_scopes = Config.num_level2_scopes
        self.num_level1_scopes = Config.num_level1_scopes
        self.num_resource_types = len(Config.resource_weights)

        self.scopes = torch.tensor(Config.scopes) # The array of scopes each position belongs to
        self.level23_map = torch.tensor(Config.level23_map)
        self.level13_map = torch.tensor(Config.level13_map)
        self.resource_table = torch.tensor(Config.resource_table)

        self.res_limit = torch.tensor(Config.L) 
        # shape: (scope, resource_type)
        self.level2_res_limit = torch.tensor(Config.level2_L) 
        # shape: (scope, resource_type)
        self.level1_res_limit = torch.tensor(Config.level1_L) 
        # shape: (scope, resource_type)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # RMP specific
        ####################################
        self.demand = None
        # shape: (batch, pomo)
        self.action_limit = None
        # shape: (batch, pomo)
        self.prev_pos_rack_map = None
        # shape: (batch, pomo, position)
        self.pos_rack_map = None
        # shape: (batch, pomo, position)
        self.scope_rack_res_array = None
        # shape: (batch, pomo, scope, resource_type)
        self.level2_scope_rack_res_array = None
        # shape: (batch, pomo, scope, resource_type)
        self.level1_scope_rack_res_array = None
        # shape: (batch, pomo, scope, resource_type)
        
    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.problems, self.prev_pos_rack_map, self.demand, self.action_limit\
                = get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 2)
        if len(self.res_limit.shape)==2:
            self.res_limit = torch.unsqueeze(self.res_limit, 0)  # Add a batch dimension
            self.res_limit = torch.unsqueeze(self.res_limit, 0)  # Add a pomo dimension
            self.res_limit = self.res_limit.repeat(self.batch_size, self.pomo_size, 1, 1)
        # shape: (batch, pomo, scope, resource_type)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        ##########Only the information sent to the encoder has to be in step state###########
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        #self.prev_pos_rack_map = self._init_prev_pos_rack_map()
        # shape: (batch, pomo, position)
        self.pos_rack_map = self.num_rack_types*torch.ones((self.batch_size, self.pomo_size, self.num_positions))
        self.pos_rack_map = self.pos_rack_map.long()
        # shape: (batch, pomo, position)
        self.scope_rack_res_array = torch.zeros((self.batch_size,\
                self.pomo_size, self.num_scopes, self.num_resource_types))
        # shape: (batch, pomo, scope, resource_type)
        self.level2_scope_rack_res_array = torch.zeros((self.batch_size,\
                self.pomo_size, self.num_level2_scopes, self.num_resource_types))
        # shape: (batch, pomo, scope, resource_type)
        self.level1_scope_rack_res_array = torch.zeros((self.batch_size,\
                self.pomo_size, self.num_level1_scopes, self.num_resource_types))
        # shape: (batch, pomo, scope, resource_type)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def _init_prev_pos_rack_map(self):
        total_elements = self.batch_size * self.pomo_size * self.num_positions
        # Determine the number of elements to set to -1 and to sample from 0 to 4
        num_minus_one = int(0.8 * total_elements)
        num_sampled = total_elements - num_minus_one
        # Generate a Python list with -1 and sampled values from 0 to 4
        values = [-1] * num_minus_one + random.choices(range(self.num_rack_types), k=num_sampled)
        # Shuffle the list to ensure randomness
        random.shuffle(values)
        # Convert the list to a PyTorch tensor
        init_pos_rack_map = torch.tensor(values).reshape(self.batch_size, self.pomo_size, self.num_positions)
        return init_pos_rack_map.int()

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)
        # Shape of the selected_node_list grows at every enter to the step function 

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        self.eq_heuristics(selected)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self.get_reward()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_reward(self):
        # Move the second dimension to the third dimension
        reshaped_prev_map = self.prev_pos_rack_map.permute(0,2,1)
        # Expand A to match the shape of B for broadcasting
        reshaped_prev_map = reshaped_prev_map.expand(-1, self.pos_rack_map.shape[1], -1)
        mask = (reshaped_prev_map.squeeze(1)!=self.num_rack_types)&(reshaped_prev_map.squeeze(1)!=self.pos_rack_map)
        mask = mask.float()
        reward = torch.sum(mask, dim=2)
        reward += torch.sum(self.action_limit<0)*100
        max_res_per_scope = torch.max(self.scope_rack_res_array, dim=2)[0]
        reward += torch.sum(max_res_per_scope, dim=2)
        max_res_per_level2_scope = torch.max(self.level2_scope_rack_res_array, dim=2)[0]
        reward += torch.sum(max_res_per_level2_scope, dim=2)
        max_res_per_level1_scope = torch.max(self.level1_scope_rack_res_array, dim=2)[0]
        reward += torch.sum(max_res_per_level1_scope, dim=2)
        return reward 

    def eq_heuristics(self, action):
        """
        A heuristic approach that evenly allocates rack types to the positions

        Args:
            action: int, index of the next node to visit.

        Returns:
            action_limit: the remaining allowed placement actions 
            res_limit: the total allowed amount of resource for a scope and a rack group
            pos_rack_map: the new position to rack type mapping
            scope_rack_res_array: the remaining allowed amount of resource for a scope and a rack group

        """
        diff = self.res_limit-self.scope_rack_res_array
        weight, _ = torch.min(diff, dim=-1)
        sum_weight = torch.sum(weight, dim=-1, keepdim=True)
        weight = weight/sum_weight
        #demand_selected = self.demand[:, action].unsqueeze(1).unsqueeze(2)
        _demand = self.demand.unsqueeze(-1)
        _action = action.unsqueeze(-1)
        demand_selected = _demand.gather(dim=1, index=_action)
        num_pos = weight*demand_selected
        num_pos = torch.ceil(num_pos-1e-6) 
        remaining_demand = demand_selected.squeeze(-1)-torch.sum(num_pos, dim=2, keepdim=False)

        for batch_id in range(num_pos.shape[0]):
            for pomo_id in range(num_pos.shape[1]):
                while remaining_demand[batch_id, pomo_id] < 0:
                    # Find the maximum value along the third dimension
                    max_value, max_index = torch.max(num_pos[batch_id, pomo_id], dim=-1)
                    # Subtract 1 from the maximum value
                    num_pos[batch_id, pomo_id, max_index] -= 1
                    # Update remaining demand
                    remaining_demand[batch_id, pomo_id] += 1

                for scope, npos in enumerate(num_pos[batch_id, pomo_id]):
                    # Compute the mask for the current scope
                    scope_mask = (self.scopes == scope)
                    # Get the mask for positions already assgined to action[batch_id, pomo_id]
                    assigned_mask = (self.prev_pos_rack_map[batch_id] == action[batch_id, pomo_id])
                    # Find positions that belong to the current rack type
                    mask = scope_mask & assigned_mask 
                    mask = mask.squeeze(-1)
                    num_positions = mask.sum()

                    # Adjust positions based on positions_to_assign
                    # npos: the number of positions supposed to be allocated
                    # num_positions: the number of positions already allocated
                    if num_positions > npos:
                        # Too many positions assigned, set excess positions to self.num_rack_types 
                        excess_positions = num_positions - npos
                        excess_positions = excess_positions.int()
                        excess_mask = mask.nonzero()[:excess_positions]
                        keep_mask = mask.nonzero()[:npos.int()]
                        self.pos_rack_map[batch_id, pomo_id, keep_mask] = action[batch_id, pomo_id]
                        #self.pos_rack_map[batch_id, pomo_id, excess_mask] = self.num_rack_types 
                        resource_usage = npos*self.resource_table[action[batch_id, pomo_id]]
                    elif num_positions < npos:
                        # Not enough positions assigned, assign more positions to current_rack_type
                        remaining_positions = npos - num_positions
                        remaining_positions = remaining_positions.int()
                        remaining_mask = (self.prev_pos_rack_map[batch_id] == self.num_rack_types)
                        # shape: (position, 1)
                        remaining_mask = scope_mask&remaining_mask 
                        remaining_mask = remaining_mask.squeeze(-1)
                        # shape: position
                        remaining_positions = min(remaining_positions, remaining_mask.sum())
                        total_assigned_positions = num_positions+remaining_positions
                        ## TODO-It is possible that the assigned positions will be less than the demand.
                        remaining_mask = remaining_mask.nonzero()[:remaining_positions]
                        mask = mask.nonzero()[:]
                        self.pos_rack_map[batch_id, pomo_id, remaining_mask] = action[batch_id, pomo_id]
                        self.pos_rack_map[batch_id, pomo_id, mask] = action[batch_id, pomo_id]
                        ## TODO-The assigned positions in previous mapping should be kept
                        self.action_limit[batch_id, pomo_id] -= total_assigned_positions.int() 
                        resource_usage = total_assigned_positions*self.resource_table[action[batch_id, pomo_id]]
                    else:
                        mask = mask.nonzero()[:]
                        self.pos_rack_map[batch_id, pomo_id, mask] = action[batch_id, pomo_id]
                        resource_usage = npos.int()*self.resource_table[action[batch_id, pomo_id]]

                    ## Update self.scope_rack_res_array
                    self.scope_rack_res_array[batch_id, pomo_id, scope] += resource_usage
                    level2_scope, level1_scope = self.level23_map[scope], self.level13_map[scope]
                    self.level2_scope_rack_res_array[batch_id, pomo_id, level2_scope] += resource_usage
                    self.level1_scope_rack_res_array[batch_id, pomo_id, level1_scope] += resource_usage
