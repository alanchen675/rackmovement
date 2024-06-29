
from dataclasses import dataclass
import torch
import random
import numpy as np
# import multiprocessing as mp
import torch.multiprocessing as mp
import torch.nn.functional as F

from RMProblemDef import augment_xy_data_by_8_fold, generate_coord
from RMProblemDef_clean import ProblemGenerator
from config_clean import config
from logging import getLogger
#from config import config


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
        self.logger = getLogger(name='env')

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.periods = env_params['periods']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)
        self.init_rmp()
        self.generator = ProblemGenerator(config)

    def init_rmp(self):
        # The following attributes are fixed and static for all trajectory instances.
        # ========================================
        self.num_positions = config.num_positions
        self.num_rack_types = config.num_rack_types
        self.num_groups = config.num_groups
        self.num_scopes = config.num_scopes
        self.num_level2_scopes = config.num_level2_scopes
        self.num_level1_scopes = config.num_level1_scopes
        self.num_resource_types = len(config.resource_weights)

        self.scopes = torch.tensor(config.scopes) # The array of scopes each position belongs to
        self.scopes_comp = self.one_hot_encode(self.scopes, self.num_scopes).t().float()
        self.level23_map = torch.tensor(config.level23_map)
        self.scopes23_comp = self.one_hot_encode(self.level23_map, self.num_level2_scopes).t().float()
        self.level13_map = torch.tensor(config.level13_map)
        self.scopes13_comp = self.one_hot_encode(self.level13_map, self.num_level1_scopes).t().float()
        self.resource_table = torch.tensor(config.resource_table).float()

        self.res_limit = torch.tensor(config.L).float()
        # shape: (scope, resource_type)
        self.level2_res_limit = torch.tensor(config.level2_L)
        # shape: (scope, resource_type)
        self.level1_res_limit = torch.tensor(config.level1_L)
        # shape: (scope, resource_type)

        # Dynamic
        # reset every num_rack_types running of the step function
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # RMP specific
        # update every num_rack_types running of the step function
        # Generate random demands in the reset function
        self.demand = None
        # shape: (batch, pomo)
        self.action_limit = None
        # shape: (batch, pomo)
        self.prev_pos_rack_map = None
        # shape: (batch, pomo, position)
        self.pos_rack_map = None
        # shape: (batch, pomo, position)
        # Reset as how it was updated in the reset function
        self.scope_rack_res_array = None
        # shape: (batch, pomo, scope, resource_type)
        self.level2_scope_rack_res_array = None
        # shape: (batch, pomo, scope, resource_type)
        self.level1_scope_rack_res_array = None
        # shape: (batch, pomo, scope, resource_type)

    def one_hot_encode(self, X_prime, num_columns, grad=False):
        """
        Function to one-hot encode X'
        """
        # Create a tensor of zeros with an extra dimension for the one-hot encoding
        shape = (*X_prime.shape, num_columns)
        # X_one_hot = torch.zeros(shape, dtype=torch.float32)
        X_one_hot = torch.zeros(shape, dtype=torch.float32, device=X_prime.device)

        # Use scatter to set 1s in the appropriate places, except where X_prime is num_columns 
        indices = torch.clamp(X_prime, 0, num_columns-1)
        mask = X_prime < num_columns
        X_one_hot.scatter_(-1, indices.unsqueeze(-1), mask.unsqueeze(-1).float())

        # Set the slices where X_prime is num_columns to all zeros
        X_one_hot[X_prime == num_columns] = 0
        if grad:
            X_one_hot.requires_grad_(True)
        return X_one_hot

    # def create_one_hot(X_prime, num_classes):
    #     """
    #     Create a one-hot encoded tensor based on the indices specified in X_prime using scatter_ method.
    #     Args:
    #         X_prime (torch.Tensor): Tensor containing indices of categories.
    #         num_classes (int): The number of categories.
    #     Returns:
    #         torch.Tensor: A tensor with one-hot encoding based on X_prime with gradients enabled.
    #     """
    #     # Create an output tensor of all zeros
    #     X = torch.zeros(X_prime.size(0), num_classes)

    #     # Use scatter_ to set values without in-place modification problems
    #     X.scatter_(1, X_prime.unsqueeze(1), 1)

    #     # Now, after setting up the tensor, we can enable gradient tracking
    #     X.requires_grad_(True)

    #     return X

    def one_hot_to_categorical(self, one_hot_tensor):
        """
        Converts a one-hot encoded tensor back to its categorical format by returning the index
        of the maximum value in each vector along the specified dimension.

        Args:
        - one_hot_tensor (torch.Tensor): The one-hot encoded tensor.

        Returns:
        - torch.Tensor: A tensor containing the indices of the maximum values (categories).
        """
        # Compute the argmax
        argmax_tensor = torch.argmax(one_hot_tensor, dim=-1)
        # Check for rows that sum to 0 (i.e., all-zero vectors)
        # If the sum along the last dimension is 0, it's an all-zero vector
        is_all_zero = torch.sum(one_hot_tensor, dim=-1) == 0

        # Where it's all-zero, set the output to R
        categorical_tensor = torch.where(is_all_zero, torch.full_like(argmax_tensor, self.pomo_size), argmax_tensor)

        return categorical_tensor

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.problems, self.prev_pos_rack_map, self.demand, self.action_limit, self.demand_pool,\
                self.action_limit_pool = self.generator.get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 2)
        # self.prev_pos_rack_map from ProblemGenerator is of shape (batch_size, num_positions)
        # So, batch_size instances of prev_pos_rack_map is generated.
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

    def step_baseline(self, period, pos_rack_map):
        """
        The step function for the baselines which solve the whole optimization problem
        """
        self.demand, self.action_limit = np.array(self.demand_pool[period]), np.array(self.action_limit_pool[period])
        self.prev_pos_rack_map = torch.tensor(pos_rack_map)
        self.demand = torch.tensor(self.demand)
        self.action_limit = torch.tensor(self.action_limit)
        return self.prev_pos_rack_map, self.demand, self.action_limit

    def middle_reset(self, period):
        # Regenerate the coordinates, demand, action_limit, and prev_pos_rack_map
        self.demand, self.action_limit = np.array(self.demand_pool[period]), np.array(self.action_limit_pool[period])

        # For multiprocessing, need to check the scope resource arrays
        self.prev_pos_rack_map = torch.tensor(self.pos_rack_map)
        self.demand = torch.tensor(self.demand)
        self.action_limit = torch.tensor(self.action_limit)
        self.problem = self.generator.generate_coord(self.batch_size, self.prev_pos_rack_map, self.demand)
        # Reset
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
        self.pos_rack_map = self.one_hot_encode(self.prev_pos_rack_map, self.num_rack_types, True)
        reward = None
        done = False
        return Reset_State(self.problems), reward, done

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

        # The following operation makes it be of shape (batch_size, pomo_size, num_positions) and
        # The prev_pos_rack_map instance for each batch is the same for different pomo
        if self.prev_pos_rack_map.shape[2]==1:
            duplicated_prev_pos_rack_map = self.prev_pos_rack_map.repeat(1, 1, self.pomo_size).permute(0, 2, 1)
            self.pos_rack_map = self.one_hot_encode(duplicated_prev_pos_rack_map, self.num_rack_types, True)
        else:
            self.pos_rack_map = self.one_hot_encode(self.prev_pos_rack_map, self.num_rack_types, True)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

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

        #==========RMP==========
        ## Decide whether parallelization is needed.
        #self.eq_heuristics_parallel(selected)
        # self.eq_heuristics(selected)
        self.comp_heuristics(selected)
        #self.comp_heuristics_parallel(selected)
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self.get_reward()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def obj(self, X):
        SXR = torch.einsum('ij,abjk,kc->abic', self.scopes_comp, X, self.resource_table)
        # Shape = (batch_size, pomo_size, num_scopes, num_resource_types)
        S2XR = torch.einsum('ij,abjc->abic', self.scopes23_comp, SXR)
        # Shape = (batch_size, pomo_size, num_level2_scopes, num_resource_types)
        S1XR = torch.einsum('ij,abjc->abic', self.scopes13_comp, SXR)
        # Shape = (batch_size, pomo_size, num_level1_scopes, num_resource_types)
        # Computing the terms with softplus and sum reductions
        SXRmin = torch.minimum(torch.zeros_like(self.res_limit, device='cuda'), self.res_limit - SXR)
        #self.logger.info(f'The shape of tensor after min is {SXRmin.shape}')
        first_term = F.softplus(SXRmin, beta=1).sum(dim=(2, 3))
        third_term = F.softplus(torch.minimum(torch.zeros_like(self.level2_res_limit, device='cuda'), self.level2_res_limit.unsqueeze(0).unsqueeze(0) - S2XR), beta=1).sum(dim=(2, 3))
        forth_term = F.softplus(torch.minimum(torch.zeros_like(self.level1_res_limit, device='cuda'), self.level1_res_limit.unsqueeze(0).unsqueeze(0) - S1XR), beta=1).sum(dim=(2, 3))
        second_term = 1e2 * torch.std(SXR, dim=2).sum(dim=-1)
        return first_term + second_term + third_term + forth_term

    def get_reward(self):
        reward = self.obj(self.pos_rack_map)
        self.pos_rack_map = self.one_hot_to_categorical(self.pos_rack_map)
        # Move the second dimension to the third dimension
        if self.prev_pos_rack_map.shape[2]==1:
            reshaped_prev_map = self.prev_pos_rack_map.permute(0,2,1)
            # Expand A to match the shape of B for broadcasting
            reshaped_prev_map = reshaped_prev_map.expand(-1, self.pos_rack_map.shape[1], -1)
            mask = (reshaped_prev_map.squeeze(1)!=self.num_rack_types)&(reshaped_prev_map.squeeze(1)!=self.pos_rack_map)
        else:
            # reshaped_prev_map = self.prev_pos_rack_map
            mask = (self.prev_pos_rack_map.squeeze(1)!=self.num_rack_types)&\
                (self.prev_pos_rack_map.squeeze(1)!=self.pos_rack_map)
        mask = mask.float()
        reward += torch.sum(mask, dim=2)
        # Check whether this is a bug. The action_limit violation should contribute individually to each batch.
        reward += torch.sum(self.action_limit<0)*100
        return reward

    def comp_heuristics_parallel(self, action):
        num_gpus = torch.cuda.device_count()
        batch_size = self.pos_rack_map.size(0)
        split_size = (batch_size + num_gpus - 1) // num_gpus
        X_splits = torch.split(self.pos_rack_map, split_size)
        action_splits = torch.split(action, split_size)
        demand_splits = torch.split(self.demand, split_size)
        outputs = []
        for i, X_chunk in enumerate(X_splits):
            X_gpu = X_chunk.to(f'cuda:{i}')
            action_gpu = action_splits[i].to(f'cuda:{i}')
            demand_gpu = demand_splits[i].to(f'cuda:{i}')
            output_gpu = self._comp_heuristics(X_gpu, action_gpu, demand_gpu)
            output_gpu.to('cuda:0')

    def _comp_heuristics(self, pr_map, action, demand):
        for batch_id in range(pr_map.shape[0]):
            for pomo_id in range(pr_map.shape[1]):
                rack_type = action[batch_id][pomo_id]
                self.comp_sequential_put(pr_map[batch_id][pomo_id],\
                    rack_type, demand[batch_id][pomo_id])
        return pr_map

    def comp_heuristics(self, action):
        """
        A heuristic approach developed by META that utilizes computational graph
        approach to allocate on rack at a time.
        """
        # print(f'action is {action} at iteration {self.selected_count}')
        for batch_id in range(self.pos_rack_map.shape[0]):
            for pomo_id in range(self.pos_rack_map.shape[1]):
                rack_type = action[batch_id, pomo_id]
                self.comp_sequential_put(self.pos_rack_map[batch_id, pomo_id],\
                    rack_type, self.demand[batch_id, rack_type])

        #self.logger.info(f'Number of racks allocated for each type {self.pos_rack_map.sum(dim=2)}')

    def f(self, X):
        SXR = self.scopes_comp @ X @ self.resource_table
        # Shape = (num_scopes, num_resource_types)
        S2XR = self.scopes23_comp @ SXR
        # Shape = (num_level2_scopes, num_resource_types)
        S1XR = self.scopes13_comp @ SXR
        # Shape = (num_level1_scopes, num_resource_types)

        sub_scope_rack_res = self.res_limit.squeeze(0).squeeze(0)
        first_term = -F.softplus(torch.min(torch.zeros_like(sub_scope_rack_res), sub_scope_rack_res - SXR), beta=1).sum()
        third_term = -F.softplus(torch.min(torch.zeros_like(self.level2_res_limit), self.level2_res_limit - S2XR), beta=1).sum()
        forth_term = -F.softplus(torch.min(torch.zeros_like(self.level1_res_limit), self.level1_res_limit - S1XR), beta=1).sum()

        second_term = -1e2 * torch.std(SXR, dim=0).sum()

        return first_term + second_term + third_term + forth_term
        # return first_term + second_term

    def comp_sequential_put(self, X, rack_type, demand):
        """
        Given the current position and rack type mapping, allocate rack_type to fulfill
        the demand.
        """
        X.retain_grad()
        # Calculate the current allocation based on the demand and what's already allocated
        current_allocation = X[:, rack_type].sum()
        allocation = demand - current_allocation
        # Calculate the number of steps required based on the absolute value of allocation
        steps = int(abs(allocation.item()))
        v = torch.ones(self.num_positions, 1)                        # Column vector of ones
        for i in range(steps):
            Xw = X.sum(dim=1, keepdim=True)
            output = self.f(X)
            output.backward()
            if allocation > 0:
                # Positive demand: Increase X where Xw is 0
                valid_indices = (Xw[:,0] == 0)
                # Ensure that there's at least one valid index to avoid empty selection
                if valid_indices.any():
                    # Apply mask to gradients and find the index with the largest gradient in column i
                    masked_grads = X.grad[:, rack_type].masked_fill(~valid_indices, float('-inf'))
                    selected_pos = torch.argmax(masked_grads)
                    X.data[selected_pos, rack_type] += 1  # Update X at the selected index
                else:
                    print("No valid index with Xw[j] == 0 found.")
            elif allocation < 0:
                # Negative demand: Decrease X where X[:, rack_type] is not 0
                non_zero_indices = (X[:, rack_type] > 0)
                if non_zero_indices.any():
                    masked_grads = X.grad[:, rack_type].masked_fill(~non_zero_indices, float('inf'))
                    selected_pos = torch.argmin(masked_grads)
                    X.data[selected_pos, rack_type] = 0  # Set X at the selected position to 0
                else:
                    print("No valid index with X[:, rack_type] != 0 found.")

            # Zero the gradients after the update
            X.grad.zero_()

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
        print(f'[heuristic][first line]Shape of new action_limit is {self.action_limit.shape}')
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
        print(f'[heuristic][before for loop]Shape of new action_limit is {self.action_limit.shape}')

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

class RMPEnvCPU(RMPEnv):
    def __init__(self, **env_params):
        super().__init__(env_params)
        # Use fork to start the process
        mp.set_start_method('fork')
        # For CUDA
        #mp.set_start_method('spawn')
        self.scopes = torch.tensor(config.scopes, device='cpu') # The array of scopes each position belongs to
        self.level23_map = torch.tensor(config.level23_map, device='cpu')
        self.level13_map = torch.tensor(config.level13_map, device='cpu')
        self.resource_table = torch.tensor(config.resource_table, device='cpu')

        self.res_limit = torch.tensor(config.L, device='cpu')
        # shape: (scope, resource_type)
        self.level2_res_limit = torch.tensor(config.level2_L, device='cpu')
        # shape: (scope, resource_type)
        self.level1_res_limit = torch.tensor(config.level1_L, device='cpu')
        # shape: (scope, resource_type)
    def load_problems(self, batch_size, aug_factor=1):
        super().load_problem(batch_size, aug_factor)
        self.prev_pos_rack_map = self.prev_pos_rack_map.to('cpu')
        self.demand = self.demand.to('cpu')
        self.action_limit = self.action_limit.to('cpu')
        self.res_limit = self.res_limit.to('cpu')

    def middle_reset(self, period):
        super().middle_reset(period)
        self.demand = torch.tensor(self.demand, device='cpu')
        self.action_limit = torch.tensor(self.action_limit, device='cpu')

    def reset(self):
        super().reset()
        self.pos_rack_map = self.pos_rack_map.to('cpu')
        self.scope_rack_res_array = self.scope_rack_res_array.to('cpu')
        self.level2_scope_rack_res_array = self.level2_scope_rack_res_array.to('cpu')
        self.level1_scope_rack_res_array = self.level1_scope_rack_res_array.to('cpu')
        # For multiprocessing
        self.pos_rack_map.share_memory_()
        self.scope_rack_res_array.share_memory_()
        self.level2_scope_rack_res_array.share_memory_()
        self.level1_scope_rack_res_array.share_memory_()

    def step(self, selected):
        super.step(selected)
        self.eq_heuristics_parallel(selected.to('cpu'))

    def eq_heuristics_parallel(self, action):
        """
        An experiment for parallelizing eq_heuristic()
        """
        #action = action.to('cpu')
        diff = self.res_limit-self.scope_rack_res_array
        # shape: (batch, pomo, scope, resource_type)

        weight, _ = torch.min(diff, dim=-1)
        # shape: (batch, pomo, scope)

        sum_weight = torch.sum(weight, dim=-1, keepdim=True)
        # shape: (batch, pomo, scope)

        weight = weight/sum_weight
        # shape: (batch, pomo, scope)

        _demand = self.demand.unsqueeze(-1)
        # shape: (batch, pomo, 1)

        _action = action.unsqueeze(-1)
        # shape: (batch, pomo, 1)

        demand_selected = _demand.gather(dim=1, index=_action)
        # shape: (batch, pomo, 1)
        # Get the number of racks (float) to get from each scope

        num_pos = weight*demand_selected
        # shape: (batch, pomo, scope)
        # To turn the number of racks to integers
        num_pos = torch.ceil(num_pos-1e-6)
        # shape: (batch, pomo, 1)
        # To get how many racks to get to fulfil the demand given the num_pos

        remaining_demand = demand_selected.squeeze(-1)-torch.sum(num_pos, dim=2, keepdim=False)
        # shape: (batch, pomo)
        # remaining_demand = actual demand - initial allocation

        # Show the device for the tensor
        self.logger.info("Before multiprocessing")
        self.logger.info(f"The device for self.res_limit is {self.res_limit.device}")
        self.logger.info(f"The device for self.scope_rack_res_array is {self.scope_rack_res_array.device}")
        self.logger.info(f"The device for self.demand is {self.demand.device}")
        self.logger.info(f"The device for self.prev_pos_rack_map is {self.prev_pos_rack_map.device}")
        self.logger.info(f"The device for self.pos_rack_map is {self.pos_rack_map.device}")
        self.logger.info(f"The device for self.level2_scope_rack_res_array is {self.level2_scope_rack_res_array.device}")
        self.logger.info(f"The device for self.level1_scope_rack_res_array is {self.level1_scope_rack_res_array.device}")

        processes = []
        for i in range(num_pos.shape[0]):
            for j in range(num_pos.shape[1]):
                #print(f"Create subprocess for batch {i} and pomo {j}")
                p = mp.Process(target=self._heuristics_parallel, args=(i,j,remaining_demand[i,j],\
                    num_pos[i,j],action[i,j]))
                p.start()
                processes.append(p)
                #print(f"Append subprocess for batch {i} and pomo {j}")

        for p in processes:
            #print("Join subprocesses")
            p.join()

        #action = action.to('cuda')

    def _heuristics_parallel(self, batch_id, pomo_id, remaining_demand, num_pos, action):
        """
        Worker function for the parallelization implementation
        """
        self.logger.info(f"[Subprocess]The subprocess for batch id {batch_id} and pomo id {pomo_id} starts")
        while remaining_demand < 0:
            # Find the maximum value along the third dimension
            max_value, max_index = torch.max(num_pos, dim=-1)
            # Subtract 1 from the maximum value
            num_pos[max_index] -= 1
            # Update remaining demand
            remaining_demand += 1

        #print(f"[Subprocess]The device for self.prev_pos_rack_map is {self.prev_pos_rack_map.device}")
        #print(f"[Subprocess]The device for self.pos_rack_map is {self.pos_rack_map.device}")
        #print(f"[Subprocess]The device for self.level2_scope_rack_res_array is {self.level2_scope_rack_res_array.device}")
        #print(f"[Subprocess]The device for self.level1_scope_rack_res_array is {self.level1_scope_rack_res_array.device}")

        for scope, npos in enumerate(num_pos):
            # Compute the mask for the current scope
            scope_mask = (self.scopes == scope)
            # Get the mask for positions already assgined to action
            assigned_mask = (self.prev_pos_rack_map[batch_id] == action)
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
                self.pos_rack_map[batch_id, pomo_id, keep_mask] = action
                #self.pos_rack_map[batch_id, pomo_id, excess_mask] = self.num_rack_types 
                resource_usage = npos*self.resource_table[action]
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
                self.pos_rack_map[batch_id, pomo_id, remaining_mask] = action
                self.pos_rack_map[batch_id, pomo_id, mask] = action
                ## TODO-The assigned positions in previous mapping should be kept
                self.action_limit[batch_id, pomo_id] -= total_assigned_positions.int() 
                resource_usage = total_assigned_positions*self.resource_table[action]
            else:
                mask = mask.nonzero()[:]
                self.pos_rack_map[batch_id, pomo_id, mask] = action
                resource_usage = npos.int()*self.resource_table[action]

            ## Update self.scope_rack_res_array
            self.scope_rack_res_array[batch_id, pomo_id, scope] += resource_usage
            level2_scope, level1_scope = self.level23_map[scope], self.level13_map[scope]
            self.level2_scope_rack_res_array[batch_id, pomo_id, level2_scope] += resource_usage
            self.level1_scope_rack_res_array[batch_id, pomo_id, level1_scope] += resource_usage
        self.logger.info(f"[Subprocess]The subprocess for batch id {batch_id} and pomo id {pomo_id} ends")

    def get_reward(self):
            # Move the second dimension to the third dimension
            reshaped_prev_map = self.prev_pos_rack_map.permute(0,2,1)
            # Expand A to match the shape of B for broadcasting
            reshaped_prev_map = reshaped_prev_map.expand(-1, self.pos_rack_map.shape[1], -1)
            mask = (reshaped_prev_map.squeeze(1)!=self.num_rack_types)&(reshaped_prev_map.squeeze(1)!=self.pos_rack_map)
            mask = mask.float()
            reward = torch.sum(mask, dim=2)
            reward += torch.sum(self.action_limit<0)*100
            # Assume that the shared tensors can still be used to calculate reward
            max_res_per_scope = torch.max(self.scope_rack_res_array, dim=2)[0]
            reward += torch.sum(max_res_per_scope, dim=2)
            max_res_per_level2_scope = torch.max(self.level2_scope_rack_res_array, dim=2)[0]
            reward += torch.sum(max_res_per_level2_scope, dim=2)
            max_res_per_level1_scope = torch.max(self.level1_scope_rack_res_array, dim=2)[0]
            reward += torch.sum(max_res_per_level1_scope, dim=2)
            reward = reward.to('cuda')
            return reward
