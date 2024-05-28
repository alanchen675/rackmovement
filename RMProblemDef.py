import torch
import numpy as np
from config_complicated import Config
#from config import Config


def get_random_problems(batch_size, problem_size, time_steps=5):
    num_positions = Config.num_positions
    num_racks = Config.num_racks
    rack_group = Config.rt_groups
    num_group = Config.num_groups
    resource_int = Config.resource_int
    max_res_int = Config.max_res_int
    shift = Config.shift

    ## Generate demand
    dm_low, dm_high = Config.demand_range
    # Generate random integers within the range [a, b]
    demand = np.random.randint(dm_low, dm_high + 1, (batch_size, num_racks))
    rest_demand = np.random.randint(dm_low, dm_high+1, (time_steps-1, batch_size, num_racks))
    demand_pool = np.concatenate([demand[None,:,:], rest_demand], axis=0)

    ## Generate action limit
    act_low, act_high = Config.action_limit_range
    action_limit = np.random.randint(act_low, act_high+1, (batch_size, 1))
    rest_action_limit = np.random.randint(act_low, act_high+1, (time_steps-1, batch_size, 1))
    action_limit_pool = np.concatenate([action_limit[None,:,:], rest_action_limit], axis=0)
    action_limit = np.tile(action_limit, (1, num_racks))

    rack_vals = list(range(num_racks+1))
    problem = []
    pos_rack_map = []
    for batch_id in range(batch_size):
        ## Position rack mapping
        sub_pos_rack_map = np.random.choice(np.array(rack_vals),\
                (num_positions, 1), p=np.array(Config.rack_pos_mapping_prob))
        # Flatten the pos_rack_map to a 1D array
        pos_rack_map_flat = sub_pos_rack_map.flatten()
        # Initialize a dictionary to count occurrences
        counts_dict = {}
        # Count occurrences of each value
        for value in pos_rack_map_flat:
            if value==num_racks:
                continue
            if value in counts_dict:
                counts_dict[value] += 1
            else:
                counts_dict[value] = 1
        # Convert counts dictionary to a NumPy array
        num_prev_pos_assigned = np.zeros((num_racks, 1), dtype=int)
        for key, value in counts_dict.items():
            num_prev_pos_assigned[key][0] = value

        ## Encode attributes to coordinates
        x_coord = (rack_group*shift+resource_int)/((num_group-1)*shift+max_res_int)
        y_coord = (np.reshape(demand[batch_id], (num_racks,1))*num_positions+\
                num_prev_pos_assigned)/((dm_high+1)*num_positions)
        
        sub_problem = np.hstack((x_coord, y_coord))
        problem.append(sub_problem)
        pos_rack_map.append(sub_pos_rack_map)
    problem = np.array(problem)
    problem = torch.from_numpy(problem).to('cuda')

    pos_rack_map = np.array(pos_rack_map)
    pos_rack_map = torch.from_numpy(pos_rack_map).to('cuda')

    # problem = torch.tensor(problem)
    # pos_rack_map = torch.tensor(pos_rack_map)
    demand = torch.tensor(demand)
    action_limit = torch.tensor(action_limit)
    problem = problem.float()
    return problem, pos_rack_map, demand, action_limit, demand_pool, action_limit_pool

def generate_coord(batch_size, demand):
    num_positions = Config.num_positions
    num_racks = Config.num_racks
    rack_group = Config.rt_groups
    num_group = Config.num_groups
    resource_int = Config.resource_int
    max_res_int = Config.max_res_int
    shift = Config.shift
    _, dm_high = Config.demand_range

    rack_vals = list(range(num_racks+1))
    problem = []
    pos_rack_map = []
    for batch_id in range(batch_size):
        ## Position rack mapping
        sub_pos_rack_map = np.random.choice(np.array(rack_vals),\
                (num_positions, 1), p=np.array(Config.rack_pos_mapping_prob))
        # Flatten the pos_rack_map to a 1D array
        pos_rack_map_flat = sub_pos_rack_map.flatten()
        # Initialize a dictionary to count occurrences
        counts_dict = {}
        # Count occurrences of each value
        for value in pos_rack_map_flat:
            if value==num_racks:
                continue
            if value in counts_dict:
                counts_dict[value] += 1
            else:
                counts_dict[value] = 1
        # Convert counts dictionary to a NumPy array
        num_prev_pos_assigned = np.zeros((num_racks, 1), dtype=int)
        for key, value in counts_dict.items():
            num_prev_pos_assigned[key][0] = value

        ## Encode attributes to coordinates
        x_coord = (rack_group*shift+resource_int)/((num_group-1)*shift+max_res_int)
        y_coord = (np.reshape(demand[batch_id], (num_racks,1))*num_positions+\
                num_prev_pos_assigned)/((dm_high+1)*num_positions)
        
        sub_problem = np.hstack((x_coord, y_coord))
        problem.append(sub_problem)
        pos_rack_map.append(sub_pos_rack_map)
    problem = np.array(problem)
    problem = torch.from_numpy(problem).to('cuda')
    # problem = torch.tensor(problem)

    pos_rack_map = np.array(pos_rack_map)
    pos_rack_map = torch.from_numpy(pos_rack_map).to('cuda')


    problem = problem.float()
    return problem

def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems
