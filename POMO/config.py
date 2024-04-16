import numpy as onp
import random

class Config:
    # Number of objects
    seed = 0
    #key = random.PRNGKey(seed)
    num_positions = 100

    # For debug
    #num_positions = 20

    num_racks = 10
    num_groups = 4
    num_scopes = 5
    # Random generation weight
    scope_weights = [0.2]*num_scopes
    rack_group_weights = [0.25]*num_groups
    ## TODO-The following causes bugs. It should be solved soon.
    rack_pos_mapping_prob = [0.2/num_racks]*(num_racks+1)
    rack_pos_mapping_prob[-1] = 0.8 
    resource_weights = [1, 0.5, 0.25, 0.5, 0.75, 0.25, 0.25, 0.25, 0.75, 0.5]
    # Rack
    rt_groups = onp.random.choice(num_groups, (num_racks,1), p=rack_group_weights)
    # Position
    scopes = onp.random.choice(num_scopes, (num_positions,1), p=scope_weights)
    scopes = onp.array(scopes)
    # Resource table
    resource_table = []
    resource_int = []
    for rack_id in range(num_racks):
        rack_resource = []
        res_int = 0
        for res, prob in enumerate(resource_weights):
            rack_resource.append(1 if random.random()<prob else 0)
            res_int += rack_resource[-1]*2**res
        resource_table.append(rack_resource)
        resource_int.append(res_int)
    resource_int = onp.array(resource_int)
    resource_int = onp.reshape(resource_int, (num_racks, 1))
    resource_table = onp.array(resource_table)
    max_res_int = 2**len(resource_weights)-1
    shift = 10**len(str(max_res_int))
    # Demand and action limit range
    demand_range = [6, 10]

    # For debug
    #demand_range = [1, 2]
    action_limit_range = [80*num_racks,100*num_racks]
    # Dataframe column names
    position_cols = ['pos_id', 'status', 'scope']
    rack_type_cols = ['rt_id', 'group'] + [f'res_{i}' for i in range(len(resource_weights))] 
    # Constraints -- resource limit and spread metrics requirements
    L = [[30 for _ in range(10)] for _ in range(len(scope_weights))]
    L = onp.array(L)
    #spread_metrics = [({0,1}, {0, 2, 4, 6, 8}, 3), ({2,3,4}, {1, 3, 5, 7, 9}, 2)]
    spread_metrics = [(set(range(5)), set(range(10)), i) for i in range(len(resource_weights))]

    scope_rack_res_array = onp.zeros((num_scopes, len(resource_weights))) 
    scope_rack_res_spread_array = onp.zeros((num_scopes, num_groups, len(resource_weights))) 
