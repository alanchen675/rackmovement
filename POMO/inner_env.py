import random
import pandas as pd
import numpy as np
from config_complicated import Config
#from config import Config
from mip_solvers import Solver, SubSolver

class RackSystem:
    def __init__(self, args):
        self.inner_counter = 0
        self.counter = 0
        self.visited = set()
        self.sub_solver = SubSolver(args)

    def reset(self, seed=None, filepath=None):
        self.inner_counter = 0
        self.counter = 0
        self.visited = set()
        self.gen_demand()
        self.gen_action_limit()
        if not filepath:
            random.seed(seed)
            rack_arr, position_arr = [], []
            rack_arr = self.add_empty_rack_type()
            # Genrate rack df
            for rack_id in range(len(rack_arr), len(rack_arr)+Config.num_racks):
            #for rack_id in range(Config.num_racks):
                rack = [rack_id]
                rack.append(random.choices(list(range(len(Config.rack_group_weights))),\
                    weights=Config.rack_group_weights, k=1)[0])
                for res, prob in enumerate(Config.resource_weights):
                    rack.append(1 if random.random()<prob else 0)
                rack_arr.append(rack)
            self.rack_df = pd.DataFrame(np.array(rack_arr), columns=Config.rack_type_cols) 
            res_values_df = self.rack_df.loc[:, 'res_0':'res_9']
            # Encode the rack group and resource table to the x-axis of the rack type
            # Convert each row of 'res_1' to 'res_10' to integers by viewing them as bits
            self.rack_df['res_integers'] = res_values_df.apply(lambda row:\
                int(''.join(map(str, row)), 2), axis=1)
            self.rack_df['x_axis'] = self.rack_df['res_integers']
            base_res_weight = 2**len(Config.resource_weights)
            shift = 10**(len(str(base_res_weight)))
            self.rack_df['x_axis'] += self.rack_df['group']*shift
            self.rack_df['x_axis'] /= (shift*len(Config.rack_group_weights))
            # Generate position df
            for pos_id in range(Config.num_positions):
                position = [pos_id]
                position.append(random.choices(list(range(-1, Config.num_racks)),\
                    weights=Config.rack_pos_mapping_prob, k=1)[0])
                #position.append(random.choices(list(range(len(Config.scope_weights))),\
                #    weights=Config.scope_weights, k=1)[0])
                position.append(Config.scopes[pos_id])
                position_arr.append(position)
            self.pos_df = pd.DataFrame(np.array(position_arr), columns=Config.position_cols)
        else:
            self.rack_df = pd.read_csv(filepath) 
            self.pos_df = pd.read_csv(filepath)
        self.L = Config.L
        self.y_axis_encoder()
        self.gen_coordinate()
        self.new_action = [-1]*Config.num_positions
        self.old_rack_df = self.rack_df.copy()
        self.old_pos_df = self.pos_df.copy()
        self.movement_cost = 0
        self.scope_rack_res_dict = {}
        for sc in range(len(Config.scope_weights)):
            for rt_id in range(Config.num_racks):
                self.scope_rack_res_dict[(sc, rt_id)] = {f'res_{r}':0 for r\
                    in range(len(Config.resource_weights))}

    def add_empty_rack_type(self, activate=False):
        if not activate:
            return []
        else:
            rack = [0]
            rack.append(-1)
            for res, prob in enumerate(Config.resource_weights):
                rack.append(0)
            return [rack]

    def update(self):
        self.counter += 1
        self.visited = set()
        self.gen_demand()
        self.gen_action_limit()
        self.L = Config.L
        self.y_axis_encoder()
        self.new_action = [-1]*Config.num_positions
        self.gen_coordinate()
        self.old_rack_df = self.rack_df.copy()
        self.old_pos_df = self.pos_df.copy()
        self.movement_cost = 0
        self.scope_rack_res_dict = {}
        for sc in range(len(Config.scope_weights)):
            for rt_id in range(Config.num_racks):
                self.scope_rack_res_dict[(sc, rt_id)] = {f'res_{r}':0 for r\
                    in range(len(Config.resource_weights))}

    def y_axis_encoder(self):
        position_assigned = self.pos_df[self.pos_df['status']!=-1]
        st = position_assigned['status'].value_counts().to_dict()
        status_counts = {k:st[k] if k in st else 0 for k in range(Config.num_racks)}
        try:
            demand_arr = self.demand
        except AttributeError:
            self.gen_demand()
            self.gen_action_limit()
            demand_arr = self.demand
        self.y_axis = []
        for k in range(Config.num_racks):
            y = demand_arr[k]*10**(Config.num_positions)+status_counts[k]
            self.y_axis.append(y)

    def gen_demand(self):
        self.demand = []
        low, high = Config.demand_range
        for rack in range(Config.num_racks):
            self.demand.append(random.randint(low, high))

    def gen_action_limit(self):
        low, high = Config.action_limit_range
        self.action_limit = random.randint(low, high)

    def gen_coordinate(self):
        self.cord = [[a, b] for a, b in zip(self.rack_df['x_axis'].to_list(), self.y_axis)]

    def get_sub_state(self, rack_type):
        return self.demand[rack_type], self.action_limit, self.L, self.new_action,\
            self.scope_rack_res_dict, self.old_rack_df, self.old_pos_df

    def get_reward(self):
        rewards = 0
        merged_df = pd.merge(position_df, rack_df, left_on='status',\
            right_on='rt_id', how='left', suffixes=('_pos', '_rack'))
        determined_positions = merged_df[(merged_df['status']!=-1)]
        sum_by_scope = determined_positions.groupby('scope')[[f'res_{i}' for i in\
            range(len(Config.resource_weights))]].sum().reset_index()
        sum_by_scope_rack = determined_positions.groupby(['scope', 'rt_id'])[[f'res_{i}' for i in\
            range(len(Config.resource_weights))]].sum().reset_index()
        scope_res_dict = sum_by_scope.set_index('scope').to_dict(orient='index')
        scope_rack_res_dict = sum_by_scope_rack.set_index(['scope', 'rt_id']).to_dict(orient='index')
        for scopes, group, res_id in Config.spread_metrics:
            sub_reward = 0
            for sc in scopes:
                sub_reward = max(sub_reward, scope_rack_res_dict[sc][group][res_id])
            rewards += sub_reward
        return rewards

    def get_updated_constraints(self, rack_type, action, scope_rack_res_dict, action_limit, L):
        self.L = L
        self.action_limit = action_limit
        self.scope_rack_res_dict = scope_rack_res_dict
        for pos, t in action.items():
            if t==rack_type:
                self.new_action[pos] = t
            elif t==-1:
                if self.old_pos_df.loc[pos, 'status']==rack_type:
                    self.movement_cost -= 1
            else:
                print(f'The element in sub-action should be either -1 or {rack_type}')
                raise RuntimeError
            self.pos_df.loc[pos, 'status'] = t

    def _sub_step(self, rack_type, action, scope_rack_res_dict, action_limit, L):
        self.get_updated_constraints(rack_type, action, scope_rack_res_dict, action_limit, L)
        k = len(self.visited)
        self.visited.add(rack_type)
        if len(self.visited)==Config.num_racks:
            self.update()

    def sub_step(self, rack_type):
        demand, action_limit, L, mask, scope_rack_res_dict, rack_df, pos_df = self.get_sub_state(rack_type)
        status, obj, action, scope_rack_res_dict, action_limit,\
            L = self.sub_solver.solve(rack_df, pos_df, rack_type,\
                demand, action_limit, L, scope_rack_res_dict, mask)
        #assert status=="Pass"
        self._sub_step(rack_type, action, scope_rack_res_dict, action_limit, L)
        return status, obj, scope_rack_res_dict, action_limit, L

    def step_whole(self):
        status, obj, action, movement = self.sub_solver.solve_whole(self.old_rack_df,\
                self.old_pos_df, self.demand, self.action_limit, self.L)
        print(f'The number of movement is {movement}')
        print(f'The objective value is {obj}')
        if status!='Pass':
            print(f'The status is {status}')
        return status, movement, obj

def run():
    args = {}
    env = RackSystem(args)
    env.reset()
    # Get input (next node) from compass
    for rack_type in range(Config.num_racks):
        env.sub_step(rack_type)
    reward = env.get_reward()

def run_whole():
    args = {}
    env = RackSystem(args)
    movements = []
    objs = []
    for _ in range(100):
        env.reset()
        status, m, o = env.step_whole()
        if status=='Pass':
            movements.append(m)
            objs.append(o)
    movements = np.array(movements)
    objs = np.array(objs)
    print('=================================================')
    print(f'Number of passed optimization solving is {len(movements)}')
    print(f'Mean of the movements are {np.mean(movements)}')
    print(f'Std of the movements are {np.std(movements)}')
    print(f'Mean of the objectives are {np.mean(objs)}')
    print(f'Std of the objectives are {np.std(objs)}')

if __name__=='__main__':
    run_whole()
