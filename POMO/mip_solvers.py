import pandas as pd
import numpy as np
from config_complicated import Config
#from config import Config
from mip import Model, xsum, minimize, BINARY, INTEGER, OptimizationStatus

class SubSolver:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

    def solve_whole(self, rack_df, position_df, demand, action_limit, L):
        scope_positions_dict = position_df.groupby('scope')['pos_id'].agg(list).to_dict()

        m = Model("Subsolver")
        m.verbose = 0
        pos_indices = position_df.index.tolist()
        rack_indices = rack_df.index.tolist()
        scope_indices = list(range(len(Config.scope_weights)))
        res_indices = list(range(len(Config.resource_weights)))

        prev_mapping = [[0 for _ in pos_indices] for _ in rack_indices]
        prev_unassigned = [1 for _ in pos_indices]
        for p in pos_indices:
            r = position_df.loc[p, 'status']
            if r!=-1:
                prev_mapping[r][p] = 1
            else:
                prev_unassigned[p] = 0

        x, w, y = {}, {}, {}
        secw, fstw = {}, {}
        for r in rack_indices:
            x[r] = {}
            y[r] = {}
            for p in pos_indices:
                x[r][p]=m.add_var(var_type=BINARY, name=f"x[{r},{p}]")
                y[r][p]=m.add_var(var_type=BINARY, name=f"y[{r},{p}]")
        for i in range(len(Config.spread_metrics)):
            w[i]=m.add_var(var_type=INTEGER, lb=0, name=f"w[{i}]")
            secw[i]=m.add_var(var_type=INTEGER, lb=0, name=f"secw[{i}]")
            fstw[i]=m.add_var(var_type=INTEGER, lb=0, name=f"fstw[{i}]")

        # Objective first term
        #mapping = {rt:1 for rt in range(Config.num_racks)}
        #mapping[-1] = 0.5
        #position_df['trans'] = position_df['status'].map(mapping)
        for r in rack_indices:
            m.objective += xsum(prev_unassigned[p]*y[r][p] for p in pos_indices)
            #m.objective += xsum(2*(position_df.loc[p,'trans']-0.5)*\
            #    (position_df.loc[p,'trans']-x[r][p]) for p in pos_indices)

        # Objective second term
        m.objective += xsum(w[i] for i in range(len(Config.spread_metrics)))
        m.objective += xsum(secw[i] for i in range(len(Config.spread_metrics)))
        m.objective += xsum(fstw[i] for i in range(len(Config.spread_metrics)))

        ## Induced constraints
        level2_scope_res_usage = [set() for _ in range(Config.num_level2_scopes)]
        level1_scope_res_usage = [set() for _ in range(Config.num_level1_scopes)]
        for idx, metric  in enumerate(Config.spread_metrics):
            scope_set, rack_group, res = metric
            # level 3 scope
            for c in scope_set:
                res_usage = set()
                level2_scope = Config.level23_map[c]
                level1_scope = Config.level13_map[c]
                for r in rack_indices:
                    for p in scope_positions_dict[c]:
                        if rack_df.loc[r, 'group'] in rack_group:
                            res_usage.add(x[r][p]*rack_df.loc[r, f'res_{res}'])
                            level2_scope_res_usage[level2_scope].add(x[r][p]*rack_df.loc[r, f'res_{res}'])
                            level1_scope_res_usage[level1_scope].add(x[r][p]*rack_df.loc[r, f'res_{res}'])
                m += w[idx]>=xsum(res_usage), f'spread_idx_{idx}'
            for sc in range(Config.num_level2_scopes):
                m += secw[idx]>=xsum(level2_scope_res_usage[sc])
            for fc in range(Config.num_level1_scopes):
                m += fstw[idx]>=xsum(level1_scope_res_usage[fc])

        for r in rack_indices:
            for p in pos_indices:
                m += y[r][p] >= prev_mapping[r][p]-x[r][p] 
                
        # Constraint 1
        # Demand fulfill
        for r in rack_indices:
            m += demand[r]<=xsum(x[r][p] for p in pos_indices)
        # Every position maps to at most one rack type
        for p in pos_indices:
            m += 1>=xsum(x[r][p] for r in rack_indices)

        # Contraint 2
        # Placement action limit
        actions_taken = set()
        for r in rack_indices:
            for p in pos_indices:
                if position_df.loc[p, 'status']!=r:
                    actions_taken.add(x[r][p]) 
        m += xsum(actions_taken)<=action_limit

        # Constraint 3
        ## TODO-Initilize lists of set for level2 and level1 scopes. 
        ## TODO-Store all the terms in these lists
        res_contribution_level2 = [{res:set() for res in res_indices} for _ in range(Config.num_level2_scopes)]
        res_contribution_level1 = [{res:set() for res in res_indices} for _ in range(Config.num_level1_scopes)]
        for sc, pos_list in scope_positions_dict.items():
            ## TODO-use level23_map to find the level2 scope for pos_list
            ## TODO-use level13_map to find the level1 scope for pos_list
            level2_scope = Config.level23_map[sc] 
            level1_scope = Config.level13_map[sc] 
            for res in res_indices:
                res_contribution = set()
                for r in rack_indices:
                    for p in pos_list:
                        res_contribution.add(x[r][p]*rack_df.loc[r, f'res_{res}'])
                        res_contribution_level2[level2_scope][res].add(x[r][p]*rack_df.loc[r, f'res_{res}'])
                        res_contribution_level1[level1_scope][res].add(x[r][p]*rack_df.loc[r, f'res_{res}'])
                m += xsum(res_contribution)<=L[sc][res]
        ## TODO-Constraint 4 for level1 and level2 resource limits
        ## TODO-Iterate through the lists of sets stored above to add constraint.
        for sc2 in range(Config.num_level2_scopes):
            for res in res_indices:
                m += xsum(res_contribution_level2[sc2][res])<=Config.level2_L[sc2][res]
        for sc1 in range(Config.num_level1_scopes):
            for res in res_indices:
                m += xsum(res_contribution_level1[sc1][res])<=Config.level1_L[sc1][res]
        m.emphasis = 2
        m.objective = minimize(m.objective)
        status = m.optimize(max_seconds=60)
        ans = {r:{p:-1 for p in pos_indices} for r in rack_indices}
        if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
            return "Infeasible", m.objective_value, ans, 0 
        
        movement = 0
        for r in rack_indices:
            for p in pos_indices:
                if x[r][p].x<0.99: continue
                ans[r][p] = r
                if position_df.loc[p, 'status'] not in {-1, r}:
                    movement += 1

        #for sc, pos_list in scope_positions_dict.items():
        #    for p in pos_list:
        #        if ans[p]>-1:
        #            for r in res_indices:
        #                L[sc][r] -= ans[p]*rack_df.loc[k,f'res_{r}']
        #                #scope_rack_res_array[(sc, k)][f'res_{r}'] += ans[p]*rack_df.loc[k, f'res_{r}']
        #                scope_rack_res_array[sc, k, r] += ans[p]*rack_df.loc[k, f'res_{r}']
        #            action_limit -= 1

        #num_placement = [sum([key for key,v in ans[r].items() if v>-1]) for r in rack_indices]
        #assert num_placement >=demand,\
        #    f"position allocated greater than {demand} expected, got:\
        #    {sum([key for key,v in ans.items() if v>-1])}"
        #assert num_placement>2*demand
        return "Pass", m.objective_value, ans, movement

    def solve_no_constr(self, rack_df, position_df, k, demand, action_limit, L, scope_rack_res_dict, mask):
        decision_positions = position_df[position_df['pos_id'].map(lambda x: mask[x]==-1)]
        scope_positions_dict = decision_positions.groupby('scope')['pos_id'].agg(list).to_dict()

        m = Model("Subsolver")
        m.verbose = 0
        pos_indices = decision_positions.index.tolist()
        scope_indices = list(range(len(Config.scope_weights)))
        res_indices = list(range(len(Config.resource_weights)))

        x, w = {}, {}
        for p in pos_indices:
            x[p]=m.add_var(var_type=BINARY, name=f"x[{p}]")
        for i in range(len(Config.spread_metrics)):
            w[i]=m.add_var(var_type=INTEGER, name=f"w[{i}]")

        # Objective first term
        mapping = {rt:0 for rt in range(Config.num_racks)}
        mapping[-1] = 0.5
        mapping[k] = 1
        decision_positions['trans'] = decision_positions['status'].map(mapping)
        m.objective += xsum(2*(decision_positions.loc[p,'trans']-0.5)*\
            (decision_positions.loc[p,'trans']-x[p]) for p in pos_indices)

        # Objective second term
        m.objective += xsum(w[i] for i in range(len(Config.spread_metrics)))

        # Induced constraints
        for idx, metric  in enumerate(Config.spread_metrics):
            scope_set, rack_group, r = metric
            if k not in rack_group:
                continue
            constant = {c:sum([scope_rack_res_dict[(c,other_rack)][f'res_{r}']\
                for other_rack in rack_group]) for c in scope_set}
            for c in scope_set:
                m += w[idx]>=xsum(x[p]*rack_df.loc[k, f'res_{r}']\
                    for p in pos_indices if decision_positions.loc[p, 'scope']==c)\
                        +constant[c], f'spread_idx_{idx}_scope_{c}'

        # Constraint 1
        m += demand==xsum(x[p] for p in pos_indices)

        # Contraint 2
        m += xsum(x[p] for p in pos_indices if position_df.loc[p, 'status']!=k)<=action_limit

        # Constraint 3
        a = {}
        for i in range(len(Config.spread_metrics)):
            a[i]=m.add_var(var_type=INTEGER, name=f"a[{i}]")

        for sc, pos_list in scope_positions_dict.items():
            for r in res_indices:
                m += xsum(x[p]*rack_df.loc[k, f'res_{r}'] for p in pos_list)<=L[sc][r]

        m.emphasis = 2
        m.objective = minimize(m.objective)
        status = m.optimize(max_seconds=60)
        ans = {p:-1 for p in pos_indices}
        if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
            return "Infeasible", m.objective_value, ans, scope_rack_res_dict, action_limit, L

        for p in pos_indices:
            if x[p].x<0.99: continue
            ans[p] = k

        for sc, pos_list in scope_positions_dict.items():
            for p in pos_list:
                if ans[p]>-1:
                    for r in res_indices:
                        L[sc][r] -= ans[p]*rack_df.loc[k,f'res_{r}']
                        scope_rack_res_dict[(sc, k)][f'res_{r}'] += ans[p]*rack_df.loc[k, f'res_{r}']
                    action_limit -= 1

        num_placement = sum([key for key,v in ans.items() if v>-1])
        assert num_placement >=demand,\
            f"position allocated greater than {demand} expected, got:\
            {sum([key for key,v in ans.items() if v>-1])}"
        assert num_placement>2*demand
        return "Pass", m.objective_value, ans, scope_rack_res_dict, action_limit, L

    def solve(self, rack_df, position_df, k, demand, action_limit, L, scope_rack_res_array, mask):
        decision_positions = position_df[position_df['pos_id'].map(lambda x: mask[x]==-1)]
        scope_positions_dict = decision_positions.groupby('scope')['pos_id'].agg(list).to_dict()

        m = Model("Subsolver")
        m.verbose = 0
        pos_indices = decision_positions.index.tolist()
        scope_indices = list(range(len(Config.scope_weights)))
        res_indices = list(range(len(Config.resource_weights)))

        x, w = {}, {}
        for p in pos_indices:
            x[p]=m.add_var(var_type=BINARY, name=f"x[{p}]")
        for i in range(len(Config.spread_metrics)):
            w[i]=m.add_var(var_type=BINARY, name=f"w[{i}]")

        # Objective first term
        mapping = {rt:0 for rt in range(Config.num_racks)}
        mapping[-1] = 0.5
        mapping[k] = 1
        decision_positions['trans'] = decision_positions['status'].map(mapping)
        m.objective += xsum(2*(decision_positions.loc[p,'trans']-0.5)*\
            (decision_positions.loc[p,'trans']-x[p]) for p in pos_indices)

        # Objective second term
        m.objective += xsum(w[i] for i in range(len(Config.spread_metrics)))

        # Induced constraints
        for idx, metric  in enumerate(Config.spread_metrics):
            scope_set, rack_group, r = metric
            if k not in rack_group:
                continue
            constant = {c:sum([scope_rack_res_array[(c,other_rack)][f'res_{r}']\
                for other_rack in rack_group]) for c in scope_set}
            #constant = scope_rack_res_array[:,:,r].sum(axis=1)
            for c in scope_set:
                m += w[idx]>=xsum(x[p]*rack_df.loc[k, f'res_{r}']\
                    for p in pos_indices if decision_positions.loc[p, 'scope']==c)\
                        +constant[c], f'spread_idx_{idx}_scope_{c}'

        # Constraint 1
        m += demand==xsum(x[p] for p in pos_indices)

        # Contraint 2
        m += xsum(x[p] for p in pos_indices if position_df.loc[p, 'status']!=k)<=action_limit

        # Constraint 3
        for sc, pos_list in scope_positions_dict.items():
            for r in res_indices:
                m += xsum(x[p]*rack_df.loc[k, f'res_{r}'] for p in pos_list)<=L[sc][r]

        m.emphasis = 2
        m.objective = minimize(m.objective)
        status = m.optimize(max_seconds=60)
        ans = {p:-1 for p in pos_indices}
        if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
            return "Infeasible", m.objective_value, ans, scope_rack_res_array, action_limit, L

        for p in pos_indices:
            if x[p].x<0.99: continue
            ans[p] = k

        for sc, pos_list in scope_positions_dict.items():
            for p in pos_list:
                if ans[p]>-1:
                    for r in res_indices:
                        L[sc][r] -= ans[p]*rack_df.loc[k,f'res_{r}']
                        scope_rack_res_array[(sc, k)][f'res_{r}'] += ans[p]*rack_df.loc[k, f'res_{r}']
                        #scope_rack_res_array[sc, k, r] += ans[p]*rack_df.loc[k, f'res_{r}']
                    action_limit -= 1

        num_placement = sum([key for key,v in ans.items() if v>-1])
        assert num_placement >=demand,\
            f"position allocated greater than {demand} expected, got:\
            {sum([key for key,v in ans.items() if v>-1])}"
        assert num_placement>2*demand
        return "Pass", m.objective_value, ans, scope_rack_res_array, action_limit, L


    def _solve(self, rack_df, position_df, k, demand, action_limit, L, mask):
        # Merge the DataFrames on the 'status' column (rt_id) and 'group' column
        merged_df = pd.merge(position_df, rack_df, left_on='status',\
            right_on='rt_id', how='left', suffixes=('_pos', '_rack'))

        # Filter rows where 'status' matches a 'rt_id'
        determined_positions = merged_df[merged_df['pos_id'].map(lambda x: mask[x]!=-1)]
        decision_positions = merged_df[merged_df['pos_id'].map(lambda x: mask[x]==-1)]
        #determined_positions = merged_df[(merged_df['status']!=-1)&(merged_df['status']!=k)]
        #decision_positions = merged_df[(merged_df['status']==1)|(merged_df['status']==k)]

        ## Create a dictionary with 'scope' as the key and an array of 'pos_id' as the value
        #pos_id_dict = selected_positions.groupby('scope')['pos_id'].apply(list).to_dict()

        # Group by 'scope' and sum 'res_1' to 'res_10' for each group
        sum_by_scope = determined_positions.groupby('scope')[[f'res_{i}' for i in\
            range(len(Config.resource_weights))]].sum().reset_index()
        sum_by_scope_rack = determined_positions.groupby(['scope', 'rt_id'])[[f'res_{i}' for i in\
            range(len(Config.resource_weights))]].sum().reset_index()
        scope_res_temp = sum_by_scope.set_index('scope').to_dict(orient='index')
        scope_res_dict = {sc: {f'res_{r}':0 for r in range(len(Config.resource_weights))}\
            for sc in range(len(Config.scope_weights))}
        for sc in scope_res_temp:
            for res, val in scope_res_temp[sc].items():
                scope_res_dict[sc][res] = val
        scope_rack_res_dict = {}
        for sc in range(len(Config.scope_weights)):
            for rt_id in range(Config.num_racks):
                scope_rack_res_dict[(sc, rt_id)] = {f'res_{r}':0 for r in range(len(Config.resource_weights))}
        scope_rack_res_temp = sum_by_scope_rack.set_index(['scope', 'rt_id']).to_dict(orient='index')
        for key, val in scope_rack_res_temp.items():
            scope_rack_res_dict[key] = val
        ## Iterate over rows in the DataFrame and populate the nested dictionary
        #for index, row in sum_by_scope.iterrows():
        #    scope = row['scope']
        #    res_values = {f'res_{i}': row[f'res_{i}'] for i in range(1, len(Config.resource_weights)+1)}
        #    scope_res_dict[scope] = res_values


        m = Model("Subsolver")
        m.verbose = 0
        pos_indices = decision_positions.index.tolist()
        #scope_indices = list(range(len(Config.scope_weights)))
        res_indices = list(range(len(Config.resource_weights)))

        #x, y, z, w = {}, {}, {}, {}
        x, w = {}, {}
        for p in pos_indices:
            x[p]=m.add_var(var_type=BINARY, name=f"x[{p}]")
            #y[p]=m.add_var(var_type=BINARY, name=f"y[{p}]")
            #z[p]=m.add_var(var_type=BINARY, name=f"z[{p}]")
        for i in range(len(Config.spread_metrics)):
            w[i]=m.add_var(var_type=BINARY, name=f"w[{i}]")

        # Objective first term
        mapping = {rt:0 for rt in range(Config.num_racks)}
        mapping[-1] = 0.5
        mapping[k] = 1
        # Use decision_positions instead of the position_df because we don't consider positions
        # already assigned.
        decision_positions['trans'] = decision_positions['status'].map(mapping)
        m.objective += xsum(2*(decision_positions.loc[p,'trans']-0.5)*\
            (decision_positions.loc[p,'trans']-x[p]) for p in pos_indices)

        # Objective second term
        m.objective += xsum(w[i] for i in range(len(Config.spread_metrics)))

        # Induced constraints
        for idx, metric  in enumerate(Config.spread_metrics):
            scope_set, rack_group, r = metric
            if k not in rack_group:
                continue
            constant = {c:sum([scope_rack_res_dict[(c,other_rack)][f'res_{r}']\
                for other_rack in rack_group]) for c in scope_set}
            for c in scope_set:
                m += w[idx]>=xsum(x[p]*rack_df.loc[k, f'res_{r}']\
                    for p in pos_indices if decision_positions.loc[p, 'scope']==c)\
                        +constant[c], f'spread_idx_{idx}_scope_{c}'

        # Constraint 1
        m += demand==xsum(x[p] for p in pos_indices)

        # Contraint 2
        # Only the positions previously not mapped to rack type k have to be
        # considered when we are counting the number of placements
        m += xsum(x[p] for p in pos_indices if position_df.loc[p, 'status']!=k)<=action_limit

        # Constraint 3
        for p in pos_indices:
            sc = decision_positions.loc[p, 'scope']
            for r in res_indices:
                m += x[p]*rack_df.loc[k, f'res_{r}']+scope_res_dict[sc][f'res_{r}']<=L[sc][r]

        m.emphasis = 2
        m.objective = minimize(m.objective)
        status = m.optimize(max_seconds=60)
        ans = {p:-1 for p in pos_indices}
        if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
            return "Infeasible", m.objective_value, ans

        for p in pos_indices:
            if x[p].x<0.99: continue
            ans[p] = k

        assert sum([key for key,v in ans.items() if v>-1])>=demand,\
            f"position allocated greater than {demand} expected, got:\
            {sum([key for key,v in ans.items() if v>-1])}"
        return "Pass", m.objective_value, ans

class Solver:
    pass
