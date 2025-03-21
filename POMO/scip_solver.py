import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
import gurobipy as gp
import numpy as np
from config_clean import config
from pyscipopt import Model, quicksum
from RMProblemDef_clean import ProblemGenerator
from gurobipy import GRB

class Solver:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.config = env_params['config']
        self.problem_size = env_params['problem_size']
        self.batch_size = env_params['test_batch_size']
        self.generator = ProblemGenerator(self.config)
        self.problems, self.prev_pos_rack_map, self.demand, self.action_limit,\
            _, _ = self.generator.get_random_problems(self.batch_size, self.problem_size)

    def set_sys_params(self):
        self.P = self.config.num_positions
        self.K = self.config.num_rack_types
        self.num_cols_R = len(self.config.resource_weights)
        self.ss1 = self.config.num_level1_scopes
        self.ss2 = self.config.num_level2_scopes
        self.ss3 = self.config.num_scopes
        self.LS1 = self.config.level1_L
        self.LS2 = self.config.level2_L
        self.LS3 = self.config.L

        self.R = self.config.resource_table
        self.S3 = self.one_hot_encode_np(self.config.scopes, self.ss3)
        self.S2 = self.one_hot_encode_np(self.config.level23_map, self.ss2)
        self.S1 = self.one_hot_encode_np(self.config.level13_map, self.ss1)
        self.S2 = np.dot(self.S3, self.S2)
        self.S1 = np.dot(self.S3, self.S1)

    def one_hot_encode_np_v2(self, data, num_categories):
        return np.eye(num_categories)[data]

    def one_hot_encode_np_v3(self, X_prime, num_columns):
        """
        Function to one-hot encode a NumPy array
        X_prime: input array of shape (N, ...)
        num_columns: number of columns for one-hot encoding
        """
        # Create a one-hot encoded array with an extra dimension for the one-hot encoding
        shape = (*X_prime.shape, num_columns)
        X_one_hot = np.zeros(shape, dtype=np.float32)
        # Create indices for all dimensions
        rows, cols = np.indices(X_prime.shape)

        # Create a mask for valid indices (less than num_columns)
        mask = X_prime < num_columns
        # Set 1s in the appropriate places, except where X_prime is num_columns
        X_one_hot[rows[mask], cols[mask], X_prime[mask]] = 1.0
        return X_one_hot

    def one_hot_encode_np(self, X_prime, num_columns):
        """
        Function to one-hot encode a NumPy array
        X_prime: input array of shape (N, ...)
        num_columns: number of columns for one-hot encoding
        """
        # Ensure that X_prime is an integer array for indexing
        X_prime = np.clip(X_prime, 0, num_columns - 1)
        # Check if X_prime is 1D and reshape it to 2D for uniform processing
        if X_prime.ndim == 1:
            X_prime = X_prime[:, None]  # Reshape (a,) to (a, 1)
            is_1d = True
        else:
            is_1d = False
        # Create a one-hot encoded array with an extra dimension for the one-hot encoding
        shape = (*X_prime.shape, num_columns)
        X_one_hot = np.zeros(shape, dtype=np.float32)
        # Iterate over each position in X_prime to set the correct one-hot encoding
        rows = np.arange(X_prime.shape[0])[:, None]  # Reshape to (a, 1) for broadcasting with X_prime
        cols = np.arange(X_prime.shape[1])  # Create an array for column indices (for the second dimension)
        # Set the appropriate locations to 1 using advanced indexing
        X_one_hot[rows, cols, X_prime] = 1
        # Return the result, ensuring it has the right shape for 1D inputs
        return X_one_hot if not is_1d else X_one_hot[:, 0, :]

    def reset_sol_dict(self):
        self.sol_dict = {}
        self.sol_dict['obj'] = []
        self.sol_dict['var'] = []
        self.sol_dict['status'] = []
        self.sol_dict['reward1'] = []
        self.sol_dict['reward2'] = []
        self.sol_dict['reward3'] = []
        self.sol_dict['reward4'] = []
        self.sol_dict['move_cost'] = []

    def solve_whole(self):
        self.reset_sol_dict()
        prev_pos_rack_map_np = self.prev_pos_rack_map.cpu().numpy()
        prev_pos_rack_map_np = np.squeeze(prev_pos_rack_map_np, axis=2)
        demand_np = self.demand.cpu().numpy()
        action_limit_np = self.action_limit.cpu().numpy()

        oh_enc_prev_pos_rack_map = self.one_hot_encode_np_v3(prev_pos_rack_map_np, self.config.num_rack_types)

        for b in range(self.batch_size):
            objval, sol, status = self._solve(oh_enc_prev_pos_rack_map[b], demand_np[b], action_limit_np[b], b)
            if status in [3, 4]:
                self.sol_dict['obj'].append(0)
                #self.sol_dict['var'].append(0)
                X = oh_enc_prev_pos_rack_map[b]
            else:
                self.sol_dict['obj'].append(objval)
                #self.sol_dict['var'].append(sol)
                X = self.convert_sol_to_np(sol)
            mc, r1, r2, r3, r4 = self.get_reward_terms(oh_enc_prev_pos_rack_map[b], X)
            self.sol_dict['reward1'].append(r1)
            self.sol_dict['reward2'].append(r2)
            self.sol_dict['reward3'].append(r3)
            self.sol_dict['reward4'].append(r4)
            self.sol_dict['move_cost'].append(mc)
            self.sol_dict['status'].append(status)

        return self.sol_dict

    def convert_sol_to_np(self, sol):
        # Initialize a zero array of shape (P, K)
        result = np.zeros((self.P, self.K), dtype=int)
        # Set the corresponding positions to 1 where (p, k) is in vars
        for p, k in sol:
            result[p, k] = 1
        return result

    def get_reward_terms(self, X_bar, X):
        move_cost = np.sum(np.abs(X-X_bar))
        reward_one = self.res_constr_terms(X, self.S1, self.R, self.LS1)
        reward_two = self.compute_sum_variance(self.S3, X, self.R)
        reward_three = self.res_constr_terms(X, self.S2, self.R, self.LS2)
        reward_four = self.res_constr_terms(X, self.S3, self.R, self.LS3)
        return move_cost, reward_one, reward_two, reward_three, reward_four

    def compute_sum_variance(self, S, X, R):
        product = np.dot(S.T, np.dot(X, R))
        variances = np.var(product, axis=0)
        return np.sum(variances)

    def res_constr_terms(self, X, S, R, L):
        product = np.dot(S.T, np.dot(X, R))
        diff = L-product
        min_diff = np.minimum(0, diff)
        #softplus_result = self.softplus(min_diff, -3, 20)
        #return np.sum(softplus_result)
        violation_amount = np.sum(min_diff**2)
        return violation_amount

    def softplus(self, x, a, b):
        return 1/b*np.log1p(np.exp(a*b*x))

class ScipSolver(Solver):
    def __init__(self, **env_params):
        super().__init__(**env_params)
        self.model = Model("SCIP_Optimization")
        self.x = None  # Will hold SCIP decision variables

    def define_variables(self):
        """Define decision variables specific to SCIP."""
        self.x = {}
        for p in range(self.P):
            for k in range(self.K):
                self.x[p, k] = self.model.addVar(vtype="B")

    def objective(self, X_bar):
        obj = quicksum((self.x[p, k] - X_bar[p, k]) ** 2 for p in range(self.P) for k in range(self.K))

        for j in range(self.num_cols_R):
            col_elements = [
                quicksum(self.S3[p, i] * quicksum(self.x[p, k] * self.R[k, j] for k in range(self.K)) for p in range(self.P))
                for i in range(self.ss3)
            ]
            col_mean = quicksum(col_elements) / self.ss3
            variance = quicksum((z - col_mean) ** 2 for z in col_elements) / self.ss3
            obj += variance
        return obj

    def _solve(self, X_bar, d, q):
        self.set_sys_params()
        self.define_variables()

        # Set the nonlinear objective
        self.model.setObjective(self.objective(X_bar), "minimize")

        # Constraints
        for p in range(self.P):
            self.model.addCons(quicksum(self.x[p, k] for k in range(self.K)) == 1)

        for k in range(K):
            self.model.addCons(quicksum(self.x[p, k] for p in range(self.P)) >= d[k])

        #for S, LS, ss in [(self.S1, self.LS1, self.ss1), (self.S2, self.LS2, self.ss2), (self.S3, self.LS3, self.ss3)]:
        #    for i in range(ss):
        #        for j in range(num_cols_R):
        #            product = quicksum(S[p, i] * quicksum(self.x[p, k] * self.R[k, j]\
        #                for k in range(self.K)) for p in range(self.P))
        #            self.model.addCons(self.LS[i, j] - product >= 0)

        # Handling the max function in the constraint
        for p in range(self.P):
            for k in range(self.K):
                diff_var = self.model.addVar(name=f"diff_{p}_{k}")
                self.model.addCons(diff_var >= self.x[p, k] - X_bar[p, k])
                self.model.addCons(diff_var >= 0)
        self.model.addCons(quicksum(diff_var for p in range(self.P) for k in range(self.K)) <= q)

        #max_diff_sum = quicksum(model.max(x[p, k] - X_bar[p, k], 0) for p in range(P) for k in range(K))
        self.model.addCons(max_diff_sum <= q)

        # Set a time limit (in seconds) for solving the problem
        self.model.setParam("limits/time", 300)
        # Set verbose mode to see what's going on
        self.model.setParam("display/verblevel", 1)
        # Optimize the model
        self.model.optimize()

        # Process results
        status = self.model.getStatus()
        if status in ["timelimit", "optimal"]:
            objval = self.model.getObjVal()
            var = [(p, k) for p in range(self.P) for k in range(self.K) if self.model.getVal(self.x[p, k]) > 0.5]
            return objval, var
        else:
            print(f"No optimal solution found. Status: {status}")
            return None, None

class GRBSolver(Solver):
    def __init__(self, **env_params):
        super().__init__(**env_params)
        self.model = gp.Model("Gurobi_Optimization")
        self.x = None  # Will hold Gurobi decision variables

    def define_variables(self):
        """Define decision variables specific to Gurobi."""
        self.x = self.model.addVars(self.P, self.K, vtype=GRB.BINARY, name="x")

    def add_constraints(self, X_bar, d, q, res_penalty=True):
        """Add constraints specific to Gurobi."""
        # Row-wise sum constraint: sum_{k} x_{p,k} <= 1 for all p
        for p in range(self.P):
            self.model.addConstr(gp.quicksum(self.x[p, k] for k in range(self.K)) <= 1, name=f"row_sum_{p}")

        # Column-wise minimum sum constraint: sum_{p} x_{p,k} >= d_k for all k
        for k in range(self.K):
            self.model.addConstr(gp.quicksum(self.x[p, k] for p in range(self.P)) >= d[k], name=f"col_min_{k}")

        if not res_penalty:
            for S, LS, ss, rows in [(self.S1, self.LS1, self.ss1, self.ss3), (self.S2, self.LS2,\
                                            self.ss2, self.ss3), (self.S3, self.LS3, self.ss3, self.P)]:
                for i in range(ss):
                    for j in range(self.num_cols_R):
                        product = gp.quicksum(S[r_id, i] * gp.quicksum(self.x[r_id, k] * self.R[k, j]\
                            for k in range(self.K)) for r_id in range(rows))
                        self.model.addConstr(LS[i, j] - product >= 0)

        # Handling the max function in the constraint
        diff_vars = self.model.addVars(self.P, self.K, name="diff")
        diff_sum = 0
        for p in range(self.P):
            for k in range(self.K):
                self.model.addConstr(diff_vars[p, k] - self.x[p, k] >= - X_bar[p, k], name=f"diff_{p}_{k}")
                self.model.addConstr(diff_vars[p, k] >= 0)
                diff_sum += diff_vars[p, k]

        self.model.addConstr(diff_sum <= q[0], name="ac_limit")
        #try:
        #    self.mode.addConstr(diff_sum <= q, name="ac_limit")
        #except Exception as e:
        #    print(f"Error adding constraint: {e}")

    def update_constraints(self, X_bar, d, q):
        for k in range(self.K):
            self.model.getConstrByName(f"col_min_{k}").RHS = d[k]
            for p in range(self.P):
                self.model.getConstrByName(f"diff_{p}_{k}").RHS = -X_bar[p, k]

        self.model.getConstrByName("ac_limit").RHS = q[0]

    def add_objective(self, X_bar, res_penalty=True):
        # Objective function
        obj = gp.QuadExpr()
        for p in range(self.P):
            for k in range(self.K):
                obj += (self.x[p, k] - X_bar[p, k]) * (self.x[p, k] - X_bar[p, k])

        obj = self.compute_variance(obj)
        # Set the objective function
        if res_penalty:
            obj = self.resource_limit_penalty(obj)
        self.model.setObjective(obj, GRB.MINIMIZE)

    def add_objective_penalty(self, X_bar):
        # Objective function
        obj = gp.QuadExpr()
        for p in range(self.P):
            for k in range(self.K):
                obj += (self.x[p, k] - X_bar[p, k]) * (self.x[p, k] - X_bar[p, k])

        obj = self.compute_variance(obj)
        # Set the objective function
        self.model.setObjective(obj, GRB.MINIMIZE)

    def resource_limit_penalty(self, obj):
        """Compute the penalty caused by scope resource limit violations"""
        for S, LS, ss in [(self.S1, self.LS1, self.ss1), (self.S2, self.LS2,\
                                        self.ss2), (self.S3, self.LS3, self.ss3)]:
            for i in range(ss):
                for j in range(self.num_cols_R):
                    product = gp.quicksum(S[p, i] * gp.quicksum(self.x[p, k] * self.R[k, j]\
                        for k in range(self.K)) for p in range(self.P))
                    aux_var = self.model.addVar(name=f"aux_{i}_{j}")
                    self.model.addConstr(aux_var >= 0)
                    self.model.addConstr(aux_var >= product - LS[i, j])
                    obj += aux_var*aux_var
        return obj

    def compute_variance(self, obj):
        """Compute the variance term for Gurobi."""
        # Variance calculation using lifting technique
        for j in range(self.num_cols_R):
            # Create variables for each element in the column
            y = self.model.addVars(self.ss3, name=f"y_{j}")
            # Define y variables
            for i in range(self.ss3):
                self.model.addConstr(y[i] == gp.quicksum(self.S3[p, i] * gp.quicksum(\
                    self.x[p, k] * self.R[k, j] for k in range(self.K)) for p in range(self.P)))
            # Calculate mean
            mean = gp.quicksum(y) / self.ss3
            # Create variables for squared differences
            z = self.model.addVars(self.ss3, name=f"z_{j}")
            # Define constraints for z variables (lifting technique)
            for i in range(self.ss3):
                self.model.addConstr(z[i] >= (y[i] - mean) * (y[i] - mean))
            # Add variance to objective
            obj += gp.quicksum(z) / self.ss3 * 100

        return obj

    def _solve(self, X_bar, d, q, p_id, res_penalty=True):
        """Solve the Gurobi model."""
        if p_id==0:
            self.set_sys_params()
            self.define_variables()
            self.add_objective(X_bar, res_penalty)
            self.add_constraints(X_bar, d, q, res_penalty)
            # Solver parameters
            self.model.setParam("TimeLimit", 3600)  # Increased time limit to 5 minutes
            self.model.setParam("OutputFlag", 1)  # Enable output (similar to display/verblevel)
            if res_penalty:
                self.model.setParam("NonConvex", 2)
        else:
            self.add_objective(X_bar, res_penalty)
            self.update_constraints(X_bar, d, q)
        self.model.optimize()
        # Process results
        status = self.model.status

        if self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Applying FeasRelax to relax constraints...")

            # Apply FeasRelax to relax constraints and try to make the model feasible
            self.model.feasRelaxS(0, True, False, True)  # Relax constraints only (crelax=True)
            self.model.optimize()

            # Check if a solution was found after relaxation
            if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                print("FeasRelax found a feasible solution with relaxed constraints.")
                print(f"Objective value after relaxation: {self.model.ObjVal}")
                objval = self.model.objVal
                try:
                    var = [(p, k) for p in range(self.P) for k in range(self.K) if self.x[p, k].X > 0.5]
                    return objval, var, 2
                except Exception as e:
                    print(type(e))
                    print(e)
                    return None, None, 3
            else:
                print("No feasible solution found, even after applying FeasRelax.")
                return None, None, 3
        elif status in [GRB.TIME_LIMIT, GRB.OPTIMAL]:
            objval = self.model.objVal
            try:
                var = [(p, k) for p in range(self.P) for k in range(self.K) if self.x[p, k].X > 0.5]
                return objval, var, 1
            except Exception as e:
                print(type(e))
                print(e)
                return None, None, 3
        else:
            print(f"No optimal solution found. Status: {self.model.status}")
            return None, None, 4

def scip_solve(**env_params):
    solver = ScipSolver(**env_params)
    solver.solve_whole()

def gurobi_solve(**env_params):
    solver = GRBSolver(**env_params)
    solver.solve_whole()

if __name__=='__main__':
    env_params = {
        'problem_size': 10,
        'pomo_size': 10,
        'config': config,
        'periods': 1,
    }
    gurobi_solve(**env_params)
