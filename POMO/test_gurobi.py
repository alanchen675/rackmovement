import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("example")

# Create decision variables
x = model.addVar(vtype=GRB.BINARY, name="x")
y = model.addVar(vtype=GRB.BINARY, name="y")
z = model.addVar(vtype=GRB.BINARY, name="z")

# Set the objective function
model.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

# Add constraints
model.addConstr(x + 2 * y + 3 * z <= 4, "c0")
model.addConstr(x + y >= 1, "c1")

# Optimize the model
model.optimize()

# Print the decision variable values and the objective value
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found:")
    print(f"x = {x.X}")
    print(f"y = {y.X}")
    print(f"z = {z.X}")
    print(f"Objective value: {model.ObjVal}")
else:
    print(f"No optimal solution found. Status code: {model.status}")

