from pulp import *
import numpy as np
import matplotlib.pyplot as plt

class LPPSolverError(Exception): #Custom Exception
    ...

# Function to convert expression into usable variables
def expr_to_vars(mode: str, expression: str):
    index = 0
    for i in expression:
        if i == "+":
            vars = list(expression.partition("+"))
            break
        # This part is probably not required but future proofing is good practice
        elif i == "-" and index != 0:
            if expression.count("-") == 1:
                vars = list(expression.partition("-"))
                vars[2] = "-" + vars[2]
            else:
                vars = list(expression[1::].partition("-"))
                vars[0] = "-" + vars[0]
                vars[2] = "-" + vars[2]
            break
        index += 1
    unstripped = vars[2]
    vars[2] = vars[2].rstrip("-<=>1234567890")
    coeffs = []
    for i in vars:
        if i != "+" and i != "-":
            temp = i.strip("xy")
            if temp == "":
                coeffs.append("1")
            elif temp == "-":
                coeffs.append("-1")
            else:
                coeffs.append(temp)
    if mode == "constraint":
        vars[2] = unstripped.lstrip("-1234567890xy")
        relation = vars[2][0:2]
        if relation not in ("<=", "==", ">="):
            raise LPPSolverError("Invalid relational operator. Expected '<=', '==' or '>='.")
        coeffs.append(relation)
        coeffs.append(vars[2][2:])
    return coeffs

# Main function that plots graph and prints the 'Z min' or 'Z max' value
def plot_problem(condition: str, expression: str, constraints: list[str]):
    num_vars = 2
    num_cons = len(constraints)

    # Define the problem
    if condition == "maximize":
        prob = LpProblem("MODEL", LpMaximize)
        label = "Z max"
    elif condition == "minimize":
        prob = LpProblem("MODEL", LpMinimize)
        label = "Z min"
    else:
        raise LPPSolverError("Invalid input. Expected 'maximize' or 'minimize'.")

    # Define the decision variables
    vars_list = ["x", "y"]
    vars_dict = {}
    for i in range(num_vars):
        vars_dict[vars_list[i]] = LpVariable(name=vars_list[i], lowBound=0)

    # Define the objective function
    coeffs = expr_to_vars("expression", expression)
    obj_func = LpAffineExpression([(vars_dict[vars_list[i]], int(coeffs[i])) for i in range(num_vars)])
    prob += obj_func

    # Define the constraints
    cons_list = []
    for i in range(num_cons):
        coeffs = expr_to_vars("constraint", constraints[i])
        cons_lhs_list = [(vars_dict[vars_list[j]] , int(coeffs[j])) for j in range(num_vars)] #[(x, n1), (y, n2)]
        cons_lhs = LpAffineExpression(cons_lhs_list)
        cons_rhs = int(coeffs[3])
        cons_list.append(cons_lhs_list)
        cons_list.append(cons_rhs)
        cons_op = coeffs[2]
        if cons_op == "<=":
            prob += cons_lhs <= cons_rhs
        elif cons_op == "==":
            prob += cons_lhs == cons_rhs
        else:
            prob += cons_lhs >= cons_rhs

    # Suppress solver output and solve the problem
    LpSolverDefault.msg = False
    prob.solve()

    # Print the optimal solution
    print("\nOptimal solution:")
    print(f"{label} = {value(prob.objective)} at ({value(vars_dict["x"])}, {value(vars_dict["y"])})")

    # Plot the constraints and add the solution
    plt.figure()

    # Plot the constraints
    plot_values = []
    for i in range(0, len(cons_list), 2):
        cons_lhs_np = np.array([cons_list[i][j][1] for j in range(num_vars)])
        cons_rhs_np = cons_list[i+1]
        if cons_lhs_np[0] != 0: # if x is in the constraint
            x_vals = np.linspace(0, cons_rhs_np/cons_lhs_np[0], 2)
            y_vals = (cons_rhs_np - cons_lhs_np[0]*x_vals)/cons_lhs_np[1]
        elif cons_lhs_np[1] != 0: # if y is in the constraint
            y_vals = np.linspace(0, cons_rhs_np/cons_lhs_np[1], 2)
            x_vals = (cons_rhs_np - cons_lhs_np[1]*y_vals)/cons_lhs_np[0]
        else:
            raise LPPSolverError("Invalid constraint. Constraint not plotted.")
        plot_values.append((x_vals, y_vals))

    for i in range(len(plot_values)):
        plt.plot(plot_values[i][0], plot_values[i][1], label=constraints[i], marker="x", markersize=10)

    # Plot the solution (Maximize / Minimize coordinates)
    plt.scatter(value(vars_dict["x"]), value(vars_dict["y"]), color="red", label=label)

    # Set the plot labels and legend
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()

    # Show the plot
    plt.show()

print("""Welcome to the LPP Solver! Please enter the following details without spaces!
Note that while entering constraints, use '<=', '==' or '>=' equality symbols. Enjoy graphing!\n""")
condition = input("Do you want to maximize or minimize? ").lower()
expression = input("Enter the Z expression: ")
num_cons = int(input("Enter the number of constraints: "))
constraints = []
for i in range(1, num_cons+1):
    constraint = input(f"Enter constraint {i}: ")
    constraints.append(constraint)

plot_problem(condition, expression, constraints)