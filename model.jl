using JuMP, Gurobi, DataFrames, CSV
include("src/state_space.jl")
include("src/optimization.jl")



# Constants and time horizon extension
T = 12  # Total number of time periods extended to 12 hours
I = 3   # Number of vehicles
P_max = 10.0  # Maximum charging power
E_max = 20.0  # Maximum state of charge
η = 1.0  # Charging efficiency
λ_t = [0.1 for t in 1:T]  # Cost rate per kWh charged

# Define actions for each car
# Action format: (charge amount, trip status)
# (10, 0) -> Charge 10 units, do not take the trip
# (5, 0) -> Charge 5 units, do not take the trip
# (0, 1) -> No charge, take the trip
actions = [(10, 0), (5, 0), (0, 1)]

# Generate all possible action combinations for three cars
action_vectors = generate_combinations(actions)

# Generate all possible combinations of demand
state_vectors = generate_state_space_vector(E_max, 5)