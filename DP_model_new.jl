using JuMP, Gurobi, DataFrames, CSV
include("src/state_space.jl")
include("src/optimization.jl")
include("src/action_space.jl")
include("Price_data/price_extraction.jl")

# Constants and parameters
T = 12  # Total number of time periods extended to 12 hours
I = 3   # Number of vehicles
P_max = 10.0  # Maximum charging power
E_max = 20.0  # Maximum state of charge
η = 1.0  # Charging efficiency
num_scenarios = 17  # Number of price scenarios
λ_t = zeros(17)
time_stamp = collect(8:19)
penalty_level = 1.0

# Load price data
prices = read_extract_prices("Price_data/price_scenario_data.h5", num_scenarios)  # adjust indices as per your data structure

#price adjustment
prices.Price = prices.Price*0.001

# Load trip data
trip_data = CSV.read("Trip_data/trip_data.csv", DataFrame)

# Generate all possible action combinations for three cars
actions = [(10, 0), (5, 0), (0,0), (0,1)]
action_vectors = generate_combinations(actions)

# Generate all possible combinations of state
state_vectors = generate_state_space_vector(E_max, 5)

# Initialize dynamic programming value function array (simplified assumption)
V = zeros(Float64, length(state_vectors), T+1)  # Terminal condition V[:, T+1] = 0 already set


# Call the function to process and save the data
data_frames_array = save_action_state_data_to_array(V)

test_df = save_optimal_actions_states_to_file(V)

# Call the function to execute the consolidation process
all_optimal_actions, all_optimal_states = consolidate_optimal_data()

all_optimal_actions

