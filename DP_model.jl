
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
λ_t = [0.1 for t in 1:num_scenarios]  # Cost rate per kWh charged

# Load price data
prices = read_extract_prices("Price_data/price_scenario_data.h5", num_scenarios)  # adjust indices as per your data structure

# Load trip data
trip_data = CSV.read("Trip_data/trip_data.csv", DataFrame)

# Generate all possible action combinations for three cars
actions = [(10, 0), (5, 0), (0, 1), (0,0)]
action_vectors = generate_combinations(actions)

# Generate all possible combinations of state
state_vectors = generate_state_space_vector(E_max, 5)

# Initialize dynamic programming value function array (simplified assumption)
V = zeros(Float64, length(state_vectors), T+1)  # Terminal condition V[:, T+1] = 0 already set

function state_to_value(state, state_vectors, t, V)
    # This function finds the index of 'state' in 'state_vectors'
    rounded_state = round.(state / 5) * 5
    if any(x -> x > E_max, state)
        return 1000000
    end

    if any(x -> x < 0, state)
        return 1000000
    end

    for (idx, s) in enumerate(state_vectors)
        if s == rounded_state
            return V[idx, t]
        end
    end
    state
    error("State not found in state vectors")
end

function calculate_new_state(current_state, action, eta, trip_data, t, num_vehicles)
    # Initialize the new state vector
    new_state = zeros(length(current_state))

    # Iterate over each vehicle
    for i in 1:num_vehicles
        # Extract current state of charge, charging action, and trip decision
        E_i_t = current_state[i]
        P_C_i_t, u_i_t = action[i]
        
        # Extract trip demand for vehicle i at time t (assumed to be stored in trip_data DataFrame)
        d_i_t = trip_data[t, Symbol("Car_$i")]

        # Calculate the new state of charge based on the dynamics
        E_i_t_plus_1 = E_i_t + eta * P_C_i_t * (1 - u_i_t) - d_i_t * u_i_t
        new_state[i] = E_i_t_plus_1
    end
    
    return round.(Int64, new_state)
end

function calculate_cost(lambda_t, action, state, trip_data, t, num_vehicles)
    total_cost = 0.0

    # Calculate cost for each vehicle
    for i in 1:num_vehicles
        # Extract charging power and trip decision from the action
        P_C_i_t, u_i_t = action[i]
        d_i_t = trip_data[t, Symbol("Car_$i")]

        # Calculate the charging cost for this vehicle at this time
        charging_cost = lambda_t.Price .* P_C_i_t

        # Calculate penalty for negative state of charge (if applicable)
        # Assuming state vector contains E_i_t directly
        E_i_t = state[i]
        penalty_cost = 1000 * max(0, -E_i_t) - 100*d_i_t*u_i_t  # Penalty only applies if E_i_t is negative

        # Aggregate cost for this vehicle
        for j in 1:length(charging_cost)[1]
            total_cost += 0.17*(charging_cost[j] + penalty_cost)
        end
    end

    return total_cost
end


# create a vector V 
V = zeros(Float64, length(state_vectors), T+1)
time = collect(8:19)
optimal_actions = Array{Any}(undef, length(state_vectors), T)

# SDP recursive calculations
for t in T:-1:1
    λ_t = prices[prices.Time .== time[t], :]  # Price at time t
    for state_idx in eachindex(state_vectors)
        state = state_vectors[state_idx];
        min_cost = Inf;
        optimal_action = nothing;
        for action in action_vectors
            # Calculate new state based on the dynamics
            new_state = calculate_new_state(state, action, η, trip_data, t, I);
            cost = calculate_cost(λ_t, action, state, trip_data, t, I) + state_to_value(new_state, state_vectors, t+1, V); # Include future cost from V
            if cost < min_cost
                min_cost = cost;
                optimal_action = action;
            end
        end
        V[state_idx, t] = min_cost;  # Store the minimal cost found for this state at time t
        optimal_actions[state_idx, t] = optimal_action;
    end
    println("Optimal costs computed for time period $t.")
end

println("Optimal costs computed for each state and time period.")


# λ_t = prices[prices.Time .== time[11], :] 
# test_state = [10, 10, 12]
# test_action = [(10, 0), (5, 0), (0, 1)]

# calculate_cost(λ_t, test_action, test_state, trip_data, 11, 3)

# # P_C_i_t, u_i_t = test_action[1]
# # λ_t.Price
# # charging_cost = λ_t.Price .* P_C_i_t


# for state_index in eachindex(state_vectors)
#     println("State: ", state_vectors[state_index])
#     println("Optimal cost: ", V[state_index, 1])
# end

# state_to_value(test_state, state_vectors, 13, V)

# if any(x -> x > E_max, [30, 20, 20])
#     println("yes")
# end

# rounded_state = round.(test_state/5)*5

# save V to CSV

V_df = DataFrame(V, :auto)
optimal_actions_df = DataFrame(optimal_actions, :auto)
# add a column to the left in the dataframe
insertcols!(V_df, 1, :state => state_vectors)
insertcols!(optimal_actions_df, 1, :state => state_vectors)
CSV.write("Results/V.csv", V_df)
CSV.write("Results/optimal_actions.csv", optimal_actions_df)