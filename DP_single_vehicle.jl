using JuMP, Gurobi, DataFrames, CSV
include("src/state_space.jl")
include("src/optimization.jl")
include("src/action_space.jl")
include("Price_data/price_extraction.jl")

# Constants and parameters
T = 12  # Total number of time periods extended to 12 hours
I = 1   # Number of vehicles
P_max = 10.0  # Maximum charging power
E_max = 20.0  # Maximum state of charge
η = 1.0  # Charging efficiency
num_scenarios = 17  # Number of price scenarios
λ_t = zeros(17)
time_stamp = collect(8:19)

# Load price data
prices = read_extract_prices("Price_data/price_scenario_data.h5", num_scenarios)  # adjust indices as per your data structure

#price adjustment
prices.Price = prices.Price*0.1

# Load trip data
trip_data = CSV.read("trip_data.csv", DataFrame)

# Generate all possible action combinations for three cars
actions = [(10, 0), (5, 0), (0,1)]
action_vectors = generate_combinations_single(actions)

# Generate all possible combinations of state
state_vectors = generate_state_space_vector_single(E_max, 5)

# Initialize dynamic programming value function array (simplified assumption)
V = zeros(Float64, length(state_vectors), T+1)  # Terminal condition V[:, T+1] = 0 already set

λ_t = prices[prices.Time .== time_stamp[12], :]

function state_to_value(state, state_vectors, t, V)
    # This function finds the index of 'state' in 'state_vectors'
    rounded_state = round.(state / 5) * 5
    if any(x -> x > E_max, state)
        return Inf
    end

    if any(x -> x < 0, state)
        return Inf
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
    
    return new_state
end

function calculate_cost(lambda_t, action, trip_data, t, num_vehicles)
    total_cost = 0.0

    # Calculate cost for each vehicle
    for i in 1:num_vehicles
        # Extract charging power and trip decision
        P_C_i_t, u_i_t = action[i]
        
        # Extract trip demand for vehicle i at time t (assumed to be stored in trip_data DataFrame)
        d_i_t = trip_data[t, Symbol("Car_$i")]
        
        # Calculate the cost for vehicle i
        cost_i = sum(0.17*(lambda_t .* P_C_i_t .- 300*d_i_t*u_i_t .+ 50*d_i_t*(1-u_i_t)))
        total_cost += cost_i
    end
    
    return total_cost
end


# Function to compute and save DataFrames in an array
function save_action_state_data_to_array()
    data_frames = Array{DataFrame, 1}(undef, T)

    for t in T:-1:1
        lambda_t = prices[prices.Time .== time_stamp[t], :].Price
        results = []

        for state in state_vectors
            for action in action_vectors
                new_state = calculate_new_state(state, action, η, trip_data, t, I)
                new_state_val = state_to_value(new_state, state_vectors, t+1, V)
                cost = calculate_cost(lambda_t, action, trip_data, t, I) 
                total_cost = cost + new_state_val
                push!(results, (time=time_stamp[t], state=state, action=action, new_state=new_state, cost=cost,  new_state_val=new_state_val, total_cost=total_cost))
            end
        end

        df = DataFrame(results)
        data_frames[t] = df  # Store DataFrame in the array at the index corresponding to the time step
        println("Data frame saved for time period $(time_stamp[t]).")
        CSV.write("Results/Tensor/data_frame_time_single_$(time_stamp[t]).csv", df)
    end

    return data_frames
end

# Call the function to process and save the data
data_frames_array = save_action_state_data_to_array()

function save_optimal_actions_states_to_file()
    data_frames = Array{DataFrame, 1}(undef, T)

    for t in T:-1:1
        lambda_t = prices[prices.Time .== time_stamp[t], :].Price
        results = []

        for state in state_vectors
            min_cost = Inf
            optimal_action = nothing
            optimal_new_state = nothing

            for action in action_vectors
                new_state = calculate_new_state(state, action, η, trip_data, t, I)
                new_state_val = state_to_value(new_state, state_vectors, t+1, V)
                cost = calculate_cost(lambda_t, action, trip_data, t, I) + new_state_val
                
                if cost < min_cost
                    min_cost = cost
                    optimal_action = action
                    optimal_new_state = new_state
                end
            end

            # Store the results for the optimal action of this state
            push!(results, (time=time_stamp[t], state=state, action=optimal_action, new_state=optimal_new_state, total_cost=min_cost))
        end

        # Convert results to DataFrame and store in array
        df = DataFrame(results)
        data_frames[t] = df
        println("Optimal actions and costs saved for time period $(time_stamp[t]).")
        CSV.write("Results/Tensor/optimal_actions_states_time_single_$(time_stamp[t]).csv", df)
    end

    return data_frames
end


test_df = save_optimal_actions_states_to_file()


function consolidate_optimal_data()
    # Assuming the number of time periods and the base file path are defined
    num_time_periods = T
    base_file_path = "Results/Tensor/"

    # Initialize empty DataFrames to store consolidated data
    all_optimal_actions = DataFrame()
    all_state_transitions = DataFrame()

    for t in 1:num_time_periods
        # Construct the file path
        file_path = base_file_path * "optimal_actions_states_time_single_$(time_stamp[t]).csv"

        # Read the current time step's DataFrame
        current_df = CSV.read(file_path, DataFrame)

        # Extract columns related to actions and state transitions
        actions_df = select(current_df, :time, :state, :action, :new_state, :total_cost)
        states_df = select(current_df, :time, :state, :new_state, :total_cost)

        # Append the extracted data to the master DataFrames
        all_optimal_actions = vcat(all_optimal_actions, actions_df)
        all_state_transitions = vcat(all_state_transitions, states_df)
    end

    # Optionally, you can sort the data by time if needed
    sort!(all_optimal_actions, :time)
    sort!(all_state_transitions, :time)

    # Save the consolidated data to new CSV files
    CSV.write("Results/all_optimal_actions_single.csv", all_optimal_actions)
    CSV.write("Results/all_optimal_state_transitions_single.csv", all_state_transitions)

    return all_optimal_actions, all_state_transitions
end

# Call the function to execute the consolidation process
consolidate_optimal_data()


# lambda_t = prices[prices.Time .== time_stamp[12], :].Price

# calculate_cost(lambda_t, [(10, 0), (10, 0), (10, 0)], trip_data, 12, 3)


# total_cost = 0.0

# num_vehicles = 3

# action = [(10, 0), (5, 0), (0,1)]
# t=12

# P_C_i_t, u_i_t = action[1]
# # Extract trip demand for vehicle i at time t (assumed to be stored in trip_data DataFrame)
# d_i_t = trip_data[t, Symbol("Car_1")]

# cost_i = sum(0.17*(lambda_t .* P_C_i_t .- (d_i_t * u_i_t)))

# # Calculate cost for each vehicle
# for i in 1:num_vehicles
#     # Extract charging power and trip decision
#     P_C_i_t, u_i_t = action[i]
    
#     # Extract trip demand for vehicle i at time t (assumed to be stored in trip_data DataFrame)
#     d_i_t = trip_data[t, Symbol("Car_$i")]
    
#     # Calculate the cost for vehicle i
#     cost_i = 0.17*(lambda_t .* P_C_i_t .- d_i_t * u_i_t)
#     total_cost += cost_i
# end