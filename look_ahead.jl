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
average_prices = CSV.read("Price_data/extracted_prices_average.csv", DataFrame)

#price adjustment
average_prices.average_price = average_prices.average_price*0.001

# Load trip data
trip_data = CSV.read("Trip_data/trip_data.csv", DataFrame)

# Generate all possible action combinations for three cars
actions = [(10, 0), (5, 0), (0,1)]
action_vectors = generate_combinations(actions)

# Generate all possible combinations of state
state_vectors = generate_state_space_vector(E_max, 5)

data_frames = Array{DataFrame, 1}(undef, T)



save_action_state_data_to_array_lookahead()


data_frames = Array{DataFrame, 1}(undef, T)

for t in 1:T
    lambda_t = average_prices[average_prices.time .== time_stamp[t], :].average_price[1]
    results = []
    optimal_action = []
    optimal_new_state = []
    current_cost = 0.0
    new_state_cost = 0.0
    total_cost = 0.0
    optimal_new_state_action = []

    for state in state_vectors
        min_cost = Inf
        for action in action_vectors
            new_state = calculate_new_state(state, action, η, trip_data, t, I)
            new_state_val = state_to_value_lookahead(new_state, t+1, lambda_t, trip_data)
            cost = calculate_cost_lookahead(lambda_t, action, trip_data, t, I) 
            total_cost = cost + new_state_val

            if total_cost < min_cost
                min_cost = total_cost
                optimal_action = action
                optimal_new_state = new_state
                current_cost = cost
                new_state_cost = new_state_val
            end
        end
        push!(results, (time=time_stamp[t], state=state, action=optimal_action, new_state=optimal_new_state, cost=current_cost, new_state_cost = new_state_cost, total_cost=total_cost))
    end

    df = DataFrame(results)
    data_frames[t] = df  # Store DataFrame in the array at the index corresponding to the time step
    println("Data frame saved for time period $(time_stamp[t]).")
    CSV.write("Results/Tensor/Optimal_data_frame_time_lookahead_$(time_stamp[t]).csv", df)
end




for t in 1:T
    lambda_t = average_prices[average_prices.time .== time_stamp[t], :].average_price[1]
    if t == 12
        lambda_t_next = 0
    else
        lambda_t_next = average_prices[average_prices.time .== time_stamp[t+1], :].average_price[1]
    end
    min_costs = Dict()

    for state in state_vectors
        for action in action_vectors
            new_state = calculate_new_state(state, action, η, trip_data, t, I)
            new_state_val = state_to_value_lookahead(new_state, t+1, lambda_t_next, trip_data)
            cost = calculate_cost_lookahead(lambda_t, action, trip_data, t, I) 
            total_cost = cost + new_state_val

            # Check if the current state has a recorded cost or if the current cost is lower than the recorded one
            if !haskey(min_costs, state) || total_cost < min_costs[state][:total_cost]
                min_costs[state] = (time=time_stamp[t], state=state, action=action, new_state=new_state, cost=cost, new_state_val=new_state_val, total_cost=total_cost)
            end
        end
    end

    # Convert the dictionary to a DataFrame and store in data_frames
    data_frames[t] = DataFrame(values(min_costs))
end

# Assume data_frames is the Array of DataFrames generated from your loop
consolidated_df = vcat(data_frames...)

# Optionally, you can sort the consolidated DataFrame by 'time' and 'state' for better readability or further analysis
sort!(consolidated_df, [:time, :state])

# Now, `consolidated_df` contains all the rows from each time step's DataFrame in one place
# save consolidated_df to a CSV file
CSV.write("Results/all_optimal_action_lookahead.csv", consolidated_df)



# --------------------Data Consolidation--------------------

# Load and process data
df = consolidated_df

# Helper function to round array elements
function round_to_nearest_5(x::Float64)
    return 5 * round(Int, x / 5)
end

# Helper function to parse and round array elements
function parse_and_round_array_str(array_str::AbstractString)
    # Parse the string to a Float64 array, round to the nearest multiple of 5, and convert to Int
    float_array = parse.(Float64, split(array_str[2:end-1], ", "))
    return round_to_nearest_5.(float_array)
end

# df.state = [parse_and_round_array_str(s) for s in df.state]
# df.new_state = [parse_and_round_array_str(s) for s in df.new_state]

# Define the time horizon and consolidate transitions
T = 12
time_stamp = collect(8:19)


# intial DataFrame
df_test = df[df.time .== time_stamp[1], :]

for t in 1:T
    current_state = df[df.time .== time_stamp[t], :]
    state_col = Vector{Any}(undef, size(current_state, 1))
    new_state_col = Vector{Any}(undef, size(current_state, 1))
    action_col = Vector{Any}(undef, size(current_state, 1))
    cost_col = Vector{Any}(undef, size(current_state, 1))
    new_state_val_col = Vector{Any}(undef, size(current_state, 1))
    total_cost_col = Vector{Any}(undef, size(current_state, 1))

    for i in 1:size(current_state, 1)
        new_state = current_state.new_state[i]
        next_state = df[df.time .== time_stamp[t + (t < T ? 1 : 0)], :]
        for j in 1:size(next_state, 1)
            if next_state.state[j] == new_state
                # Save data to corresponding columns
                state_col[i] = new_state
                new_state_col[i] = next_state.new_state[j]
                action_col[i] = next_state.action[j]
                cost_col[i] = next_state.cost[j]
                new_state_val_col[i] = next_state.new_state_val[j]
                total_cost_col[i] = next_state.total_cost[j]
            end
        end
    end

    # Assign columns to DataFrame
    df_test[!, Symbol("state_t", t)] = state_col
    df_test[!, Symbol("action_t", t)] = action_col
    df_test[!, Symbol("new_state_t", t)] = new_state_col
    df_test[!, Symbol("cost_t", t)] = cost_col
    df_test[!, Symbol("new_state_val_t", t)] = new_state_val_col
    df_test[!, Symbol("total_cost_t", t)] = total_cost_col
end

df_test

# Save the consolidated data to a new CSV file
CSV.write("Results/consolidated_optimal_state_lookahead.csv", df_test)


