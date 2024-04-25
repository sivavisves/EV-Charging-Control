using CSV, DataFrames

# Load and process data
df = CSV.read("Results/all_optimal_actions.csv", DataFrame)

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

df.state = [parse_and_round_array_str(s) for s in df.state]
df.new_state = [parse_and_round_array_str(s) for s in df.new_state]

# Define the time horizon and consolidate transitions
T = 12
time_stamp = collect(8:19)


# state_columns = [:state_1; Symbol.("state_t", 2:T)]
# new_state_columns = [:state_1; Symbol.("state_t", 2:T)]
# action_columns = [:action_1; Symbol.("action_t", 2:T)]
# cost_columns = [:cost_1; Symbol.("cost_t", 2:T)]

# intial DataFrame
df_test = df[df.time .== time_stamp[1], :]
time_stamp[2]
for t in 1:T
    if t ==12
        println("Complete")
        break
    end
    current_state = df[df.time .== time_stamp[t], :]
    state_col = Vector{Vector{Int64}}(undef, size(current_state, 1))
    new_state_col = Vector{Vector{Int64}}(undef, size(current_state, 1))
    action_col = Vector{Any}(undef, size(current_state, 1))
    cost_col = Vector{Any}(undef, size(current_state, 1))
    new_state_val_cost = Vector{Any}(undef, size(current_state, 1))
    total_cost_opt = Vector{Any}(undef, size(current_state, 1))

    for i in 1:size(current_state, 1)
        new_state = current_state.new_state[i]
        next_state = df[df.time .== time_stamp[t + 1], :]
        for j in 1:size(next_state, 1)
            if next_state.state[j] == new_state
                # Save data to corresponding columns
                state_col[i] = new_state
                new_state_col[i] = next_state.new_state[j]
                action_col[i] = next_state.action[j]
                cost_col[i] = next_state.cost[j]
                new_state_val_cost[i] = next_state.new_state_val_opt[j]
                total_cost_opt[i] = next_state.total_cost_opt[j]
                break
            end
            if j == 125
                println("State not found")
                println("Time period: ", time_stamp[t])
                println("Current state: ", current_state.state[i])
                println("New state: ", new_state)
            end
        end
    end

    # Assign columns to DataFrame
    df_test[!, Symbol("state_t", time_stamp[t])] = state_col
    df_test[!, Symbol("action_t", time_stamp[t])] = action_col
    df_test[!, Symbol("new_state_t", time_stamp[t])] = new_state_col
    df_test[!, Symbol("cost_t", time_stamp[t])] = cost_col
    df_test[!, Symbol("new_state_val_cost_t", time_stamp[t])] = new_state_val_cost
    df_test[!, Symbol("total_cost_t", time_stamp[t])] = total_cost_opt
end

df_test

df_test[:, 9:20]

# Save the consolidated data to a new CSV file
CSV.write("Results/consolidated_optimal_state.csv", df_test)


for t in 1:T
    println("Time period: ", time_stamp[t])
    println("Unique states: ", length(unique(df_test[!, Symbol("state_t", t)])))
end

t=2
new_state = [0,0,5]
next_state = df[df.time .== time_stamp[t + 1], :]
current_state = df[df.time .== time_stamp[t], :]
for j in 1:size(next_state, 1)
    if next_state.state[j] == new_state
        println("Found")
    end
end


df[df.time .== time_stamp[4], :]
