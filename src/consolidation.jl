using CSV, DataFrames, Plots, PrettyTables

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
    df_test[!, Symbol("state_t", time_stamp[t+1])] = state_col
    df_test[!, Symbol("action_t", time_stamp[t+1])] = action_col
    df_test[!, Symbol("new_state_t", time_stamp[t+1])] = new_state_col
    df_test[!, Symbol("cost_t", time_stamp[t+1])] = cost_col
    df_test[!, Symbol("new_state_val_cost_t", time_stamp[t+1])] = new_state_val_cost
    df_test[!, Symbol("total_cost_t", time_stamp[t+1])] = total_cost_opt
end

df_test


# Save the consolidated data to a new CSV file
#CSV.write("Results/consolidated_optimal_state.csv", df_test)


#---------------------------plotting--------------------------------



# remove time columns
df_test = select!(df_test, Not([:time]))

# change the name of state to state_t8
rename!(df_test, :state => :state_t8)
rename!(df_test, :new_state => :new_state_t8)
rename!(df_test, :action => :action_t8)
rename!(df_test, :cost => :cost_t8)
rename!(df_test, :new_state_val_opt => :new_state_val_cost_t8)
rename!(df_test, :total_cost_opt => :total_cost_t8)


states_of_interest = [[0,0,0], [5,5,5], [10,10,10], [15,15,15], [20,20,20]]

x = filter(row -> row.state_t8 == [5,5,5], df_test)


# only take cost information from x
cost_1 = zeros(12)

# define df_cost
df_cost = DataFrame()

for i in eachindex(states_of_interest)
    y = filter(row -> row.state_t8 == states_of_interest[i], df_test)
    cost_1 = zeros(12)
    for j in 1:12
        cost_1[j] = y[:, Symbol("cost_t", time_stamp[j])][1]
    end
    df_cost[!, Symbol("cost_$(i)")] = cost_1
end

cost_plot = plot(time_stamp ,df_cost[:, "cost_1"], label = "State: [0,0,0]", xlabel = "Time", ylabel = "Cost", title = "Cost vs Time", lw = 2, xticks = 8:1:19);

for i in 2:5
    plot!(time_stamp ,df_cost[:, "cost_$(i)"], label = "State: $(states_of_interest[i])", lw = 2)
end

cost_plot

savefig(cost_plot, "Results/DP_cost_plot.png")


# Create a table with the data used in the plot
table_data = hcat(time_stamp, [df_cost[:, Symbol("cost_$(i)")] for i in 1:length(states_of_interest)]...)
table_header = ["Time", "State: [0,0,0]", "State: [5,5,5]", "State: [10,10,10]", "State: [15,15,15]", "State: [20,20,20]"]
pretty_table(table_data, header = table_header, backend = Val(:latex)) 


sum(df_cost[:, "cost_1"])
sum(df_cost[:, "cost_2"])
sum(df_cost[:, "cost_3"])
sum(df_cost[:, "cost_4"])
sum(df_cost[:, "cost_5"])

