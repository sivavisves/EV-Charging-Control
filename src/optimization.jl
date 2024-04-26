export state_to_value, calculate_new_state, calculate_cost, save_action_state_data_to_array, save_optimal_actions_states_to_file, consolidate_optimal_data

function state_to_value(state, state_vectors, t, V)
    # This function finds the index of 'state' in 'state_vectors'
    #rounded_state = round.(state / 5) * 5
    if any(x -> x > E_max, state)
        return Inf
    end

    if any(x -> x < 0, state)
        return Inf
    end

    for (idx, s) in enumerate(state_vectors)
        if s == state
            return V[idx, t]
        end
    end
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
    
    return round.(Int64,new_state)
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
        cost_i = sum(0.17.*(lambda_t .* P_C_i_t .- 20*d_i_t * u_i_t .+ (penalty_level*100)*d_i_t*(1-u_i_t)))
        total_cost += cost_i
    end
    
    return total_cost
end


# Function to compute and save DataFrames in an array
function save_action_state_data_to_array(V)
    data_frames = Array{DataFrame, 1}(undef, T)

    for t in T:-1:1
        lambda_t = prices[prices.Time .== time_stamp[t], :].Price
        results = []

        for state in state_vectors
            cost_min = Inf
            for action in action_vectors
                state_index = findall(x -> x == state, state_vectors)[1]
                new_state = calculate_new_state(state, action, η, trip_data, t, I)
                new_state_val = state_to_value(new_state, state_vectors, t+1, V)
                cost = calculate_cost(lambda_t, action, trip_data, t, I) 
                total_cost = cost + new_state_val
                if total_cost < cost_min
                    V[state_index, t] = total_cost
                end
                push!(results, (time=time_stamp[t], state=state, action=action, new_state=new_state, cost=cost,  new_state_val=new_state_val, total_cost=total_cost))
            end
        end

        df = DataFrame(results)
        data_frames[t] = df  # Store DataFrame in the array at the index corresponding to the time step
        println("Data frame saved for time period $(time_stamp[t]).")
        CSV.write("Results/Tensor/data_frame_time_$(time_stamp[t]).csv", df)
    end

    return data_frames
end


function save_optimal_actions_states_to_file(V)
    data_frames = Array{DataFrame, 1}(undef, T)

    for t in T:-1:1
        lambda_t = prices[prices.Time .== time_stamp[t], :].Price
        results = []

        for state in state_vectors
            min_cost = Inf
            optimal_action = nothing
            optimal_new_state = nothing
            new_state_val_opt = nothing
            total_cost_opt = Inf  # Initialize total_cost_opt here

            for action in action_vectors
                state_index = findall(x -> x == state, state_vectors)[1]
                new_state = calculate_new_state(state, action, η, trip_data, t, I)
                new_state_val = state_to_value(new_state, state_vectors, t+1, V)
                cost = calculate_cost(lambda_t, action, trip_data, t, I) 
                total_cost = cost + new_state_val  # Define total_cost within this scope
                
                if total_cost < min_cost  # Compare with total_cost instead of cost
                    min_cost = cost  # Store the total_cost if it's the smallest found
                    optimal_action = action
                    optimal_new_state = new_state
                    new_state_val_opt = new_state_val
                    V[state_index, t] = total_cost
                    total_cost_opt = total_cost  # Update total_cost_opt
                end
            end
            push!(results, (time=time_stamp[t], state=state, action=optimal_action, new_state=optimal_new_state, cost=min_cost, new_state_val_opt=new_state_val_opt, total_cost_opt=total_cost_opt))
        end

        # Convert results to DataFrame and store in array
        df = DataFrame(results)
        data_frames[t] = df
        println("Optimal actions and costs saved for time period $(time_stamp[t]).")
        CSV.write("Results/Tensor/optimal_actions_states_time_$(time_stamp[t]).csv", df)
    end

    return data_frames
end


function consolidate_optimal_data()
    # Assuming the number of time periods and the base file path are defined
    num_time_periods = T
    base_file_path = "Results/Tensor/"

    # Initialize empty DataFrames to store consolidated data
    all_optimal_actions = DataFrame()
    all_state_transitions = DataFrame()

    for t in 1:num_time_periods
        # Construct the file path
        file_path = base_file_path * "optimal_actions_states_time_$(time_stamp[t]).csv"

        # Read the current time step's DataFrame
        current_df = CSV.read(file_path, DataFrame)

        # Extract columns related to actions and state transitions
        actions_df = select(current_df, :time, :state, :action, :new_state, :cost, :new_state_val_opt, :total_cost_opt)
        states_df = select(current_df, :time, :state, :new_state, :total_cost_opt)

        # Append the extracted data to the master DataFrames
        all_optimal_actions = vcat(all_optimal_actions, actions_df)
        all_state_transitions = vcat(all_state_transitions, states_df)
    end

    # Optionally, you can sort the data by time if needed
    sort!(all_optimal_actions, :time)
    sort!(all_state_transitions, :time)

    # Save the consolidated data to new CSV files
    CSV.write("Results/all_optimal_actions.csv", all_optimal_actions)
    CSV.write("Results/all_optimal_state_transitions.csv", all_state_transitions)

    return all_optimal_actions, all_state_transitions
end

function calculate_cost_lookahead(lambda_t, action, trip_data, t, num_vehicles)
    total_cost = 0.0

    # Calculate cost for each vehicle
    for i in 1:num_vehicles
        # Extract charging power and trip decision
        P_C_i_t, u_i_t = action[i]
        
        # Extract trip demand for vehicle i at time t (assumed to be stored in trip_data DataFrame)
        d_i_t = trip_data[t, Symbol("Car_$i")]

        typeof(d_i_t)
        
        # Calculate the cost for vehicle i
        cost_i = lambda_t * P_C_i_t - 20*d_i_t * u_i_t + (penalty_level*100)*d_i_t*(1-u_i_t)
        total_cost += cost_i
    end
    
    return total_cost
end



function state_to_value_lookahead(state, t, lambda_t, trip_data)
    # This function finds the index of 'state' in 'state_vectors'
    #rounded_state = round.(state / 5) * 5
    if any(x -> x > E_max, state)
        return Inf
    end

    if any(x -> x < 0, state)
        return Inf
    end

    final_cost = Inf
    final_action = []
    for action in action_vectors
        if t == 13
            return 0
            break
        end

        cost = calculate_cost_lookahead(lambda_t, action, trip_data, t, I) 

        if cost < final_cost
            final_cost = cost
            final_action = action
        end 
    end
    return final_cost
end

# Function to compute and save DataFrames in an array
function save_action_state_data_to_array_lookahead()
    data_frames = Array{DataFrame, 1}(undef, T)

    for t in 1:T
        lambda_t = average_prices[average_prices.time .== time_stamp[t], :].average_price[1]
        if t == 12
            lambda_t_next = 0;
        else
            lambda_t_next = average_prices[average_prices.time .== time_stamp[t+1], :].average_price[1]
        end
        
        results = []

        for state in state_vectors
            for action in action_vectors
                new_state = calculate_new_state(state, action, η, trip_data, t, I)
                new_state_val = state_to_value_lookahead(new_state, t+1, lambda_t_next, trip_data)
                cost = calculate_cost_lookahead(lambda_t, action, trip_data, t, I) 
                total_cost = cost + new_state_val[1]
                
                push!(results, (time=time_stamp[t], state=state, action=action, new_state=new_state, cost=cost, new_state_cost = new_state_val, total_cost=total_cost))
            end
        end

        df = DataFrame(results)
        data_frames[t] = df  # Store DataFrame in the array at the index corresponding to the time step
        println("Data frame saved for time period $(time_stamp[t]).")
        CSV.write("Results/Tensor/data_frame_time_lookahead_$(time_stamp[t]).csv", df)
    end

    return data_frames
end