export generate_state_space_vector


# Function to generate all possible combinations of charge levels for the three cars
# at increments of 'increment' up to 'max_charge'
function generate_state_space_vector(max_charge, increment)
    # Determine the number of steps for each car
    steps = max_charge ÷ increment + 1
    
    # Initialize an array to hold all combinations as vectors
    state_space_vectors = Vector{Int}[]  # An array of integer vectors
    
    # Generate combinations using nested loops
    for i in 0:increment:max_charge
        for j in 0:increment:max_charge
            for k in 0:increment:max_charge
                push!(state_space_vectors, [i, j, k])
            end
        end
    end
    
    return state_space_vectors
end

function generate_state_space_vector_single(max_charge, increment)
    # Determine the number of steps for each car
    steps = max_charge ÷ increment + 1
    
    # Initialize an array to hold all combinations as vectors
    state_space_vectors = Vector{Int}[]  # An array of integer vectors
    
    # Generate combinations using nested loops
    for i in 0:increment:max_charge
        push!(state_space_vectors, [i])
    end
    
    return state_space_vectors
end

