using IterTools: product

export generate_combinations

# Generate all possible action combinations for three cars
function generate_combinations(actions)
    # Generate the Cartesian product of actions for three cars
    combinations = collect(product(actions, actions, actions))
    # Convert tuple of tuples into vectors
    action_vectors = [vcat(combo...) for combo in combinations]
    return action_vectors
end

# Generate all possible action combinations for three cars
function generate_combinations_single(actions)
    # Generate the Cartesian product of actions for three cars
    combinations = collect(product(actions))
    # Convert tuple of tuples into vectors
    action_vectors = [vcat(combo...) for combo in combinations]
    return action_vectors
end

# Display each combination as a vector
function display_combinations(action_vectors)
    println("Possible action combinations for three cars as vectors:")
    for vec in action_vectors
        println(vec)
    end
end


