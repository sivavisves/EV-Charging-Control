using JuMP, Gurobi

export solve_ev_charging_optimization

function solve_ev_charging_optimization(I, J, P_max, E_max, η, λ, D_j, a_j, E_given, penalty_cost)
    model = Model(Gurobi.Optimizer)

    @variable(model, 0 <= P_C[i=1:I] <= P_max)
    @variable(model, E[i=1:I] >= 0)
    @variable(model, Trip[i=1:I, j=1:J], Bin)
    @variable(model, d[i=1:I, j=1:J] >= 0)  # Served demand
    @variable(model, UnservedDemand[j=1:J] >= 0)

    @objective(model, Min, sum(λ * P_C[i] for i in 1:I) + penalty_cost * sum(UnservedDemand[j] for j in 1:J))

    @constraint(model, soc_updates[i=1:I], E[i] == E_given[i] + η * P_C[i] - sum(d[i, j] for j in 1:J))
    @constraint(model, soc_limits[i=1:I], E[i] <= E_max)
    @constraint(model, charge_limit[i=1:I], P_C[i] <= P_max * (1 - sum(Trip[i, j] for j in 1:J)))
    @constraint(model, trip_availability[j=1:J], sum(Trip[i, j] for i in 1:I) <= a_j[j])
    @constraint(model, demand_served[i=1:I, j=1:J], d[i, j] <= D_j[j] * Trip[i, j])
    @constraint(model, unserved_demand_calc[j=1:J], UnservedDemand[j] == D_j[j] - sum(d[i, j] for i in 1:I))

    optimize!(model)
    
    return value.(P_C), value.(E), value.(Trip), objective_value(model)
end

using JuMP, Gurobi

function solve_ev_charging_optimization_time(I, J, P_max, E_max, η, λ, D_j_t, a_j_t, E_initial, penalty_cost)
    model = Model(Gurobi.Optimizer)

    # Time steps defined explicitly
    T = 2  # Total hours of planning, adjust as needed

    @variable(model, 0 <= P_C[i=1:I, t=1:T] <= P_max)
    @variable(model, E[i=1:I, t=0:T] >= 0)  # Include E[.,0] for initial states
    @variable(model, Trip[i=1:I, j=1:J, t=1:T], Bin)
    @variable(model, d[i=1:I, j=1:J, t=1:T] >= 0)  # Served demand
    @variable(model, UnservedDemand[j=1:J, t=1:T] >= 0)

    # Objective function
    @objective(model, Min, sum(λ[t] * P_C[i, t] for i in 1:I, t in 1:T) + 
               penalty_cost * sum(UnservedDemand[j, t] for j in 1:J, t in 1:T))

    # Constraints for each time step
    for t in 1:T
        @constraint(model, [i=1:I], E[i, t] == E[i, t-1] + η * P_C[i, t] - sum(d[i, j, t] for j in 1:J))
        @constraint(model, [i=1:I], E[i, t] <= E_max)
        @constraint(model, [i=1:I], P_C[i, t] <= P_max * (1 - sum(Trip[i, j, t] for j in 1:J)))
        @constraint(model, [j=1:J], sum(Trip[i, j, t] for i in 1:I) <= a_j_t[j, t])
        @constraint(model, [i=1:I, j=1:J], d[i, j, t] <= D_j_t[j, t] * Trip[i, j, t])
        @constraint(model, [j=1:J], UnservedDemand[j, t] == D_j_t[j, t] - sum(d[i, j, t] for i in 1:I))
    end

    # Initial state of charge for each vehicle
    @constraint(model, [i=1:I], E[i, 0] == E_initial[i])

    optimize!(model)
    
    return value.(P_C), value.(E), value.(Trip), objective_value(model)
end


# # Example usage
# I, J = 3, 2
# P_max, E_max, η = 100.0, 200.0, 0.9
# λ = 10.0  # Cost per unit charging power
# D_j = [50, 40]  # Demand for each trip
# a_j = [1, 1]  # Max trips that can be taken
# E_given = [0, 0, 0]  # Initial state of charge for each vehicle

# P_C_d, E_d, trip_d, end_val = solve_ev_charging_optimization(I, J, P_max, E_max, η, λ, D_j, a_j, E_given, 100)
