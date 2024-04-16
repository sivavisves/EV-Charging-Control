using JuMP, Gurobi

# Constants and time horizon extension
T = 12  # Total number of time periods extended to 12 hours
I = 3   # Number of vehicles
J = 2   # Number of trips (Trip A and Trip B)
P_max = 10.0  # Maximum charging power
E_max = 100.0  # Maximum state of charge
η = 0.9  # Charging efficiency
λ_t = [0.1 for t in 1:T]  # Cost rate per kWh charged

# Trip demands and availability data, repeated every 3 hours
D_j_t = [
    [4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3],  # Trip A demand repeated
    [6, 7, 0, 6, 7, 0, 6, 7, 0, 6, 7, 0]   # Trip B demand repeated
]

a_j_t = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Trip A availability (1 = Yes, 0 = No)
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]   # Trip B availability (1 = Yes, 0 = No)
]

