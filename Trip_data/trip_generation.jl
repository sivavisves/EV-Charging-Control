using CSV, DataFrames

# Constants and parameters
T = 12  # Total number of time periods (12 hours)
I = 3   # Number of vehicles

# Generate sample trip demands
# We'll use random values for demand, you can specify any range or logic here
function generate_trip_data(T, I)
    # Create a DataFrame to store trip data
    trip_data = DataFrame()
    for i in 1:I
        trip_data[!, Symbol("Car_$i")] = rand([0,5,10], T)  # Random demands between 1 and 10 kWh
    end
    return trip_data
end

# Generate data
trip_data = generate_trip_data(T, I)

# Display the data to check
println(trip_data)

### Step 3: Save the Data to a CSV File

output_filepath = "Trip_data/trip_data.csv"
CSV.write(output_filepath, trip_data)
