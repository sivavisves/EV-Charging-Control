using HDF5

export read_extract_prices

# Function to read and extract price data
function read_extract_prices(filepath, num_scenarios)
    # Open the HDF5 file
    file = h5open(filepath, "r")

    # Read the dataset assuming it is named "prices"
    # You may need to adjust "prices" to the correct dataset name

    # create DataFrames with scenarios and prices
    prices = DataFrame(Time = Int[], Scenario = Int[], Price = Float64[])
    time = collect(8:19)

    for i in 1:num_scenarios
        prices_file = read(file, "scenario_$i");
        extracted_prices = prices_file["price"][8:19];
        for j in eachindex(extracted_prices)
            push!(prices, (time[j], i, extracted_prices[j]))
        end
    end

    return prices
end

# filepath = "Price_data/price_scenario_data.h5"

# file = h5open(filepath, "r")
# scenario_1 = read(file, "scenario_1")
# x = scenario_1["price"][8:19]

# test_prices = read_extract_prices(filepath, 17)

# # extract only 8am prices
# eight_am_prices = test_prices[test_prices.Time .== 8, :]