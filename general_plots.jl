using CSV, DataFrames, Plots, PrettyTables

#load extracted prices
df_price = CSV.read("Price_data/extracted_prices.csv", DataFrame)

# extract scenario 1
df_price_s1 = df_price[df_price.Scenario .== 1, :]

price_plot = plot(df_price_s1.Time, df_price_s1.Price, label="Scenario 1", xlabel="Time (Hour)", ylabel="Log of Price (\$)", title="Electricity Price Scenarios over time", lw=2, legend=:outerbottom, yaxis=:log, xticks=8:1:19, legend_columns=4);


for i in 2:17
    df_price_si = df_price[df_price.Scenario .== i, :]
    plot!(df_price_si.Time, df_price_si.Price, label="Scenario $i", lw=2)
end

display(price_plot)

savefig(price_plot, "Plots/price_plot.png")