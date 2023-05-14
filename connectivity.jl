using Plots
Plots.scalefontsizes(1.5)
using LaTeXStrings

# Results for the sort task

x = [10, 15, 20, 5]
A_mean = [0.6219999849796295, 0.5639999806880951, 0.5240000009536743, 0.7339999973773956]
A_std = [0.14274452060331327, 0.08064737258528903, 0.11757551232562345, 0.20362710572307874]
B_mean = [0.75, 0.6659999668598175, 0.6959999918937683, 0.8299999833106995]
B_std = [0.0, 0.12306095037845727, 0.16414627437610643, 0.0]
C = [0.5800000131130219, 0.5800000131130219, 0.6200000047683716, 0.6599999964237213]

print(size(x),size(A_mean), size(A_std), size(B_mean), size(B_std), size(C))


# Create plot with error bars
scatter(
    x,   # x values are the indices
    A_mean,             # y values are the mean vector
    yerr = A_std,       # error bars are the standard deviation vector
    label = "GNN",     # set label for the data points
    xlabel = "Number of vertices",  # label x axis
    ylabel = "Accuracy",   # label y axis
    title = "", # add title
    markersize = 12, # set marker size
    markerstrokewidth = 0,
    color = :blue,
    fmt = :pdf,         # set output format
    size = (1000, 800),  # set size of plot
    legend=:topright,
    legendfont = font(16,"Computer Modern"), # set legend font size
    guidefont = font(16,"Computer Modern"),  # set guide font size
    tickfont = font(16,"Computer Modern"),   # set tick font size
)

scatter!(
    x,   # x values are the indices
    B_mean,             # y values are the mean vector
    yerr = B_std,       # error bars are the standard deviation vector
    label = "RGraphC",   # set label for B data points
    markersize = 12,     # set marker size
    markerstrokewidth = 0,
    color = :green,       # set color for B data points
)

scatter!(
     x,   # x values are the indices
     C,             # y values are the mean vector
     label = "Constant classifier",   # set label for B data points
     markersize = 12,     # set marker size
     color = :red,       # set color for B data points
     markerstrokewidth = 0
 )

savefig("connectivity-plot.pdf")
