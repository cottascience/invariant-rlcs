using Plots
Plots.scalefontsizes(1.5)
using LaTeXStrings

# Results vectors
x = []
A_mean = [3.5, 2.0, 5.1, 4.3]
A_std = [0.2, 0.4, 0.1, 0.3]
B_mean = [1.2, 4.5, 3.1, 2.9]
B_std = [0.3, 0.1, 0.2, 0.4]
C = [1, 4, 3, 2]

# Create plot with error bars
scatter(
    x,   # x values are the indices
    A_mean,             # y values are the mean vector
    yerr = A_std,       # error bars are the standard deviation vector
    label = "Method1",     # set label for the data points
    xlabel = "Set size",  # label x axis
    ylabel = "Accuracy",   # label y axis
    title = "", # add title
    markersize = 10,     # set marker size
    color = :green,
    fmt = :pdf,         # set output format
    size = (800, 600),  # set size of plot
    legend=:bottomright,
    legendfont = font(16,"Computer Modern"), # set legend font size
    guidefont = font(16,"Computer Modern"),  # set guide font size
    tickfont = font(16,"Computer Modern"),   # set tick font size
)

scatter!(
    1:length(B_mean),   # x values are the indices
    B_mean,             # y values are the mean vector
    yerr = B_std,       # error bars are the standard deviation vector
    label = "Method2",   # set label for B data points
    markersize = 10,     # set marker size
    color = :purple,       # set color for B data points
)

scatter!(
     1:length(C),   # x values are the indices
     C,             # y values are the mean vector
     label = "Constant classifier",   # set label for B data points
     markersize = 10,     # set marker size
     color = :red,       # set color for B data points
 )

savefig("sort-plot.pdf")
