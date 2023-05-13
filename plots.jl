using Plots
Plots.scalefontsizes(1.5)
using LaTeXStrings

# Results for the sort task
x = [101, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 5, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 7, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 9, 91, 93, 95, 97, 99]
A_mean = [0.532999986410141, 0.5059999763965607, 0.591999989748001, 0.574999988079071, 0.5579999804496765, 0.5549999862909317, 0.542999991774559, 0.5539999961853027, 0.5179999709129334, 0.5259999811649323, 0.5189999788999557, 0.5680000007152557, 0.5139999836683273, 0.5269999831914902, 0.560999971628189, 0.48499998152256013, 0.5619999796152115, 0.5269999802112579, 0.5439999878406525, 0.5069999873638154, 0.5089999973773957, 0.5609999865293502, 0.5049999803304672, 0.5069999903440475, 0.4999999821186066, 0.5419999957084656, 0.5339999824762345, 0.5079999893903733, 0.5120000004768371, 0.49899997711181643, 0.4879999905824661, 0.5049999982118607, 0.6169999867677689, 0.5249999791383744, 0.5120000004768371, 0.5239999949932098, 0.5079999834299087, 0.55799999833107, 0.5019999742507935, 0.4999999850988388, 0.5089999794960022, 0.49899999499320985, 0.5009999871253967, 0.5489999830722809, 0.5089999854564666, 0.49099999070167544, 0.524999988079071, 0.5009999990463256, 0.5049999833106995]
A_std = [0.04337049826421517, 0.09789789562701057, 0.10156770456447771, 0.06545990178078522, 0.06690291258226812, 0.08464632309269597, 0.053488315449378925, 0.07323933774403973, 0.05617827606019752, 0.08114185150882679, 0.07395267826560453, 0.06720120283339164, 0.07735632185669614, 0.023685437240975365, 0.08251666249578486, 0.06917368855413736, 0.08047358710704086, 0.07308213360917154, 0.04758150748019698, 0.03769614647768043, 0.018138353322188314, 0.05990826601467982, 0.02578758376954567, 0.06388270833073584, 0.025690454577639747, 0.05741080459692921, 0.03322649336358635, 0.03487119401652953, 0.028565719022431463, 0.05974110911870159, 0.057061371206127746, 0.050049984159192075, 0.0907799462452972, 0.056258319589169305, 0.024413111387723882, 0.03666059791164842, 0.04812483164770286, 0.006000012159347534, 0.0359999934832266, 0.019999995827674866, 0.03884585174006144, 0.008306625986885132, 0.022113353487218004, 0.09004998529541076, 0.05299999320282131, 0.02699998815854427, 0.04432832530544351, 0.0029999971389770507, 0.020615522923838482]
B_mean = [0.6529999732971191, 0.6699999570846558, 0.6589999675750733, 0.6499999761581421, 0.7379999995231629, 0.6870000004768372, 0.6349999904632568, 0.5229999780654907, 0.6259999990463256, 0.6109999895095826, 0.6009999811649323, 0.5109999895095825, 0.5259999662637711, 0.5769999861717224, 0.537999963760376, 0.6190000057220459, 0.6719999849796295, 0.622000002861023]
B_std = [0.004582571324671096, 0.0, 0.005385159671441138, 0.00894426338007676, 0.004000020027160644, 0.004582571324671096, 0.006708197535057569, 0.004582571324671096, 0.00663324325475104, 0.009433999934666051, 0.007000016314658307, 0.006999993324279785, 0.02653299188905473, 0.006403118130937718, 0.0039999961853027345, 0.0029999971389770507, 0.008717816921636097, 0.005999994277954102]
C = [0.52, 0.54, 0.61, 0.5, 0.51, 0.52, 0.51, 0.55, 0.59, 0.54, 0.52, 0.61, 0.5700000000000001, 0.53, 0.6, 0.58, 0.52, 0.52, 0.52, 0.51, 0.5, 0.52, 0.52, 0.51, 0.52, 0.55, 0.52, 0.55, 0.5, 0.54, 0.55, 0.55, 0.54, 0.53, 0.5, 0.5, 0.52, 0.56, 0.54, 0.52, 0.54, 0.5, 0.51, 0.57, 0.52, 0.53, 0.52, 0.5, 0.52]

print(size(x),size(A_mean), size(A_std), size(B_mean), size(B_std), size(C))

# Create plot with error bars
scatter(
    x,   # x values are the indices
    A_mean,             # y values are the mean vector
    yerr = A_std,       # error bars are the standard deviation vector
    label = "Deep Sets",     # set label for the data points
    xlabel = "Set size",  # label x axis
    ylabel = "Accuracy",   # label y axis
    title = "", # add title
    markersize = 8, # set marker size
    markerstrokewidth = 0,
    color = :green,
    fmt = :pdf,         # set output format
    size = (1000, 800),  # set size of plot
    legend=:bottomright,
    legendfont = font(16,"Computer Modern"), # set legend font size
    guidefont = font(16,"Computer Modern"),  # set guide font size
    tickfont = font(16,"Computer Modern"),   # set tick font size
)

scatter!(
    x,   # x values are the indices
    B_mean,             # y values are the mean vector
    yerr = B_std,       # error bars are the standard deviation vector
    label = "RSetC",   # set label for B data points
    markersize = 8,     # set marker size
    markerstrokewidth = 0,
    color = :purple,       # set color for B data points
)

scatter!(
     x,   # x values are the indices
     C,             # y values are the mean vector
     label = "Constant classifier",   # set label for B data points
     markersize = 8,     # set marker size
     color = :red,       # set color for B data points
     markerstrokewidth = 0
 )

savefig("sort-plot.pdf")
