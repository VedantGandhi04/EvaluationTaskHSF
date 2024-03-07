# Importing the required Julia packages
using Flux, Statistics, MLDatasets, DataFrames, OneHotArrays, Plots, CSV

# Importing the dataset
df = CSV.read("/home/vedant/JULIA/flux/dataset.csv", DataFrame)
 
# Splitting the dataset into train and test sets in the ratio 4:1
split_index = Int(0.8 * size(df, 1))
df_train = df[1:split_index, :]
df_test = df[(split_index + 1):end, :]
 
 
#= Seperating the features and labels : 
 - X_train/test consists of 3 coordinate parameters(x, y, z) 
 - Y_train/test consists of 1 label which takes values ('s', 'b') =#
# OneHotEncoding the labels - to convert this data into a form that can be fed to a machine learning model
X_train = Matrix(df_train[:, 1:3])
X_train = X_train'
y_train = hcat(Flux.onehotbatch((df_train[:, 4]),  ["s", "b"])[1, :])'
 
X_test = Matrix(df_test[:, 1:3])
X_test = X_test'
y_test = hcat(Flux.onehotbatch((df_test[:, 4]),  ["s", "b"])[1, :])'
 
#Visualizing the data 
Plot_data = scatter(df_train[:, 3], df_train[:, 2], df_train[:, 1],
    xlabel = "Z",
    ylabel = "Y",
    zlabel = "X",
    legend = false,
    color = [:red, :blue],
    markersize = 1
)
display(Plot_data)
  
#= Building a model
Given that our model has three inputs (3 features in every data point), and one output (1 or 0), we don't have to initialize the parameters, Flux does it for us. 
We use the sigmoid function because its value ranges from 0 to 1. =#
flux_model = Chain(
    Dense(3, 1, σ)
)
parameters = Flux.params(flux_model)

 
#= Loss function
We use binarycrossentropy as it return the binary cross-entropy loss directly. =#
function flux_loss(flux_model, x, y)
    ŷ = flux_model(x)
    Flux.binarycrossentropy(ŷ, y)
end;
 
 
#= Accuracy function
We compare the predictions of the model with the actual values and calculate its accuracy.
Since the output is in decimal ranging from 0 to 1, we convert it to binary and then to Int64. =#
flux_accuracy(x, y) = mean(Int.(flux_model(x) .>=0.5) .== y);
 
#= Training the model
'dLdm' includes both the gradients with respect to the weights and biases, 
and you use this information to update both sets of parameters during the training process.  =#
function train_flux_model!(flux_model, X, y)
    dLdm, _, _ = gradient(flux_loss, flux_model, X, y)
    @. flux_model[1].weight -= 0.1 * dLdm[:layers][1][:weight]
    @. flux_model[1].bias -= 0.1 * dLdm[:layers][1][:bias]
end;
 
#= The loop can be adjusted as per the user's needs, and the conditions can be specified in plain Julia. Here we will train the model for a maximum of 50 epochs, but to ensure that the model does not overfit, we will break as soon as our accuracy value crosses or becomes equal to 0.98.
To keep track of loss and accuracy, for each epoch we store it in loss_history and accuracy_history.  =#
loss_history = Float64[]
accuracy_history = Float64[]
for i = 1:50
    train_flux_model!(flux_model, X_train, y_train);
 
    push!(loss_history, flux_loss(flux_model, X_train, y_train))
    push!(accuracy_history, flux_accuracy(X_train, y_train))
 
    flux_accuracy(X_train, y_train) >= 0.98 && break
end 
 
 
# Visualizing the loss and accuracy of our model
Plot_loss = plot(1:length(loss_history), loss_history, label="Loss", xlabel="Epochs", ylabel="Loss", legend=:topright)
display(Plot_loss)
Plot_accuracy = plot!(1:length(accuracy_history), accuracy_history, label="Accuracy", xlabel="Epochs", ylabel="Accuracy", legend=:topright)
display(Plot_accuracy)
 

# Testing our model on test_set
@show flux_accuracy(X_test, y_test);
@show flux_loss(flux_model, X_test, y_test)
