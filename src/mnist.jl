using Flux
using Flux: logitcrossentropy, onehotbatch, onecold, @epochs
using Flux.Data: DataLoader
using MLDatasets: MNIST
using Random
using Zygote


function get_dataloaders(batch_size::Int, shuffle::Bool)
    train_x, train_y = MNIST.traindata(Float32)
    test_x, test_y = MNIST.testdata(Float32)

    train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

    train_loader = DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_x, test_y, batchsize=batch_size, shuffle=shuffle)

    return train_loader, test_loader
end


function accuracy(data_loader, model)
    acc_correct = 0
    for (x_batch, y_batch) in data_loader
        batch_size = size(x_batch)[end]
        acc_correct += sum(onecold(model(x_batch)) .== onecold(y_batch)) / batch_size
    end
    return acc_correct / length(data_loader)
end


function create_model(input_dim, dropout_ratio, hidden_dim, classes)
    return Chain(
        Flux.flatten,
        Dense(input_dim, hidden_dim, relu),
        Dropout(dropout_ratio),
        Dense(hidden_dim, classes)
    )
end


function main(num_epochs, batch_size, shuffle, η)
    train_loader, test_loader = get_dataloaders(batch_size, shuffle)

    model = create_model(28*28, 0.2, 128, 10)
    trainable_params = Flux.params(model)

    optimiser = ADAM(η)
    loss(x,y) = logitcrossentropy(model(x), y)

    @epochs num_epochs Flux.train!(loss, trainable_params, train_loader, optimiser)

    testmode!(model)
    @show accuracy(train_loader, model)
    @show accuracy(test_loader, model)
    println("Complete!")
end


if abspath(PROGRAM_FILE) == @__FILE__
    batch_size = 64
    shuffle_data = true
    η = 0.0001
    num_epochs = 1
    main(num_epochs, batch_size, shuffle_data, η)
end
