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

struct FFNetwork
    fc_1
    dropout
    fc_2
    FFNetwork(
        input_dim::Int, hidden_dim::Int, dropout_ratio::Float32, num_classes::Int
    ) = new(
        Dense(input_dim, hidden_dim, relu),
        Dropout(dropout_ratio),
        Dense(hidden_dim, num_classes),
    )
end

function (net::FFNetwork)(x)
    x = Flux.flatten(x)
    return net.fc_2(net.dropout(net.fc_1(x)))
end

function cross_entropy_loss(model, x, y)
    ŷ = model(x)
    return logitcrossentropy(model(x), y)
end

function main(num_epochs, batch_size, shuffle, η)
    train_loader, test_loader = get_dataloaders(batch_size, shuffle)

    model = FFNetwork(28*28, 128, 0.2f0, 10)
    trainable_params = Flux.params(model.fc_1, model.fc_2)
    optimiser = ADAM(η)

    for epoch = 1:num_epochs
        acc_loss = 0.0
        for (x_batch, y_batch) in train_loader
            loss, back = pullback(trainable_params) do
                cross_entropy_loss(model, x_batch, y_batch)
            end
            # Feed the pullback 1 to obtain the gradients and update then model parameters
            gradients = back(1f0)
            Flux.Optimise.update!(optimiser, trainable_params, gradients)
            acc_loss += loss
        end
        avg_loss = acc_loss / length(train_loader)
        println("Epoch: $epoch - loss: $avg_loss")
    end

    testmode!(model)
    @show accuracy(train_loader, model)
    @show accuracy(test_loader, model)
    println("Complete!")
end


if abspath(PROGRAM_FILE) == @__FILE__
    batch_size = 64
    shuffle_data = true
    η = 0.0001
    num_epochs = 5
    main(num_epochs, batch_size, shuffle_data, η)
end
