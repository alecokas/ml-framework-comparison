import tensorflow as tf


def get_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


def create_model(input_dims, hidden_layer_dim, dropout_ratio, num_classes):
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_dims),
        tf.keras.layers.Dense(hidden_layer_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout_ratio),
        tf.keras.layers.Dense(num_classes)
    ])


def main(num_epochs, optimiser):
    x_train, y_train, x_test, y_test = get_data()

    model = create_model(
        input_dims=(28, 28),
        hidden_layer_dim=128,
        dropout_ratio=0.2,
        num_classes=10
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=num_epochs)

    print('Evaluate on training data')
    model.evaluate(x_train, y_train, verbose=2)
    print('Evaluate on test data')
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == '__main__':
    optimiser = 'adam'
    num_epochs = 5
    main(num_epochs, optimiser)
