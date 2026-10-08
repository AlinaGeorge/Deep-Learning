import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Function to create model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Optimizers
optimizers = {
    "SGD_Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.01),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001)
}

histories = {}

# Train using each optimizer
for name, optimizer in optimizers.items():
    print(f"\nTraining with {name}")

    model = create_model()

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Accuracy ({name}): {test_acc:.4f}")

    histories[name] = history

#Accuracy vs Epochs
plt.figure(figsize=(10,6))

for name, history in histories.items():
    plt.plot(history.history['accuracy'], label=name)

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.show()

#Loss vs Epochs
plt.figure(figsize=(10,6))

for name, history in histories.items():
    plt.plot(history.history['loss'], label=name)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.show()

print("\nFinal Comparison\n")

print("{:<15}{:<15}{:<15}{:<15}{:<15}".format(
    "Optimizer",
    "Train Acc",
    "Val Acc",
    "Train Loss",
    "Val Loss"))

#Print Final Results
for name, history in histories.items():

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    print("{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}".format(
        name,
        train_acc,
        val_acc,
        train_loss,
        val_loss
    ))
