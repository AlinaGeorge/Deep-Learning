import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
import numpy as np

(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to [-1,1]
X_train = (X_train.astype("float32") - 127.5) / 127.5

# Reshape to (28,28,1)
X_train = np.expand_dims(X_train, axis=-1)

BUFFER_SIZE = X_train.shape[0]
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    model = Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7,7,256)))
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1),
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2),
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2),
                                     padding='same',
                                     activation='tanh'))
    return model

generator = build_generator()

def build_discriminator():
    model = Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2),
                            padding='same',
                            input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5,5), strides=(2,2),
                            padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
noise_dim = 100
generator_losses = []
discriminator_losses = []

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_generator = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )
    gradients_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gradients_generator,
            generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_discriminator,
            discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

# Training Loop
EPOCHS = 30
for epoch in range(EPOCHS):
    g_loss_epoch = 0
    d_loss_epoch = 0
    batches = 0

    for image_batch in dataset:
        g_loss, d_loss = train_step(image_batch)
        g_loss_epoch += g_loss.numpy()
        d_loss_epoch += d_loss.numpy()
        batches += 1
    generator_losses.append(g_loss_epoch/batches)
    discriminator_losses.append(d_loss_epoch/batches)

    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"Generator Loss: {generator_losses[-1]:.4f}   "
          f"Discriminator Loss: {discriminator_losses[-1]:.4f}")


# Generate 16 Images
noise = tf.random.normal([16, noise_dim])
generated_images = generator(noise, training=False)
generated_images = generated_images * 127.5 + 127.5
plt.figure(figsize=(6,6))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(generated_images[i,:,:,0], cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Handwritten Digits")
plt.show()

# Plot Generator Loss
plt.figure(figsize=(8,5))
plt.plot(generator_losses, marker='o')
plt.title("Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot Discriminator Loss
plt.figure(figsize=(8,5))
plt.plot(discriminator_losses, marker='o')
plt.title("Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("\nFinal Generator Loss :", generator_losses[-1])
print("Final Discriminator Loss :", discriminator_losses[-1])
