import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

(x_train,_),(x_test,_)=tf.keras.datasets.mnist.load_data()

x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

x_train=np.expand_dims(x_train,-1)
x_test=np.expand_dims(x_test,-1)

noise_factor=0.5
x_train_noisy=x_train+noise_factor*np.random.normal(size=x_train.shape)
x_test_noisy=x_test+noise_factor*np.random.normal(size=x_test.shape)

x_train_noisy=np.clip(x_train_noisy,0,1)
x_test_noisy=np.clip(x_test_noisy,0,1)

input_img=layers.Input(shape=(28,28,1))

x=layers.Conv2D(32,3,activation="relu",padding="same")(input_img)
x=layers.MaxPooling2D(2,padding="same")(x)
x=layers.Conv2D(16,3,activation="relu",padding="same")(x)
encoded=layers.MaxPooling2D(2,padding="same")(x)

x=layers.Conv2D(16,3,activation="relu",padding="same")(encoded)
x=layers.UpSampling2D(2)(x)
x=layers.Conv2D(32,3,activation="relu",padding="same")(x)
x=layers.UpSampling2D(2)(x)
decoded=layers.Conv2D(1,3,activation="sigmoid",padding="same")(x)

autoencoder=Model(input_img,decoded)

autoencoder.compile(optimizer="adam",loss="binary_crossentropy")

autoencoder.summary()

history=autoencoder.fit(
    x_train_noisy,x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy,x_test)
)

denoised=autoencoder.predict(x_test_noisy)

n=10
plt.figure(figsize=(15,6))

for i in range(n):
    plt.subplot(3,n,i+1)
    plt.imshow(x_test[i].squeeze(),cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,n,i+1+n)
    plt.imshow(x_test_noisy[i].squeeze(),cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(3,n,i+1+2*n)
    plt.imshow(denoised[i].squeeze(),cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.title("Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

print("Test Loss:",autoencoder.evaluate(x_test_noisy,x_test,verbose=0))
