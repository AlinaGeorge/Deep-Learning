import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

# Temperature data
temperature = [25,26,27,28,27,29,30,31,30,29,28,27,26,25,26,27,29,30,31,32,31,30,29,28,27,26,25,27,28,30]

# Convert to array
temperature = np.array(temperature).reshape(-1,1)

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(temperature)

# Create input sequences
x = []
y = []

for i in range(5,len(data)):
    x.append(data[i-5:i])
    y.append(data[i])

x = np.array(x)
y = np.array(y)

# Split data
split = int(0.8*len(x))
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

# Create LSTM model
model = Sequential()
model.add(LSTM(50,input_shape=(5,1)))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

# Train model
model.fit(x_train,y_train,epochs=50,batch_size=4,verbose=1)

# Predict
predicted = model.predict(x_test)

# Convert back to original temperature
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test)

# Display values
print("Actual temperatures:")
print(actual.flatten())

print("Predicted temperatures:")
print(predicted.flatten())

# Plot
plt.plot(actual,label="Actual")
plt.plot(predicted,label="Predicted")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Actual vs Predicted Temperature")
plt.legend()
plt.show()
