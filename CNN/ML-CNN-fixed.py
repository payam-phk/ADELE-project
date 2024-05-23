import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
import keras.backend as K
import time


# Name = "ML-CNN-fixed-{}".format(int(time.time()))
# tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))


input_data = pd.read_csv('Trainingdataset/projected_potential.csv', index_col=0).values
input_data = input_data[:, 75:200]
print(input_data.shape)
output_data = pd.read_csv('Trainingdataset/combined_binary_added.csv', index_col=0).values

# normalize the input data between 0 and 1
# max_data = np.max(input_data)
# min_data = np.min(input_data)
# input_data = (input_data - min_data) / (max_data - min_data)

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

@keras.saving.register_keras_serializable(package="my_package", name="custom_leaky_relu")
def custom_leaky_relu(x, alpha=0.05, shift=1.0):
    return K.maximum(alpha * (x - shift), x - shift)

model = keras.Sequential([
    #conv1D
    keras.layers.Conv1D(512, 3, activation='leaky_relu', input_shape=(125, 1)),
    keras.layers.AveragePooling1D(2),
    # keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(256, 3, activation='leaky_relu'),
    # keras.layers.MaxPooling1D(2),
    keras.layers.AveragePooling1D(2),
    keras.layers.Conv1D(128, 3, activation='leaky_relu'),
    # keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64, 3, activation='leaky_relu'),
    # keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(32, 3, activation='leaky_relu'),
    # keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(30, activation=custom_leaky_relu)
])

def lr_scheduler(epoch, lr):
    if epoch <= 3:
        return 0.002
    elif epoch <= 6:
        return 0.001
    # elif epoch <= 30:
    #     return 0.0005
    else:
        return lr * 0.75  # Reduce learning rate by 5% after epoch 50

initial_learning_rate = 0.002
lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (usually validation loss)
    min_delta=0.0001,          # Minimum change in the monitored quantity to qualify as an improvement
    patience=10,          # Number of epochs with no improvement to wait
    verbose=1,            # Verbosity mode. 0: quiet, 1: update messages
    restore_best_weights=False  # Restore model weights from the epoch with the best value of the monitored metric
)


optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)



model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), callbacks=[lr_callback, early_stopping])


loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")



plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save the model
model.save("ML-CNN-fixed125.keras")

test_input = input_data[10].reshape(1, 125)
# print(test_input)
prediction = model.predict(test_input)
# round the prediction to 3 decimal places
prediction1 = np.around(prediction, decimals=2).reshape(30,)
print(prediction1)
print(output_data[10])

# test with 20th row of input data
# print(input_data[20])
test_input = input_data[20].reshape(1, 125)
# print(test_input)
prediction = model.predict(test_input)
# round the prediction to 3 decimal places
prediction2 = np.around(prediction, decimals=2).reshape(30,)
print(prediction2)
print(output_data[20])

# test with 30th row of input data
# print(input_data[30])
test_input = input_data[30].reshape(1, 125)
# print(test_input)
prediction = model.predict(test_input)
# round the prediction to 3 decimal places
prediction3 = np.around(prediction, decimals=2).reshape(30,)
print(prediction3)
print(output_data[30])


# plot three graphs

# plot the first graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(output_data[10], label='Actual')
plt.plot(prediction1, label='Predicted')
plt.legend()

# plot the second graph
plt.subplot(1, 3, 2)
plt.plot(output_data[20], label='Actual')
plt.plot(prediction2, label='Predicted')
plt.legend()
# plot the third graph
plt.subplot(1, 3, 3)
plt.plot(output_data[30], label='Actual')
plt.plot(prediction3, label='Predicted')

plt.legend()
plt.tight_layout()
plt.show()
