from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#MAx Pooling layer --> model.add(MaxPooling2D(pool_size, strides, padding))
#dropout --> model.add(Dropout(rate, noise_shape=None, seed=None, **kwargs))

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plot the first image in the dataset
plt.imshow(X_train[0])

#check image shape
print("Shape of first image training: ", X_train[0].shape)

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
#one-hot encode target column
print("y_train before: ", y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("y_train after: ", y_train)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Show a summary of the whole model's layers, param and shapes
model.summary()

# #train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# #predict first 4 images in the test set
# #We should use a different set for testing, as the x_test has been used as validation set
# y_pred = model.predict(X_test[:4])
# print("Predicciones: ", y_pred)

# #actual results for first 4 images in test set
# print("Reales: ", y_test[:4])