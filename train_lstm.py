import numpy as np
import pandas as pd 

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Read the data
body_swing_df = pd.read_csv('BODYSWING.csv')
hand_wave_df = pd.read_csv('HANDWAVE.csv')

no_of_timesteps = 10
X = []
y = []

dataset = body_swing_df.iloc[:, 1:].values
n_sample = len(dataset)

for i in range(no_of_timesteps, n_sample):
  X.append(dataset[i-no_of_timesteps:i, :]) # 10 timesteps added to X
  y.append(0) # 0 for body swing

X, y = np.array(X), np.array(y)
print(X.shape, y.shape) # (40, 10, 132) (40,) 40 samples, 10 timesteps, 132 features, features are x, y, z, visibility

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Sequential() # Stacked LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))
model.save("model.h5")
