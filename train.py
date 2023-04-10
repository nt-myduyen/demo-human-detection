import os
import numpy as np
import pandas as pd 

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Read the data from cvs folder
csv_folder = '/home/yuu/Documents/PBL5-demo/CSV'

no_of_timesteps = 10
X = []
y = []

for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(csv_folder, file))
        
        # create variable for each csv file
        label = file[:-4] # label for each file is the file name without ".csv" extension
        
        # create binary label for "fall" and "not fall"
        if "Fall" in label:
            y_label = 1 # 1 for "fall"
        else:
            y_label = 0 # 0 for "not fall"
        
        dataset = df.iloc[:, 1:].values
        n_sample = len(dataset)

        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i-no_of_timesteps:i, :]) # 10 timesteps added to X
            y.append(y_label) # binary label added to y

X, y = np.array(X), np.array(y)
print(X.shape, y.shape) # (num_samples, 10, num_features)

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
