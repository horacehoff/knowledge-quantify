import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
from dataset import dataset


# Training data
texts = []
labels = []  # Corresponding percentage of knowledge density


for set in dataset:
    texts.append(set[0])
    labels.append(set[1])
print("DATASET COMPLETE")

# Tokenization and padding
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=100)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=50000, output_dim=512))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])

print("BEGINNING TRAINING")
history = model.fit(X_train, np.array(y_train), epochs=60, batch_size=8, validation_data=(X_val, np.array(y_val)))

model.save('model/knowledge_density.keras')
with open('model/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)


# Evaluate on validation data
val_loss, val_mae = model.evaluate(X_val, np.array(y_val))
print(f"Validation MAE: {val_mae}")

# Predict on new data
new_text = ["Mitochondria is the powerhouse of the cell."]
new_seq = tokenizer.texts_to_sequences(new_text)
new_data = pad_sequences(new_seq, maxlen=100)
prediction = model.predict(new_data)
print(f"Predicted percentage of knowledge density: {prediction[0][0]}")