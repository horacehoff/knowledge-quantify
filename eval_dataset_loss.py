from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset import dataset

# Load the model
loaded_model = load_model('model/knowledge_density.keras')

# Load the tokenizer
with open('model/tokenizer.pkl', 'rb') as file:
    loaded_tokenizer = pickle.load(file)


def quantify(text):
    new_seq = loaded_tokenizer.texts_to_sequences([text])
    new_data = pad_sequences(new_seq, maxlen=100)
    prediction = loaded_model.predict(new_data, verbose=0)
    predicted = prediction[0][0]
    if predicted < 0:
        predicted = 0
    if predicted > 100:
        predicted = 100
    # print(f"Predicted knowledge density: {predicted}")
    return predicted


i = 0
for set in dataset:
    percentage = quantify(set[0])
    if abs(percentage - set[1]) > 10:
        print("ERROR: " + str(i)+ " / DIFF. "+str(abs(percentage - set[1])))
    i += 1

print(i)
