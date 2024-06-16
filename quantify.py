from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the model
loaded_model = load_model('./model/knowledge_density.keras')

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
    return predicted