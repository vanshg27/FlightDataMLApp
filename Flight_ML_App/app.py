import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)
# Load the model
model_path = os.path.join(os.path.dirname(__file__), "ML_Model", "flight_price_logistic_model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Flight Price: $ {}'.format(output))

if __name__ == "__main__":
    app.run()