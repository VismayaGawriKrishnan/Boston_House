from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('boston_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = model.predict(final)[0]
    return render_template('index.html', prediction_text=f'Predicted House Price: ${round(prediction*1000, 2)}')

if __name__ == "__main__":
    app.run(debug=True)

model = pickle.load(open('boston_model.pkl', 'rb'))


# Feature names in the same order used for training
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# After collecting values from the form
input_features = [float(request.form[str(i)]) for i in range(13)]

# Wrap as DataFrame with column names
input_df = pd.DataFrame([input_features], columns=feature_names)

# Predict
prediction = model.predict(input_df)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
