from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('boston_model.pkl', 'rb'))

# Define feature names (must match the model's training columns exactly)
feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 
                 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs and convert to float
        input_features = [float(request.form[str(i)]) for i in range(13)]

        # Create DataFrame with correct feature names
        input_df = pd.DataFrame([input_features], columns=feature_names)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'Predicted House Price: ${round(prediction * 1000, 2)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
