from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('boston_model.pkl', 'rb'))

# Define feature names (used when converting input into DataFrame)
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from form and convert to float
        input_features = [float(request.form[str(i)]) for i in range(13)]

        # Convert to DataFrame to match training input structure
        input_df = pd.DataFrame([input_features], columns=feature_names)

        # Predict using the trained model
        prediction = model.predict(input_df)[0]

        # Show the result
        return render_template('index.html', prediction_text=f'Predicted House Price: ${round(prediction * 1000, 2)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
