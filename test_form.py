# test_form.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_homepage_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Boston House Price Predictor" in response.data

def test_form_has_13_inputs(client):
    response = client.get('/')
    assert response.status_code == 200
    # Count input tags in the form
    input_count = response.data.count(b'<input type="text"')
    assert input_count == 13, f"Expected 13 inputs, found {input_count}"

def test_prediction(client):
    test_data = {str(i): 1.0 for i in range(13)}  # Dummy values
    response = client.post('/predict', data=test_data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Predicted" in response.data or b"prediction" in response.data
