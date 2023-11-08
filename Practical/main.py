import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the SVM model trained for credit card eligibility
model = pickle.load(open('model/Practical.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # feature = request.form.values('spending')
    user_input = [float(request.form[field]) for field in ['age', 'HbA1c', 'blood_glucose_level', 'hypertension', 'heart_disease','bmi']]
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    print(prediction)
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'

    return render_template('index.html', prediction_output= f'The result is {result}')


if __name__ == "__main__":
    app.run(debug=True)