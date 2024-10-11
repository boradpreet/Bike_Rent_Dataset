from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model only once
with open('Bike.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collecting input data from the form
        season = int(request.form['season'])
        mnth = int(request.form['mnth'])
        holiday = int(request.form['holiday'])
        weekday = int(request.form['weekday'])
        workingday = int(request.form['workingday'])
        weathersit = int(request.form['weathersit'])
        temp = float(request.form['temp'])
        atemp = float(request.form['atemp'])
        hum = float(request.form['hum'])
        windspeed = float(request.form['windspeed'])
        casual = int(request.form['casual'])
        registered = int(request.form['registered'])
        Year = int(request.form['Year'])
        Day = int(request.form['Day'])
        
        # Prepare the feature vector for the model
        features = np.array([[season, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, Year, Day]])
        
        # Predict the output using the trained model
        prediction = model.predict(features)
        
        # Use the exact output value without rounding
        output = int(prediction[0])  # This should give you the exact prediction value
        
        # Create a message for the prediction result
        pred_message = f"Total Bike Rent :-  {output}"  # Exact prediction value
        
        return render_template('index.html', pred=pred_message)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
