# from flask import Flask, render_template, request, jsonify
# from joblib import load

# app = Flask(__name__)

# # Load your trained model
# model = load(r'C:\Walmart sales prediction\Models\gradientboost.sav')

# def predict_sales(date, is_holiday, store_type, department):
#     # Placeholder: You need to customize this based on your model requirements
#     # For simplicity, I'm using placeholder values and processing here.
#     # You should replace these with your actual feature processing logic.
#     processed_date = len(date)  # Placeholder: Using the length of the date as a feature
#     processed_department = department  # Placeholder: Using the department as is
    
#     # Create a feature vector for prediction
#     feature_vector = [processed_date, is_holiday, store_type, processed_department]

#     # Make the prediction
#     prediction = model.predict([feature_vector])[0]
#     return prediction

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     date = request.form['date']
#     is_holiday = int(request.form['isHoliday'])
#     store_type = int(request.form['storeType'])
#     department = int(request.form['department'])

#     prediction = predict_sales(date, is_holiday, store_type, department)
#     return jsonify({'prediction': prediction})

# if __name__ == '__main__':
#     app.run(debug=True)


import numpy as np
import pandas as pd
import datetime as dt
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(r'C:\Walmart sales prediction\Models\rf1.sav')
fet = pd.read_csv('all_features.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]

    if features[3]=='0':
        features[3]=False
    else:
        features[3]=True

    df=fet[(fet['Store']==int(features[0])) & (fet['IsHoliday']==features[3]) & (fet['Date']==features[2])]
    f_features=[]
    d=dt.datetime.strptime(features[2], '%Y-%m-%d')
    c=0
    if df['Type'][0]=='C':
        c=1
    else:
        c=0

    if features[3]==False:
        features[3]=0
    else:
        features[3]=1

    if df.shape[0]==1:
        f_features.append(df['CPI'])
        f_features.append(d.date().day)
        f_features.append(int(features[1]))
        f_features.append(df['Fuel_Price'])
        f_features.append(features[3])
        f_features.append(d.date().month)
        f_features.append(df['Size'])
        f_features.append(int(features[0]))
        f_features.append(df['Temperature'])
        f_features.append(c)
        f_features.append(df['Unemployment'])
        f_features.append(d.date().year)

    final_features = [np.array(f_features)]
    output = model.predict(final_features)[0]
    return render_template('index.html', output=output)


if __name__ == "__main__":
    app.run(debug=True)
