from flask import Flask, render_template, request, redirect, session
import joblib
import pandas as pd
import os
from model import Model

mode = Model()
mode.fit()
app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Check if the user is logged in
def is_user_logged_in():
    return 'username' in session

@app.route('/')
def home():
    if is_user_logged_in():
        return redirect('/predict')
    else:
        return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Verify user credentials
        if verify_user_credentials(username, password):
            session['username'] = username
            return redirect('/predict')
        else:
            return render_template('login.html', message='Invalid username or password')

    return render_template('login.html', message='')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# Verify user credentials
def verify_user_credentials(username, password):
    # Load user credentials from file
    user_credentials = load_user_credentials()

    # Check if the username exists and the password matches
    if username in user_credentials and user_credentials[username] == password:
        return True

    return False

# Load user credentials from file
def load_user_credentials():
    user_credentials = {}

    if os.path.exists('credentials.txt'):
        with open('credentials.txt', 'r') as file:
            for line in file:
                username, password = line.strip().split(',')
                user_credentials[username] = password

    return user_credentials

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not is_user_logged_in():
        return redirect('/login')

    if request.method == 'POST':
        # Retrieve the input data from the form
        input_data = [{
            "Item_Identifier": request.form['Item_Identifier'],
            "Item_Weight" : float(request.form['Item_Weight']),
            "Item_Visibility" : float(request.form['Item_Visibility']),
            "Item_Type" : request.form['Item_Type'],
            "Item_MRP" : float(request.form['Item_MRP']),
            "Outlet_Identifier" : request.form['Outlet_Identifier'],
            "Outlet_Establishment_Year" : float(request.form['Outlet_Establishment_Year']),
            "Item_Fat_Content" : request.form['Item_Fat_Content'],
            "Outlet_Size" : request.form['Outlet_Size'],
            "Outlet_Location_Type" : request.form['Outlet_Location_Type'],
            "Outlet_Type" : request.form['Outlet_Type'],
        }]

        # Use the model for prediction
        prediction = mode.predict(input_data)

        # Return the prediction to the user
        return render_template('result.html', prediction=prediction)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)