import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the dataset and train your model as you mentioned
df = pd.read_csv("crop_yield_me.csv")
reg = linear_model.LinearRegression()
reg.fit(df[['Crop_Year', 'Area', 'Rainfall', 'Pesticide', 'Ph']], df.Price)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        crop_year = float(request.form['crop-year'])
        area = float(request.form['area'])
        rainfall = float(request.form['rainfall'])
        pesticide = float(request.form['pesticide'])
        ph = float(request.form['ph'])
        
        # Make a prediction using your trained model
        predicted_price = reg.predict([[crop_year, area, rainfall, pesticide, ph]])
        
        return render_template('index.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)