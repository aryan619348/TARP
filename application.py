import pickle
from flask import Flask, request, render_template, app
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)


# route for home page

@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            State=request.form.get('State'),
            Crop=request.form.get('Crop'),
            Season=request.form.get('Season'),
            Area=float(((request.form.get('Area')))))

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(features=pred_df)
        results_out=results[0]/100000000
        return render_template('home.html', results=results_out)


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True)
