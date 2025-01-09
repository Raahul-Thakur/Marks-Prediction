from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve the form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),  # Fixing the score field mapping
                writing_score=float(request.form.get('writing_score'))   # Fixing the score field mapping
            )

            # Convert the data to a DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Dataframe before prediction:")
            print(pred_df)

            # Run prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Handle result and render response
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('home.html', error_message="Error during prediction, please try again later.")
    else:
        # Render the form page for GET request
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
