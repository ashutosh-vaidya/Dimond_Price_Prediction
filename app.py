from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipline import CustomData, PredictPipeline
#from src.pipeline.training_pipeline import 

application = Flask(__name__)

app = application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        #render the from to get the input for the data prediction
        return render_template("form.html")
    else:
        data = CustomData(
            carat=float(request.form.get("carat")),
            depth=float(request.form.get("depth")),
            table=float(request.form.get("table")),
            x=float(request.form.get("x")),
            y=float(request.form.get("y")),
            z=float(request.form.get("z")),
            cut=request.form.get("cut"),
            color=request.form.get("color"),
            clarity=request.form.get("clarity")
        )
        final_data = data.get_data_as_dataframe()
        prediction_pipline = PredictPipeline()
        pred = prediction_pipline.predict(final_data)

        result = round(pred[0],2)
        return render_template("result.html",final_result = result)

if __name__ == "__main__":
    app.run(host= "0.0.0.0", debug=True)