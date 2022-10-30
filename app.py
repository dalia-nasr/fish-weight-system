from flask import Flask, render_template, request, make_response, url_for
import pickle
import pandas as pd
import inference

app = Flask(__name__)

PICKLE_PATH = 'v1.model'
DF_TRAIN_PATH = 'fish_train.csv'


@app.route("/")
def system():
    return render_template('index.html')


#  predict fish weight based on six variables
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        species = request.form['species']
        vertical_length = float(request.form['length1'])
        diagonal_length = float(request.form['length2'])
        cross_length = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])
        model = pickle.load(open(PICKLE_PATH, 'rb'))
        prediction = inference.PredictionModel.predict(model, species,
                                                       vertical_length,
                                                       diagonal_length,
                                                       cross_length,
                                                       height,
                                                       width)
        if prediction is not None:
            return render_template('index.html', prediction=prediction)
        return make_response("Error: Missing data", 404)
    return make_response("Error in method", 404)


#  add a new example to the dataset with six variables and fish weight 
@app.route("/add", methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        species = request.form['species']
        vertical_length = float(request.form['length1'])
        diagonal_length = float(request.form['length2'])
        cross_length = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])
        weight = float(request.form['weight'])
        inference.PredictionModel.add(DF_TRAIN_PATH, species,
                                      vertical_length,
                                      diagonal_length,
                                      cross_length,
                                      height,
                                      width,
                                      weight)
        return render_template('index.html')
    return make_response("Error in method", 404)


# trigger manual re-training of the ML model
@app.route("/retrain", methods=['POST'])
def retrain():
    model = pickle.load(open(PICKLE_PATH, 'rb'))
    data = pd.read_csv(DF_TRAIN_PATH)
    inference.PredictionModel.train(model, data)
    inference.PredictionModel.serialize(model, PICKLE_PATH)
    return render_template('index.html')


# handle error of unavailable routes
@app.errorhandler(404)
def not_found(error):
    resp = make_response("page not found!", 404)
    return resp

if __name__ == '__main__':
    app.run(debug=True)