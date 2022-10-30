# backend-predictive-model
## Description:
Develop a system for accurate fish weight prediction.
The system re-train the ML model on a schedule and expose itself via REST API with the following capabilities:
- Predict fish weight based on six variables
- Add a new example to a dataset with six variables and fish weight
- Trigger manual re-training of the ML model


## Install Locally:
Open the cmd and follow the instructions below:

1- Clone this github repository: ``` git clone https://github.com/dalia-nasr/fish-weight-system.git ```

2- Navigate to the project folder: ``` cd fish-weight-system ```

2- Install the required modules: ``` pip install -r requirements.txt ```

3- Run the flask app: ``` flask --app app run ```

## Deployed System:
The system was deployed on Heroku cloud provider: ```https://fishweightsystem.herokuapp.com/``` 

## APIs Documentation:
- ```/predict ```: This route receives the required 6 parameters and returns the predicted fish weight.
- ``` /add ```: This route receives the required 7 parameters with the fish weight and adds rhem to the dataset.
- ``` /retrain ```: This route retrains the fish weight prediction model.

* The required parameters to predict the accurate fish weight are:
species, vertical_length, diagonal_length,  cross_length, height, width
