import pickle
import pandas as pd


# Prediction Model definition
class PredictionModel:
    def __init__(self, encoder=None, model=None):
        import pickle
        self.encoder = encoder
        self.model = model

    # predicts fish weight based on parameters
    def predict(self,
                species: str,
                vertical_length: float,
                diagonal_length: float,
                cross_length: float,
                height: float,
                width: float
                ) -> float:
        import numpy as np

        if model is None:
            raise Exception("Model wasn't trained")

        enc = np.array([species]).reshape(-1, 1)
        encoded_part = self.encoder.transform(enc).toarray()
        X = np.hstack([
            np.array([vertical_length, diagonal_length,
                     cross_length, height, width]).reshape(1, 5),
            encoded_part
        ])

        return self.model.predict(X)

    # trains new model using pandas dataframe
    # dataframe structure should replicate fish_train.csv structure
    def train(self, df: pd.DataFrame) -> None:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np

        self.encoder = OneHotEncoder()
        encoded_part = self.encoder.fit_transform(
            df.Species.values.reshape(-1, 1)).toarray()

        y = df.Weight.values
        X = df.drop(['Species', 'Weight'], axis=1)
        X = np.hstack([X.values, encoded_part])

        self.model = LinearRegression()
        self.model.fit(X, y)

    # creates new pickle file
    def serialize(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)
    
    # add sample record to the dataset
    def add(df: pd.DataFrame, 
            species: str,
            vertical_length: float,
            diagonal_length: float,
            cross_length: float,
            height: float,
            width: float,
            weight: float
            ):
        import csv
        new_sample = [species, vertical_length,
                      diagonal_length, cross_length, height, width, weight]
        with open(df, mode="a") as f1:
            review_writer = csv.writer(f1, delimiter=",")
            review_writer.writerow(new_sample)


DF_TRAIN_PATH = 'fish_train.csv'
PICKLE_PATH = 'v1.model'

# model setup
model = PredictionModel()

# training of the model
train_df = pd.read_csv(DF_TRAIN_PATH)
model.train(train_df)

# model serialization
model.serialize(PICKLE_PATH)

# model inference
prediction = model.predict('Bream', 23.2, 23.4, 36, 11.52, 5.02)
# print(f'Prediction: {prediction}')

# clearing up
del model

# loading model from pickle file
model = pickle.load(open(PICKLE_PATH, 'rb'))


# re-using loaded model
prediction = model.predict('Smelt', 23.2, 23.4, 36, 11.52, 5.02)
# print(f'Prediction: {prediction}')
