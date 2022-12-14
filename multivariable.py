import time
import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
)

# Numpy printing config
np.set_printoptions(linewidth=150)

# Globals
PROCESSING_TYPE = "SCALED"
# PROCESSING_TYPE = "NORMAL"
ALGO = "MSE"
# ALGO = "MAE"

settings = {
    "SCALED_MSE": [100000, 0.1],
    "SCALED_MAE": [100000, 0.1],
    "NORMAL_MSE": [100000, 0.0001],
    "NORMAL_MAE": [100000, 0.0001],
}

labels = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAgg",
    "FineAgg",
    "Age",
    "Strength",
]


def main():
    # Read data
    dataFilename = "Concrete_Data.xls"
    df = pd.read_excel(dataFilename, names=labels)

    # This data set has 25 duplicates. Make unique by adding an index.
    df["index"] = range(1, len(df) + 1)

    # Segregate the sets
    trainSet = df.sample(n=900, random_state=2)
    testSet = pd.concat([df, trainSet]).drop_duplicates(keep=False)

    # Drop the added "index" column
    trainSet.pop("index")
    testSet.pop("index")

    # Scale the training data in-place
    if PROCESSING_TYPE == "SCALED":
        scaler = preprocessing.MinMaxScaler()
        trainSet[trainSet.columns] = scaler.fit_transform(trainSet[trainSet.columns])

    # Split dataset
    trainSet_x = trainSet.loc[:, trainSet.columns != "Strength"]
    trainSet_y = trainSet["Strength"]

    # Call
    config = settings[f"{PROCESSING_TYPE}_{ALGO}"]
    obj = SGD(trainSet_x, trainSet_y, ALGO, config[0], config[1])
    obj.predict(trainSet_x, trainSet_y, "Train")

    # Blindly apply the training "transform" to testing data
    if PROCESSING_TYPE == "SCALED":
        testSet[testSet.columns] = scaler.transform(testSet[testSet.columns])

    # Calculate with Test Data
    testSet_y = testSet["Strength"]
    testSet_x = testSet.loc[:, testSet.columns != "Strength"]
    obj.predict(testSet_x, testSet_y, "Test")


class SGD:
    def __init__(self, x, y, algo, epochs, learnRate) -> None:
        # Convert DataFrames into Numpy arrays
        self.x, self.y = x.to_numpy(), y.to_numpy()
        self.totalFeatures, self.totalSamples = len(x.columns), len(x)
        self.algo = algo

        # Train the model
        start = time.time()
        self.w = self.train_SGD_model(epochs, learnRate)
        end = time.time()
        print(f"Training Time = {end - start}")

    """
    Stochastic Gradient Descent Algorithm 
    """

    def train_SGD_model(self, epochs, learnRate):
        # Track
        loss = {}

        # Initialization. The "b" term is embedded into this array
        w = np.zeros(shape=(self.totalFeatures + 1))

        #  Start
        for _ in range(epochs):
            # Choose a random sample
            iRand = randint(0, self.totalSamples - 1)
            xSample, ySample = self.x[iRand], self.y[iRand]

            # Embed the "b" term
            xSample = np.insert(xSample, 0, 1, axis=0)

            # Calculate the hypothesis
            yPredicted = np.dot(w, xSample.T)

            # Calculate the Loss (Mean Squared Error)
            loss[_] = mean_squared_error([ySample], [yPredicted])

            # Exit condition
            if loss[_] < 1e-12:
                break

            if self.algo == "MSE":
                # Calculate the gradient (derivative)
                wGrad = -(2 / self.totalSamples) * np.dot(
                    (ySample - yPredicted), xSample.T
                )
            else:
                wGrad = (
                    (1 / self.totalSamples) * np.sign(yPredicted - ySample) * xSample
                )

            # Update
            w = w - learnRate * wGrad

        self.saveCostPlot(loss)
        return w

    """
    Inputs:
        valuesMap: Dictionary {y value : x Value }
    """

    def saveCostPlot(self, valuesMap):
        xCosts, yCosts = zip(*valuesMap.items())
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Loss Function")
        plt.plot(xCosts, yCosts)
        plt.savefig(f"./imgs/M_{ALGO}_{PROCESSING_TYPE}")

    """
    Inputs:
        x: Pandas DataFrame with multiple features 
        y: Pandas DataFrame with a single column (response)
    """

    def predict(self, x, y, info):
        x, y = x.to_numpy(), y.to_numpy()
        _x = np.insert(x, 0, 1, axis=1)
        res = np.sum(self.w * _x, axis=1)
        # print(self.w)
        print(f"{info} Variance Explained= {explained_variance_score(y, res)}")


if __name__ == "__main__":
    main()
