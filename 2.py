import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
)


class SGD:
    def __init__(self, trainSet) -> None:
        self.y = trainSet["Strength"]
        self.x = trainSet.loc[:, trainSet.columns != "Strength"]
        self.totalFeatures = len(self.x.columns)
        self.totalSamples = len(self.x)

    def do_SGD(self, epoch, learnRate):
        # Track
        costs = {}

        # Initialization
        b = 0
        w = np.ones(shape=(self.totalFeatures))
        # Embed the "b" term
        w = np.insert(w, 0, b, axis=0)

        # Convert DataFrames into Numpy arrays
        x = self.x.to_numpy()
        y = self.y.to_numpy()

        #  Start
        for _ in range(epoch):

            # Choose a random sample
            iRand = randint(0, self.totalSamples - 1)
            xSample, ySample = x[iRand], y[iRand]

            # Embed the "b" term
            xSample = np.insert(xSample, 0, 1, axis=0)

            # Calculate the hypothesis
            yPredicted = np.dot(w, xSample.T)

            # Get cost (Mean Squared Error)
            costs[_] = np.mean(np.square(ySample - yPredicted))

            # Calculate the gradient (derivative)
            wGrad = -(2 / self.totalSamples) * np.dot((ySample - yPredicted), xSample.T)

            # Update
            w = w - learnRate * wGrad

        res = []
        x = np.insert(x, 0, 1, axis=1)

        for row in x:
            res.append(np.dot(w.T, row))

        # Print Metrics
        print(f"MSE = {mean_squared_error(y, res)}")
        print(f"MAE = {mean_absolute_error(y, res)}")
        print(f"VE = {explained_variance_score(y, res)}")
        plt.plot(costs.keys(), costs.values())
        plt.show()


# Read data
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

# Scale data in-place
df = trainSet
scaler = preprocessing.MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

a = SGD(df)
a.do_SGD(10000, 0.5)
