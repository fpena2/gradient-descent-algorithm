import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt


class SGD:
    def __init__(self, trainSet) -> None:
        self.y = trainSet["Strength"]
        self.x = trainSet.loc[:, trainSet.columns != "Strength"]
        self.totalFeatures = len(self.x.columns)
        self.totalSamples = len(self.x)

    def do_SGD(self, epoch=1000, learnRate=0.00001):
        # Track
        epochs = []
        costs = []

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
            xSample = x[iRand]
            ySample = y[iRand]

            # Embed the "b" term
            xSample = np.insert(xSample, 0, 1, axis=0)

            # Calculate the hypothesis
            yPredicted = np.dot(w, xSample.T)

            # Calculate the gradient (derivative)
            wGrad = -(2 / self.totalSamples) * np.dot((ySample - yPredicted), xSample.T)

            # Update
            w = w - learnRate * wGrad

            # Get cost (Mean Squared Error)
            cost = np.mean(np.square(ySample - yPredicted))

            costs.append(cost)
            epochs.append(_)

        return w, cost, costs, epochs

    def do_prediction(testSample, w):
        pass
        res = w[0] + w[1] + w[2] + w[3] + w[4] + w[5] + w[6] + w[7]


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

w, cost, costs, epochs = SGD(trainSet).do_SGD()


# Display info
plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epochs, costs)
plt.show()