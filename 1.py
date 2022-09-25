import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from sklearn import preprocessing


class LinearRegression:
    def __init__(self, x, y, featureName) -> None:
        self.x = x
        self.y = y
        self.featureName = featureName
        self.totalSamples = len(self.x)

    def do_BGD(self, epoch=100000, learnRate=0.2):
        # Track
        costs = {}

        # Initialization
        w = 0
        b = 0

        # Convert DataFrames into Numpy arrays
        x = self.x.to_numpy()
        y = self.y.to_numpy()

        # Preprocess and Scale the input and output arrays
        xScaler = preprocessing.MinMaxScaler()
        yScaler = preprocessing.MinMaxScaler()
        x = xScaler.fit_transform(x.reshape(-1, 1))
        y = yScaler.fit_transform(y.reshape(-1, 1))

        # Init Plot
        fig, (ax, bx) = plt.subplots(2)
        ax.set_title(f"Strength vs {self.featureName}")
        _units = " (kg)" if self.featureName != "Age" else " (days)"
        ax.set(xlabel=self.featureName + _units, ylabel="Strength (MPa)")
        ax.plot(x, y, "bo")

        for _ in range(epoch):

            # Choose a random sample
            iRand = randint(0, self.totalSamples - 1)
            xSample, ySample = x[iRand], y[iRand]

            # Calculate the hypothesis
            yPredicted = w * xSample + b

            # Get cost (Mean Squared Error)
            costs[_] = np.mean(np.square(ySample - yPredicted))

            # Calculate the gradient (derivative)
            wGrad = -(2 / self.totalSamples) * np.sum((ySample - yPredicted) * xSample)
            bGrad = -(2 / self.totalSamples) * np.sum((ySample - yPredicted))

            # Update
            w = w - learnRate * wGrad
            b = b - learnRate * bGrad

        # Plot
        res = w * x + b
        ax.plot(x, res, "ro")
        # Plot
        xCosts, yCosts = zip(*costs.items())
        bx.plot(xCosts, yCosts)
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

settings = {
    "Cement": [100000, 0.5],
    "BlastFurnaceSlag": [100000, 0.5],
    "FlyAsh": [100000, 0.5],
    "Water": [100000, 0.5],
    "Superplasticizer": [100000, 0.3],
    "CoarseAgg": [100000, 0.3],
    "FineAgg": [100000, 0.5],
    "Age": [100000, 0.5],
    "Strength": [],
}

DONT_PROCESS = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAgg",
    "FineAgg",
]

# Loop over each column
for (columnName, columnData) in trainSet.iteritems():
    if columnName not in ["Strength"] + DONT_PROCESS:
        trainSet_y = trainSet["Strength"]
        trainSet_x = columnData
        config = settings[columnName]
        # Perform
        obj = LinearRegression(trainSet_x, trainSet_y, columnName)
        obj.do_BGD(config[0], config[1])
