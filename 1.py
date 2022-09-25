import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, x, y, featureName) -> None:
        self.x = x
        self.y = y
        self.featureName = featureName
        self.totalSamples = len(self.x)

    def do_BGD(self, epoch=10000, learnRate=0.000001):
        # Track
        costs = []

        # Initialization
        w = 0
        b = 0

        # Convert DataFrames into Numpy arrays
        x = self.x.to_numpy()
        y = self.y.to_numpy()

        # Init Plot
        fig, ax = plt.subplots()
        ax.set_title(f"Strength vs {self.featureName}")
        _units = " (kg)" if self.featureName != "Age" else " (days)"
        ax.set(xlabel=self.featureName + _units, ylabel="Strength (MPa)")
        ax.plot(x, y, "bo")

        for _ in range(epoch):
            # Calculate the hypothesis
            yPredicted = w * x + b

            # Get cost (Mean Squared Error)
            costs.append({_: np.mean(np.square(y - yPredicted))})

            # Calculate the gradient (derivative)
            wGrad = -(2 / self.totalSamples) * np.sum((y - yPredicted) * x)
            bGrad = -(2 / self.totalSamples) * np.sum((y - yPredicted))

            # Update
            w = w - learnRate * wGrad
            b = b - learnRate * bGrad

        res = w * x + b
        plt.plot(x, res, "ro")
        plt.show()

        return w, b, costs


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

# Loop over each column
for (columnName, columnData) in trainSet.iteritems():
    if columnName != "Strength":
        trainSet_y = trainSet["Strength"]
        trainSet_x = columnData
        # Perform
        obj = LinearRegression(trainSet_x, trainSet_y, columnName)
        w, b, costs = obj.do_BGD()
