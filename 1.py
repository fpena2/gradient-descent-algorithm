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


class LinearRegression:
    def __init__(self, x, y, featureName, cleanData, algo) -> None:
        self.x = x
        self.y = y
        self.featureName = featureName
        self.totalSamples = len(self.x)
        self.cleanData = cleanData
        self.algo = algo

    def do_BGD(self, epoch, learnRate):
        # Track
        MSE_cost = {}
        # Initialization
        w = 0
        b = 0
        # Convert DataFrames into Numpy arrays
        x = self.x.to_numpy()
        y = self.y.to_numpy()

        # Preprocess and Scale the input and output arrays
        if self.cleanData:
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
            MSE_cost[_] = mean_squared_error(ySample, yPredicted)

            if self.algo == "MSE":
                # Calculate the gradient (MSE)
                wGrad = -(2 / self.totalSamples) * np.sum(
                    (ySample - yPredicted) * xSample
                )
                bGrad = -(2 / self.totalSamples) * np.sum((ySample - yPredicted))
            else:
                # Calculate the gradient (MAE) using approximation (|x|" = sgn(x))
                wGrad = (1 / self.totalSamples) * np.sum(
                    np.sign(yPredicted - ySample) * xSample
                )
                bGrad = (1 / self.totalSamples) * np.sum(np.sign(yPredicted - ySample))

            # Update
            w = w - learnRate * wGrad
            b = b - learnRate * bGrad

        # Plot
        res = w * x + b
        ax.plot(x, res, "r")

        # Plot MSE
        xCosts, yCosts = zip(*MSE_cost.items())
        bx.set_title(f"Cost")
        bx.set(xlabel="Epoch", ylabel="MSE")
        bx.plot(xCosts, yCosts)
        plt.tight_layout()

        # Print Metrics
        a = mean_squared_error(y, res)
        b = mean_absolute_error(y, res)
        c = explained_variance_score(y, res)
        print(f"{self.featureName},{a},{b},{c}")
        plt.savefig(self.featureName)


settings = {
    "NORMAL": {
        "Cement": [500000, 0.005],
        "BlastFurnaceSlag": [500000, 0.001],
        "FlyAsh": [500000, 0.05],
        "Water": [500000, 0.005],
        "Superplasticizer": [500000, 0.005],
        "CoarseAgg": [500000, 0.00005],
        "FineAgg": [500000, 0.0005],
        "Age": [500000, 0.0005],
        "Strength": [],
    },
    "SCALED": {
        "Cement": [100000, 0.5],
        "BlastFurnaceSlag": [100000, 0.5],
        "FlyAsh": [100000, 0.5],
        "Water": [100000, 0.5],
        "Superplasticizer": [100000, 0.5],
        "CoarseAgg": [100000, 0.5],
        "FineAgg": [100000, 0.5],
        "Age": [100000, 0.5],
        "Strength": [],
    },
}

# Read data
dataFilename = "Concrete_Data.xls"
df = pd.read_excel(dataFilename, names=settings["NORMAL"].keys())

# This data set has 25 duplicates. Make unique by adding an index.
df["index"] = range(1, len(df) + 1)

# Segregate the sets
trainSet = df.sample(n=900, random_state=2)
testSet = pd.concat([df, trainSet]).drop_duplicates(keep=False)

# Drop the added "index" column
trainSet.pop("index")
testSet.pop("index")


# ========================
#
# ========================

PROCESSING_TYPE = "SCALED"
# PROCESSING_TYPE = "NORMAL"
# ALGO = "MSE"
ALGO = "MAE"
DONT_PROCESS = [
    # "Cement",
    # "BlastFurnaceSlag",
    # "FlyAsh",
    # "Water",
    # "Superplasticizer",
    # "CoarseAgg",
    # "FineAgg",
    # "Age",
]

# ========================
#
# ========================

# Loop over each column
for (columnName, columnData) in trainSet.iteritems():
    if columnName not in ["Strength"] + DONT_PROCESS:
        trainSet_y = trainSet["Strength"]
        trainSet_x = columnData
        config = settings[PROCESSING_TYPE][columnName]
        cleanup = True if PROCESSING_TYPE == "SCALED" else False
        # Perform
        obj = LinearRegression(trainSet_x, trainSet_y, columnName, cleanup)
        obj.do_BGD(epoch=config[0], learnRate=config[1])
