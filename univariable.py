import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    r2_score,
)

PROCESSING_TYPE = "SCALED"
# PROCESSING_TYPE = "NORMAL"
ALGO = "MSE"
# ALGO = "MAE"

DONT_PROCESS = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    # "CoarseAgg",
    # "FineAgg",
    "Age",
]
# ========================
#
# ========================
settings = {
    "NORMAL_MAE": {
        "Cement": [100000, 0.005],
        "BlastFurnaceSlag": [100000, 0.0000039],
        "FlyAsh": [500000, 0.5],
        "Water": [100000, 0.000029],
        "Superplasticizer": [100000, 2.5],
        "CoarseAgg": [100000, 0.00005],
        "FineAgg": [300000, 0.000009],
        "Age": [500000, 0.5],
        "Strength": [],
    },
    "NORMAL_MSE": {
        "Cement": [500000, 0.005],
        "BlastFurnaceSlag": [500000, 0.006],
        "FlyAsh": [500000, 0.05],
        "Water": [500000, 0.005],
        "Superplasticizer": [500000, 0.005],
        "CoarseAgg": [500000, 0.00005],
        "FineAgg": [500000, 0.0005],
        "Age": [500000, 0.0005],
        "Strength": [],
    },
    "SCALED_MSE": {
        "Cement": [100000, 0.05],
        "BlastFurnaceSlag": [100000, 0.05],
        "FlyAsh": [100000, 0.05],
        "Water": [100000, 0.05],
        "Superplasticizer": [100000, 0.05],
        "CoarseAgg": [100000, 0.1],
        "FineAgg": [100000, 0.1],
        "Age": [100000, 0.05],
        "Strength": [],
    },
    "SCALED_MAE": {
        "Cement": [100000, 0.05],
        "BlastFurnaceSlag": [100000, 0.05],
        "FlyAsh": [100000, 0.05],
        "Water": [100000, 0.05],
        "Superplasticizer": [100000, 0.05],
        "CoarseAgg": [100000, 0.05],
        "FineAgg": [100000, 0.05],
        "Age": [100000, 0.05],
        "Strength": [],
    },
}
# ========================
#
# ========================


def main():
    # Read data
    dataFilename = "Concrete_Data.xls"
    df = pd.read_excel(dataFilename, names=settings["NORMAL_MSE"].keys())

    # This data set has 25 duplicates. Make unique by adding an index.
    df["index"] = range(1, len(df) + 1)

    # Segregate the sets
    trainSet = df.sample(n=900, random_state=2)
    testSet = pd.concat([df, trainSet]).drop_duplicates(keep=False)

    # Drop the added "index" column
    trainSet.pop("index")
    testSet.pop("index")

    # Scale data in-place
    if PROCESSING_TYPE == "SCALED":
        scaler = preprocessing.MinMaxScaler()
        trainSet[trainSet.columns] = scaler.fit_transform(trainSet[trainSet.columns])

    # Loop over each column
    for (columnName, columnData) in trainSet.iteritems():
        if columnName not in ["Strength"] + DONT_PROCESS:
            trainSet_y = trainSet["Strength"]
            trainSet_x = columnData
            config = settings[f"{PROCESSING_TYPE}_{ALGO}"][columnName]
            # Train
            obj = LinearRegression(
                trainSet_x,
                trainSet_y,
                columnName,
                ALGO,
                epochs=config[0],
                learnRate=config[1],
            )
            obj.predict(trainSet_x, trainSet_y)

            # Predict test data
            testSet_x = testSet[columnName]
            testSet_y = testSet["Strength"]
            obj.predict(testSet_x, testSet_y)


class LinearRegression:
    def __init__(self, x, y, featureName, algo, epochs, learnRate) -> None:
        # Convert DataFrames into Numpy arrays
        self.x = x.to_numpy().reshape(-1, 1)
        self.y = y.to_numpy().reshape(-1, 1)
        self.featureName = featureName
        self.totalSamples = len(x)
        self.algo = algo
        self.MSE_cost = {}  # Tracking variable

        print(self.featureName)

        # Calls
        self.w, self.b = self.do_BGD(epochs, learnRate)
        self.saveCostPlot()

    def do_BGD(self, epochs, learnRate):
        # Initialization
        w = 0
        b = 0
        for _ in range(epochs):
            # Choose a random sample
            iRand = randint(0, self.totalSamples - 1)
            xSample, ySample = self.x[iRand], self.y[iRand]

            # Calculate the hypothesis
            yPredicted = w * xSample + b

            # Get cost (Mean Squared Error)
            self.MSE_cost[_] = mean_squared_error(ySample, yPredicted)

            if self.algo == "MSE":
                # Calculate the gradient (MSE)
                wGrad = -(2 / self.totalSamples) * np.sum(
                    (ySample - yPredicted) * xSample
                )
                bGrad = -(2 / self.totalSamples) * np.sum((ySample - yPredicted))
            else:
                # Calculate the gradient (MAE) using approximation (derivative(|x|) = sgn(x))
                wGrad = (1 / self.totalSamples) * np.sum(
                    np.sign(yPredicted - ySample) * xSample
                )
                bGrad = (1 / self.totalSamples) * np.sum(np.sign(yPredicted - ySample))

            # Update
            w = w - learnRate * wGrad
            b = b - learnRate * bGrad

        return w, b

    """
    Inputs:
        x: Pandas DataFrame with multiple features 
        y: Pandas DataFrame with a single column (response)
    """

    def predict(self, x_test, y_test):
        x_test, y_test = x_test.to_numpy(), y_test.to_numpy()
        res = self.w * x_test + self.b
        # print(y_test)
        # print(res)
        print(f"\t EV, {explained_variance_score(list(y_test), list(res))}")
        # print(f"R2 = {r2_score(y_test, res)}")

    """
    Uses the training data to produce charts 

    Inputs:
        valuesMap: Dictionary {y value : x Value }
    """

    def saveCostPlot(self):
        # Init Plot
        fig, (ax, bx) = plt.subplots(2)
        ax.set_title(f"{self.algo} - Strength vs {self.featureName}")
        _units = " (kg)" if self.featureName != "Age" else " (days)"
        ax.set(xlabel=self.featureName + _units, ylabel="Strength (MPa)")
        ax.plot(self.x, self.y, "bo")
        # Plot
        res = self.w * self.x + self.b
        ax.plot(self.x, res, "r")
        # Plot MSE
        xCosts, yCosts = zip(*self.MSE_cost.items())
        bx.set_title(f"Cost")
        bx.set(xlabel="Epoch", ylabel="MSE")
        bx.plot(xCosts, yCosts)
        plt.tight_layout()
        plt.savefig(f"./img_output/{self.featureName}")


if __name__ == "__main__":
    main()
