# gradient-descent-algorithm

Implementation of a Gradient Descent based Linear Regression Algorithm.

## Non Processed Data Learning Rates (MAE)

```
    "NORMAL": {
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
```

## Non Processed Data Learning Rates (MSE)

```
    "NORMAL": {
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
```

<!--
sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()
scaled_X = sx.fit_transform(trainSet.drop("Strength", axis="columns"))
scaled_y = sy.fit_transform(trainSet["Strength"].values.reshape(trainSet.shape[0], 1))
labels = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAgg",
    "FineAgg",
    "Age",
]
df = pd.DataFrame(scaled_X, columns=labels)
df.hist()
plt.show()
-->
