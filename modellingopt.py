import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Experiment
mlflow.set_experiment("Latihan Credit Scoring")

# Load data
data = pd.read_csv("train_pca.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

# Dataset tracking
train_dataset = mlflow.data.from_pandas(
    pd.concat([X_train, y_train], axis=1),
    source="train_pca.csv",
    name="train_dataset"
)

# Input example
input_example = X_train.iloc[:5]

# Hyperparameter
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range = np.linspace(1, 50, 5, dtype=int)

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:

        with mlflow.start_run(
            run_name=f"rf_{n_estimators}_{max_depth}"
        ):

            # LOG DATASET
            mlflow.log_input(
                train_dataset,
                context="training"
            )

            # log parameter
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            # training
            model.fit(X_train, y_train)

            # evaluasi
            accuracy = model.score(X_test, y_test)

            # log metric
            mlflow.log_metric("accuracy", accuracy)

            print(f"Accuracy: {accuracy}")

            # log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )

            print("MODEL BERHASIL DISIMPAN")