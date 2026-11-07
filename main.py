import pandas as pd
import mlflow.sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri('http://localhost:5000')

mlflow.set_experiment('titanic')

data = pd.read_csv("train.csv")

features  = ["Pclass", "Age", "Sex"]

X = data[features].copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

max_depth = 3

with mlflow.start_run(run_name=f"tree_depth_{max_depth}"):
    # Сохраняем настройки запуска.
    mlflow.log_params({
        "max_depth": max_depth,
        "model_random_state": 42,
        "split_random_state": 42,
        "test_size": 0.2,
        "features": ",".join(features),
    })

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Сохраняем результат проверки в MLflow.
    mlflow.log_metric("test_accuracy", accuracy)

    # Сохраняем обученную модель.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print(f"Доля правильных предсказаний: {accuracy:.1%}")