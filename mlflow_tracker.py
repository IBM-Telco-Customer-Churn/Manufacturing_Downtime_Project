import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
)


def log_classification_run(
    experiment_name: str,
    run_name: str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    params: dict = None,
):
  mlflow.set_experiment(experiment_name)

  with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("framework", "scikit-learn")

    if params:
      mlflow.log_params(params)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test, average="weighted"),
        "test_f1_score": f1_score(y_test, y_pred_test, average="weighted"),
    }
    mlflow.log_metrics(metrics)

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax)
    plt.title(f"Confusion Matrix - {run_name}")
    fig.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close(fig)

    mlflow.sklearn.log_model(model, artifact_path="model")

    return metrics
