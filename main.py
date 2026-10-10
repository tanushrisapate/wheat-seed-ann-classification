"""Lab 6: Wheat Seed Classification Using an Artificial Neural Network.

Notebook-aligned ANN workflow (7 input features -> 16 -> 8 -> 3 classes).
Place the original CSV at data/seeds_new.csv or allow download from UCI.
The reported notebook metrics are from a saved previous run; actual results
may vary with initialization, environment, and early stopping.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "seeds_new.csv"
COLS = [
    "Area", "Perimeter", "Compactness", "Length of kernel",
    "Width of kernel", "Asymmetry coefficient",
    "Length of kernel groove", "Class (1, 2, 3)",
]


def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    # Public dataset used by the notebook: UCI Seeds, ID 236 (CC BY 4.0).
    # The uploaded notebook did not contain its local seeds_new.csv file.
    try:
        from ucimlrepo import fetch_ucirepo
        seeds = fetch_ucirepo(id=236)
        features = seeds.data.features.to_numpy(dtype=float)
        labels = seeds.data.targets.to_numpy().reshape(-1).astype(int)
        if features.shape != (210, 7) or labels.shape != (210,):
            raise ValueError("Unexpected UCI Seeds data dimensions")
        data = pd.DataFrame(np.column_stack([features, labels]), columns=COLS)
        data[COLS[-1]] = data[COLS[-1]].astype(int)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATA_PATH, index=False)
        print(f"Downloaded UCI Seeds dataset to {DATA_PATH}")
        return data
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Place your seeds_new.csv in the data folder "
            "or connect to the internet to download the UCI Seeds dataset."
        ) from exc


def plot_eda(df):
    df.mean(numeric_only=True).plot(kind="bar", figsize=(10, 5), color="skyblue")
    plt.title("Bar Plot of Seeds Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    df[COLS[-1]].value_counts().sort_index().plot(
        kind="pie", autopct="%1.1f%%", figsize=(6, 6)
    )
    plt.title("Pie Chart of Seed Classes")
    plt.ylabel("")
    plt.show()

    sns.histplot(data=df, x="Area", kde=True, bins=15, color="green")
    plt.title("Histogram with KDE of Area")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df[COLS[:-1]])
    plt.xticks(rotation=40, ha="right")
    plt.title("Box Plot of Seed Measurements")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Seeds Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    sns.scatterplot(data=df, x="Area", y="Perimeter", hue=COLS[-1])
    plt.title("Area vs Perimeter")
    plt.tight_layout()
    plt.show()


def train_and_evaluate(df):
    X = df.drop(columns=[COLS[-1]])
    y_original = df[COLS[-1]].astype(int)
    y = y_original - 1
    print("Original classes:", sorted(y_original.unique()))
    print("ANN encoded classes:", sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}; testing samples: {len(X_test)}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # The original notebook also called scaler.fit_transform(X_test) as a
    # display-only cell. It is intentionally omitted: never refit on test data.

    model = Sequential([
        Input(shape=(7,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train_scaled, y_train, epochs=100, batch_size=16,
        validation_split=0.2, callbacks=[early_stop], verbose=1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("ANN Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()

    loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_prob = model.predict(X_test_scaled, verbose=0)
    y_pred_original = np.argmax(y_prob, axis=1) + 1
    y_test_original = y_test.to_numpy() + 1

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {test_acc:.4%}")
    print(f"Precision (weighted): {precision_score(y_test_original, y_pred_original, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_test_original, y_pred_original, average='weighted'):.4f}")
    print(f"F1 (weighted): {f1_score(y_test_original, y_pred_original, average='weighted'):.4f}")
    print("Classification report:\n", classification_report(y_test_original, y_pred_original))
    cm = confusion_matrix(y_test_original, y_pred_original, labels=[1, 2, 3])
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.title("ANN Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_dataset()
    data = data.dropna(axis=1, how="all")
    data = data.loc[:, ~data.columns.str.startswith("Unnamed")]
    print("Dataset shape:", data.shape)
    print("Missing values:\n", data.isna().sum())
    print("Duplicates:", data.duplicated().sum())
    print(data.head())
    plot_eda(data)
    train_and_evaluate(data)
