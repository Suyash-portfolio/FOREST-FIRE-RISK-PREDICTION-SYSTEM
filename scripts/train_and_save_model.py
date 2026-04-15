from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_data_with_coords_final_normalized.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Keep model inputs aligned with the web form.
X = df[["temp", "RH", "wind", "rain"]]
y = df["y"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, MODEL_PATH)

accuracy = pipeline.score(X_test, y_test)
print(f"Saved model to: {MODEL_PATH}")
print(f"Test accuracy: {accuracy:.4f}")
