"""FreshSense AI - train a SIMPLE dense ANN to detect fresh vs rotten fruit.

Dataset : Project-AgML/fresh_rotten_fruit_classification (Hugging Face),
          raw config. Labels: 0 = fresh, 1 = rotten.
Model   : plain feedforward ANN (basic ANN concepts):
          Flatten(48x48x3) -> Dense(256, relu) -> Dropout(0.3)
                           -> Dense(128, relu) -> Dropout(0.3)
                           -> Dense(1, sigmoid)
Protocol: stratified 80/20 train/test split (fixed seed). Final score on test.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

BASE = Path(__file__).parent
IMG = 48
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
CLASSES = ["fresh", "rotten"]


def load_data():
    xs, ys = [], []
    for p in sorted((BASE / "data").glob("train-*.parquet")):
        df = pd.read_parquet(p)
        for lab, img in zip(df["label"], df["image"]):
            im = Image.open(io.BytesIO(bytes(img["bytes"]))).convert("RGB").resize((IMG, IMG))
            xs.append(np.asarray(im, dtype=np.float32) / 255.0)
            ys.append(int(lab))
    X = np.stack(xs)
    y = np.array(ys)
    print(f"loaded {len(X)} images | fresh={int((y==0).sum())} rotten={int((y==1).sum())}", flush=True)
    return X, y


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(IMG, IMG, 3)),
        keras.layers.Flatten(),                       # 48*48*3 = 6912 inputs
        keras.layers.Dense(256, activation="relu"),   # hidden layer 1
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),   # hidden layer 2
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),  # fresh (0) vs rotten (1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"train {len(Xtr)} | test {len(Xte)}", flush=True)
    # balance fresh vs rotten
    counts = np.bincount(ytr)
    cw = {i: counts.sum() / (len(counts) * c) for i, c in enumerate(counts)}

    model = build_model()
    model.summary()
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                       restore_best_weights=True, verbose=1)
    model.fit(Xtr, ytr, batch_size=128, epochs=40, validation_split=0.15,
              class_weight=cw, callbacks=[es], verbose=2)

    print("\n=== FINAL TEST RESULT ===", flush=True)
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"Test accuracy: {acc:.4f} ({acc:.1%}) | requirement >= 70% -> "
          f"{'PASS' if acc >= 0.70 else 'FAIL'}", flush=True)
    yp = (model.predict(Xte, verbose=0).ravel() > 0.5).astype(int)
    print(classification_report(yte, yp, target_names=CLASSES, digits=4), flush=True)
    print("confusion matrix:\n", confusion_matrix(yte, yp), flush=True)

    model.save(BASE / "freshsense_ann.keras")
    print("saved freshsense_ann.keras", flush=True)


if __name__ == "__main__":
    main()
