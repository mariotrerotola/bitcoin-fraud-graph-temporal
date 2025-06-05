# %%
# keras_tuner_nn.py
"""
Hyper‑parameter tuning con **KerasTuner** (niente scikeras) per il dataset Chainabuse.

✔ Evita lo stack GridSearchCV/scikeras → nessun problema di `__sklearn_tags__` 
✔ Parametri esplorati: layers, units, dropout, learning‑rate, activation, batch‑size, epochs
✔ EarlyStopping integrato
✔ Preprocessing (ColumnTransformer) identico alla versione precedente
✔ Salva preprocessor (`joblib`) e modello migliore (`best_keras_model.h5`)

Esecuzione:
    python keras_tuner_nn.py
Requisiti:
    pip install "keras-tuner>=1.4" scikit-learn joblib pandas tensorflow matplotlib packaging
"""

from pathlib import Path
import logging, sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder,
                                   StandardScaler)
from sklearn.utils.class_weight import compute_class_weight

import keras_tuner as kt
from dexire.dexire import DEXiRE
from packaging import version
import sklearn

# %%
# ---------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------
BASE_PATH = Path("../../data/processed_data")
USECASE = "case_1.csv"
DATASET_PATH = BASE_PATH / USECASE 
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)s │ %(message)s")

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# %%
# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
logging.info("Loading dataset …")
df = pd.read_csv(DATASET_PATH)
logging.info("Dataset shape: %s", df.shape)

X = df.drop(columns="class")
y_raw = df["class"]

# %%
# ---------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------
num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_pipeline = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = SkPipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
])

# %%
# ------------------------------
# 3. Encode labels & class weights
# ------------------------------
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)
class_weight_values = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight = {i: w for i, w in enumerate(class_weight_values)}
logging.info("Class weights: %s", class_weight)


# %%
# ---------------------------------------------------------------------
# 4. Train/validation/test split
# ---------------------------------------------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Fit preprocessor solo su train_full (no leakage)
X_train_full_proc = preprocessor.fit_transform(X_train_full)
X_test_proc = preprocessor.transform(X_test)

# ulteriore split in train/val per tuner
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_proc, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
)

input_dim = X_train.shape[1]
num_classes = len(le.classes_)

# %%
# ---------------------------------------------------------------------
# 5. Keras model builder per KerasTuner
# ---------------------------------------------------------------------

def model_builder(hp: kt.HyperParameters):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for i in range(hp.Int("layers", 1, 3)):
        units = hp.Choice(f"units_{i}", [32, 64, 128])
        activation = hp.Choice("activation", ["relu", "tanh"])
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        dropout_rate = hp.Choice("dropout", [0.0, 0.3, 0.5])
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="chainabuse_mlp")

    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# %%
# ---------------------------------------------------------------------
# 6. Tuner setup (Hyperband)
# ---------------------------------------------------------------------
MAX_EPOCHS = 50

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

project_dir = "kt_logs"

tuner = kt.Hyperband(
    model_builder,
    objective="val_accuracy",
    max_epochs=MAX_EPOCHS,
    factor=3,
    directory=project_dir,
    project_name="chainabuse_nn",
    overwrite=True,
)


# %%
# ---------------------------------------------------------------------
# 7. Search
# ---------------------------------------------------------------------
logging.info("Starting KerasTuner search …")
tuner.search(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[stop_early],
    class_weight=class_weight,
    batch_size=kt.HyperParameters().Choice("batch_size", [32, 64]),
)

# %%
# ---------------------------------------------------------------------
# 8. Retrieve best model
# ---------------------------------------------------------------------
logging.info("Search finished. Retrieving best model …")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    X_train_full_proc, y_train_full,
    epochs=MAX_EPOCHS,
    validation_split=0.1,
    callbacks=[stop_early],
    class_weight=class_weight,
    verbose=0,
)

# %%
# ---------------------------------------------------------------------
# 9. Evaluate on test
# ---------------------------------------------------------------------
loss, acc = model.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}\n")

print("Best hyper‑parameters:")
for k, v in best_hps.values.items():
    print(f"  {k}: {v}")

# Classification report
y_pred_test = np.argmax(model.predict(X_test_proc), axis=1)
print("\nClassification report (test):")
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot()
plt.title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.show()

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ipotesi: X_train, y_train, X_val, y_val, X_test, y_test sono già pronti
# e class_weight è definito se vuoi bilanciare le classi

# 1. Costruzione del modello
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    # Dropout(0.3),
    Dense(64, activation='relu'),
    # Dropout(0.3),
    Dense(32, activation='relu'),
    # Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# 2. Compilazione
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Callback per stop anticipato (opzionale ma consigliato)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 4. Addestramento per 80 epoche
history = model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight,  # rimuovi se non ti serve
    verbose=2
)

# 5. Valutazione sul test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f} — Test accuracy: {acc:.4f}")

# 6. (Opzionale) Plot della curva di apprendimento
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------------------------------
# 10. Rule Extraction with DEXiRE
# ---------------------------------------------------------------------
logging.info("Extracting symbolic rules with DEXiRE …")
feature_names = preprocessor.get_feature_names_out()
dexire = DEXiRE(model=model, class_names=le.classes_.tolist())

# Costruiamo DataFrame con nomi leggibili per DEXiRE
df_train_features = pd.DataFrame(X_train_full_proc, columns=feature_names)

rules = dexire.extract_rules(df_train_features, y_train_full)

with open("rules.txt", "w", encoding="utf-8") as f:
    f.write(str(rules))
logging.info("Rules saved to rules.txt")

# %%
import re

# 1. Carica il tuo preprocessor (se non è già in memoria)
from joblib import load
preprocessor = load('preprocessor.joblib')

# 2. Estrai i feature names e costruisci idx2name
feature_names = preprocessor.get_feature_names_out()
idx2name = {i: name for i, name in enumerate(feature_names)}

def prettify(rule_str):
    return re.sub(
        r"X_(\d+)",
        lambda m: idx2name.get(int(m.group(1)), m.group(0)),
        rule_str
    )

# 3. Supponiamo rules sia questa grande stringa:
#    "[IF ((X_23 > 874.85) AND (X_23 > 896.95)) THEN benign, IF ((X_23 <= 941.45) ...)]"
raw = str(rules).strip()              # roba come "[IF ... THEN ... , IF ...]"
raw = raw.lstrip('[').rstrip(']') # togli parentesi quadre

# 4. Split “grezzo” sulle virgole che separano le regole:
#    ATTENZIONE: funziona finché dentro alle regole non ci sono virgole.
raw_list = [r.strip() for r in raw.split(', IF ')]
# reinserisci la parola IF dove l’hai tolta dallo split
raw_list = [ (r if r.startswith('IF') else 'IF ' + r) for r in raw_list ]

# 5. Ora applica il mapping e stampa
pretty_rules = [prettify(r) for r in raw_list]
for pr in pretty_rules:
    print(pr)


