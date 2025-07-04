# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:26:41 2025

@author: TAKO
"""
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ================================================
# Daten laden und anzeigen
# ================================================
df = pd.read_csv('IT-Equipment.csv', sep='\t', encoding='utf-8')
print(df.head())

# Eingabe- und Zielvariablen
X = df[['Budget']].values
y = df[['Computer', 'Monitor', 'Drucker', 'Maus']].values
columns = ['Computer', 'Monitor', 'Drucker', 'Maus']

# ================================================
# Daten skalieren
# ================================================
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# ================================================
# Trainings-, Validierungs- und Testdaten splitten
# ================================================
x_train, x_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.5, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=0)

# ================================================
# Visualisierung der Datenaufteilung
# ================================================
splits = [('Train', x_train, y_train, 'blue'),
          ('Validation', x_val, y_val, 'green'),
          ('Test', x_test, y_test, 'red')]

for i, col in enumerate(columns):
    plt.figure(figsize=(6, 4))
    for name, x_split, y_split, color in splits:
        plt.scatter(x_split, y_split[:, i], alpha=0.7, label=name, color=color)
    plt.title(f"{col} vs. Budget")
    plt.xlabel('Budget (skaliert)')
    plt.ylabel(col)
    plt.legend()
    plt.show()

# ================================================
# Modell erstellen
# ================================================
model = keras.models.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(4)  # 4 Outputs, linear (default)
])

model.summary()

# ================================================
# TensorBoard-Ordner anlegen
# ================================================
lr = 0.001
batch_size = 10
root_logdir = os.path.join(os.curdir, "my_logs")
run_id = time.strftime(f"lr={lr}_batch={batch_size}_Adam_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(root_logdir, run_id)

# ================================================
# Callbacks
# ================================================
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    mode='min'
)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'it_best_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=run_logdir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# ================================================
# Kompilieren
# ================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='mse',
    metrics=['mae']
)

# ================================================
# Trainieren
# ================================================
history = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb]
)

# ================================================
# Modell-Visualisierung speichern
# ================================================
keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False
)

# ================================================
# Vorhersage & Rücktransformation
# ================================================
y_pred_scaled = model.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

# ================================================
# Vergleich: Echte vs. Vorhergesagte Werte
# ================================================
print("\nVergleich echte Werte vs. Vorhersagen:\n")
for true_vals, pred_vals in zip(y_test_orig, y_pred):
    print(f"Echte Werte: {[round(v, 2) for v in true_vals]} \t Vorhersage: {[round(v, 2) for v in pred_vals]}")

# ================================================
# R²-Bewertung
# ================================================
r2 = r2_score(y_test_orig, y_pred, multioutput='uniform_average')
print(f"\nBestimmtheitsmaß (R²) auf Testdaten: {r2:.4f}")

# R² pro Zielvariable
r2_per_output = r2_score(y_test_orig, y_pred, multioutput='raw_values')
for i, name in enumerate(columns):
    print(f"{name}: R² = {r2_per_output[i]:.4f}")
