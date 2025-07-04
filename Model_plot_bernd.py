# -*- coding: utf-8 -*-
"""
Erweitertes Keras-Beispiel mit mehreren Callbacks und Modell-Plot

@author: Angepasst von OpenAI
"""

import time
import os
import io
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

# Daten erzeugen
x, y = make_moons(noise=0.5, random_state=0, n_samples=1000)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm')
plt.title('Trainingsdaten')
plt.show()

# -----------------------------------------------
# TensorBoard-Verzeichnis definieren
# -----------------------------------------------
lr = 0.0001
batch_size = 500

root_logdir = os.path.join(os.curdir, "my_logs")
run_id = time.strftime(f"lr={lr}_batch={batch_size}_Adam_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(root_logdir, run_id)

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# -----------------------------------------------
# Weitere Callbacks
# -----------------------------------------------
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=20,
    min_lr=1e-6,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'checkpoint_1.keras',
    monitor="val_accuracy",
    save_best_only=False,
)

# -----------------------------------------------
# Modell erzeugen
# -----------------------------------------------
model = keras.models.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()

# Modellgrafik erzeugen
keras.utils.plot_model(model,
                       to_file='model_plot.png',
                       show_shapes=True,
                       show_layer_names=True)

optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# -----------------------------------------------
# Eigenes Bild in TensorBoard speichern
# -----------------------------------------------
backend = matplotlib.get_backend()

def save_fig_to_tensorboard(epoch, logs):
    y_pred = model.predict(x, verbose=False)

    matplotlib.use('agg')  # Nicht-interaktives Backend
    fig = plt.figure(figsize=(6, 4), dpi=128)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred.ravel(), cmap='coolwarm')
    plt.title(f'Predicted Data (epoch={epoch})')

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=128)
    io_buf.seek(0)

    img = tf.image.decode_png(io_buf.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)  # Für TensorBoard: (1, height, width, channels)

    io_buf.close()
    plt.close()
    matplotlib.use(backend)

    file_writer = tf.summary.create_file_writer(run_logdir)
    with file_writer.as_default():
        tf.summary.image("Prediction", img, step=epoch)

fig_callback = keras.callbacks.LambdaCallback(on_epoch_end=save_fig_to_tensorboard)

# -----------------------------------------------
# Modell trainieren
# -----------------------------------------------
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=200,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard_cb,
                               lr_reduction,
                               early_stopping,
                               checkpoint_cb,
                               fig_callback])

# -----------------------------------------------
# Lernkurven anzeigen
# -----------------------------------------------
plt.plot(history.history['accuracy'], label='Train_accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------------------------------
# Modell speichern & wieder laden
# -----------------------------------------------
model.save('final_model.keras')
model2 = keras.models.load_model('final_model.keras')

# -----------------------------------------------
# Genauigkeit evaluieren
# -----------------------------------------------
print()
print('Train_accuracy:', history.history['accuracy'][-1])
print('Validation_accuracy:', history.history['val_accuracy'][-1])

score_train = model2.evaluate(x_train, y_train, verbose=True)
print('Reloaded Train_accuracy:', score_train[1])

score_val = model2.evaluate(x_val, y_val, verbose=True)
print('Reloaded Validation_accuracy:', score_val[1])

# -----------------------------------------------
# TensorBoard-Start-Info
# -----------------------------------------------
print("\nStarte TensorBoard mit:")
print(f"tensorboard --logdir={root_logdir} --port=6006")
print("und öffne dann im Browser: http://localhost:6006")
