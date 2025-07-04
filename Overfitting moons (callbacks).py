# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:48:48 2022

@author: Bernd Ebenhoch
"""

# In diesem Beispiel wollen wir eine ganze Reihe von Callbacks ausprobieren

# In diesem Beispiel werden einige inports verwendet. Es handelt sich überwiegend um
# Standardbibliotheken, die bereits in Python vorhanden sind.
# Sollten ggf. Bibliotheken fehlen, kann der jeweilige Code-Abschnitt
# auskommentiert werden

import time
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import matplotlib


# Zufallszahlengenerator in keras initialisieren
# keras.utils.set_random_seed(0)

x, y = make_moons(noise=0.5, random_state=0, n_samples=1000)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.5, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm')
plt.show()


# %% Tensorboard Callback

lr = 0.0001
batch_size = 500

# Für Tensorboard muss ein Ordner für Log-Dateien angelegt werden
root_logdir = os.path.join(os.curdir, "my_logs")

# Für jeden Durchlauf wird ein Unterordner des aktuellen
# Tags und Uhrzeit angelegt
# '/my_logs/run_2019_06_07-15_15_22'
run_id = time.strftime("lr="+str(lr)+"batch=" +
                       str(batch_size)+"_Adam_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(root_logdir, run_id)

# Callback für das Tensorboard
# run_logdir ist der Pfad in dem die Log-Dateien gespeichert werden
tensorboard = keras.callbacks.TensorBoard(run_logdir)

# Zum Ausführen des Tensorboards nach Durchlauf des Fit-Prozesses:
# in anaconda prompt:
# in aktuelles Verzeichnis wechseln mit: cd (pfad)
# tensorboard --logdir=./my_logs --port=6006
# anschließend im Browser: http://localhost:6006

# %% ReduceLROnPlateau Callback

# die Lernrate kann bei erreichen eines Plateaus des Loss-Werts reduziert werden
# Dadurch kommt die Optimierung manchmal besser in das Minimum hinein und erreicht
# einen niedrigeren Loss-Wert, insbesondere bei der Regression
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=20,
    verbose=0,
    mode="auto",
    min_delta=0.0,
    cooldown=0,
    min_lr=1e-6)


# %% EarlyStopping Callback

# Beobachten ab wann val_accuracy wieder schlechter wird
# Nach 20 weiteren Epochen abbrechen
# anschließend auf die besten Parameter zurücksetzen
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True)

# %% ModelCheckpoint Callback


# Alle Modelle während des Fit-Prozesses auf der Festplatte speichern
# Pfad kann ein Ordner oder eine *.keras-datei sein
# Im Dateipfad kann die Epoche angegeben werden z.B. 'checkpoint{epoch}.h5',
checkpoint = keras.callbacks.ModelCheckpoint(
    'checkpoint_1.keras',
    monitor="val_accuracy",
    verbose=True,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq='epoch')  # batches)


# %% Eigener Callback


def show_fig(epoch, history):

    # der Callback liefert die aktuelle Epoche sowie, die Metriken und Lernrate
    # learning_rate = history['lr']

    # Wir erstellen eine Prognose aller Datenpunkte und stellen diese grafisch dar
    y_pred = model.predict(x, verbose=False)
    plt.scatter(x[:, 0], x[:, 1], c=(y_pred), cmap='coolwarm')
    plt.title('Predicted Data (epoch=' + str(epoch))
    plt.show()


show_fig_callback = keras.callbacks.LambdaCallback(on_epoch_end=show_fig)


# %% Figures im Tensorboard darstellen

backend = matplotlib.get_backend()


def save_fig_to_logs(epoch, history):

    # Wir erstellen eine Prognose aller Datenpunkte und stellen diese grafisch dar
    y_pred = model.predict(x, verbose=False)

    matplotlib.use('agg')  # Interaktives Backend für Matplotlib deaktivieren
    fig = plt.figure(figsize=(6, 4), dpi=128)
    plt.scatter(x[:, 0], x[:, 1], c=(y_pred), cmap='coolwarm')
    plt.title('Predicted Data (epoch=' + str(epoch))

    # Wir rufen mithilfe der IO-Bibliothek die Matplotlib-Grafik als RGB-Array ab
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=128)
    io_buf.seek(0)
    img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(1, int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close()
    matplotlib.use(backend)  # Matplotlib-Backend zurücksetzen

    # Mit einem file_writer wird das Bild zu den Log-Datein im Tensorboard geschrieben
    file_writer = tf.summary.create_file_writer(run_logdir)
    with file_writer.as_default():
        tf.summary.image("Training data", img, step=epoch)


# Nchfolgende Zeile einkommeniteren, wenn die Zwischenergebnisse nach
# jeder Epoche als Grafiken im tensorboard dargestellt werden sollen
# (der own_callback von oben wird dann überschrieben)
fig_to_tensorboard_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=save_fig_to_logs)


# %% Modell erzeugen


model = keras.models.Sequential()
model.add(keras.layers.Input((2,)))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


# Eine Zusammenfassung des Modells ausgeben
model.summary()

# Es kann auch eine grafische Darstellung erzeugt werden
# Dazu sind jedoch pydot und graphviz notwendig
keras.utils.plot_model(model)

optimizer = keras.optimizers.Adam(learning_rate=lr)  # std = 0.001
model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

# Wir können auf die gewichtungen und Bias-Werte der einzelnen Schichten zugreifen
# Ausgabe der Gewichtungen der ersten Schicht
print(model.layers[0].weights[1].shape)

# Ausgabe der Bias-Werte der ersten Schicht
print(model.layers[1].weights)

# %% Modell mit den Callbacks fitten


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=200, verbose=True,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard,
                               lr_reduction,
                               early_stopping,
                               # checkpoint,
                               # show_fig_callback,
                               fig_to_tensorboard_callback
                               ])


# Lernkurven der Loss-Funktion und Metriken anzeigen
plt.plot(history.history['accuracy'], label='Train_accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# keras.models.load_model läd das gespeicherte Modell
# So kann auch der beste Checkpoint wieder geladen werden
# Achtung: Es kann zu Fehlern führen wenn im Modell kein input_shape definiert ist
model.save('checkpoint.keras')
model2 = keras.models.load_model('checkpoint.keras')


# Wenn durch Early_stopping auf best_weights zurückgesetzt wird
# entsprechen die Ergebnisse von .evaluate nicht den letzten Punkten der
# Lernkurven
print()
print('Train_accuracy', history.history['accuracy'][-1])
print('Validation_accuracy', history.history['val_accuracy'][-1])


score = model2.evaluate(x_train, y_train, verbose=True)
print('Train_accuracy', score[1])

score = model2.evaluate(x_val, y_val, verbose=False)
print('Validation_accuracy', score[1])
