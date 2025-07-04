# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:32:55 2022

@author: Bernd Ebenhoch
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

w0 = np.linspace(-1, 1, 101)
w1 = np.linspace(-1, 1, 101)

W0, W1 = np.meshgrid(w0, w1)


# ZZ soll unsere Pseudo-Lossfunktion sein
loss = (0.8*(W0+0.4)**2+2*(W1+0.4)**2+0.2*(W0+0.3)*(W1-0.1) +
        1.5*(W0+0.3)*(W1+0.1)**2+0.00486400000000008)

ax = plt.gca()

ax.contourf(W0, W1, loss, cmap='Blues',
            levels=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8],
            linestyles='solid')
ax.contour(W0, W1, loss,
           levels=[0.0001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8],
           linestyles='solid')

learning_rate = 0.1
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

optimizer = keras.optimizers.SGD(learning_rate=learning_rate,
                                 momentum=0.5,
                                 #                                 #                                 nesterov=True
                                 )


# optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)


# optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate,
#                               epsilon=1e-07)

# optimizer = keras.optimizers.Adam(learning_rate=learning_rate)


# w sind zwei Optimierungsparameter z.B. Gewichtung und Bias
# (X- und Y-Achsen in der Darstellung)
w = tf.Variable([1.0, -1.0])

# Wir wollen eine physische Kopie der Parameter erstellen
# In numpy geht das mit .copy() in tensorflow heißt der Befehl identity
w_old = tf.identity(w)


# Leere Liste in welche wir die Losses speichern
losses = []
for i in range(50):

    # Eigene Optimierung
    with tf.GradientTape() as tape:
        loss = (0.8*(w[0]+0.4)**2+2*(w[1]+0.4)**2+0.2*(w[0]+0.3) *
                (w[1]-0.1)+1.5*(w[0]+0.3)*(w[1]+0.1)**2+0.00486400000000008)

    dloss_dw = tape.gradient(loss, w)
    # diese Zeile auskommentieren wenn Optimizier verwendet werden sollen
    # w = tf.Variable(w-(learning_rate*dloss_dw))

    losses.append(loss)
    # diese Zeile einkommentieren wenn Optimizier verwendet werden sollen
    optimizer.apply_gradients(zip([dloss_dw], [w]))

    # Wir zeichnen einen Pfeil vom alten w-Parameter zum neuen w-Parameter
    plt.arrow(w_old[0], w_old[1], (w[0]-w_old[0]), (w[1]-w_old[1]),
              width=0.005, head_width=0.05, zorder=20, color='black',
              length_includes_head=True)
    w_old = tf.identity(w)

plt.xlabel('w[0]')
plt.ylabel('w[1]')
plt.show()

plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()
