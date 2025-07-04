import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ======================================
# 1. Einfaches Modell
# ======================================
inputs = keras.Input(shape=(20,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ======================================
# 2. Compile
# ======================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# ======================================
# 3. TensorBoard Setup
# ======================================
root_logdir = os.path.join(os.curdir, "my_logs")
run_id = time.strftime("graph_run_%Y_%m_%d-%H_%M_%S")
logdir = os.path.join(root_logdir, run_id)
writer = tf.summary.create_file_writer(logdir)

# ======================================
# 4. Trace aktivieren und speichern
# ======================================
@tf.function
def trace_model():
    sample_input = tf.random.uniform([1, 20])
    return model(sample_input)

# Startet das Aufzeichnen
tf.summary.trace_on(graph=True, profiler=False)

# Führt einen Forward-Pass aus
trace_model()

# Speichert den Graph im TensorBoard
with writer.as_default():
    tf.summary.trace_export(
        name="model_trace",
        step=0,
        profiler_outdir=logdir
    )

# ======================================
# 5. Training (optional, erzeugt auch Scalars)
# ======================================
X_train = tf.random.uniform([1000, 20])
y_train = tf.random.uniform([1000, 1], maxval=2, dtype=tf.int32)

tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(X_train, y_train,
          epochs=5,
          batch_size=32,
          validation_split=0.2,
          callbacks=[tensorboard_cb])

print("\nStarte TensorBoard mit:")
print(f"tensorboard --logdir={root_logdir} --port=6006")
