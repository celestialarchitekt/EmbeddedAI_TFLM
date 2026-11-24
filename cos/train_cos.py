# Train a tiny model that approximates cos(x), convert it to int8 TFLite,
# and export model.h for the Arduino GIGA sketch. Paste into ONE Colab cell and run.

import numpy as np
import tensorflow as tf

# 1. Training data: x in [0, 2*pi], y = cos(x) plus a little noise
np.random.seed(1)
tf.random.set_seed(1)
x = np.random.uniform(0, 2 * np.pi, 1000).astype(np.float32)
y = (np.cos(x) + 0.05 * np.random.randn(1000)).astype(np.float32)

# 2. Define and train (same shape as the sine demo: 1 -> 16 -> 16 -> 1)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=300, batch_size=32, verbose=0)
print("Final loss:", model.evaluate(x, y, verbose=0))

# 3. Convert to a fully int8-quantized TFLite model
def representative_data():
    for v in x[:200]:
        yield [np.array([v], dtype=np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Per-tensor weights for Dense layers: older TFLM kernels may mishandle per-channel ones
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
tflite_model = converter.convert()
print("Model size:", len(tflite_model), "bytes")

# 4. Check the quantized model in Python (compare these with the board later)
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()




inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
in_scale, in_zp = inp["quantization"]
out_scale, out_zp = out["quantization"]
for v in [0.0, 1.57, 3.14, 4.71]:
    q = np.round(v / in_scale + in_zp).astype(np.int8)
    interp.set_tensor(inp["index"], np.array([[q]], dtype=np.int8))
    interp.invoke()
    r = interp.get_tensor(out["index"])[0][0]
    print(f"x={v:.2f}  model cos={(r - out_zp) * out_scale:.2f}  real cos={np.cos(v):.2f}")

# 5. Write model.h in the format the sketch expects (g_model)
def to_c_array(data, name="g_model"):
    rows = []
    for i in range(0, len(data), 12):
        rows.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]))
    body = ",\n".join(rows)
    return (f"alignas(8) const unsigned char {name}[] = {{\n{body}\n}};\n"
            f"const unsigned int {name}_len = {len(data)};\n")

with open("model.h", "w") as f:
    f.write(to_c_array(tflite_model))

from google.colab import files
files.download("model.h")