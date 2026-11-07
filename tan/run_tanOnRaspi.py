# Run tan_model.tflite on the Raspberry Pi 5 with LiteRT and compare against numpy tan().
# Usage: python run_tan_pi.py [path/to/tan_model.tflite]

import sys
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter   # pip install ai-edge-litert
except ImportError:
    from tflite_runtime.interpreter import Interpreter   # older fallback

model_path = sys.argv[1] if len(sys.argv) > 1 else "tan_model.tflite"

interp = Interpreter(model_path=model_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("input dtype :", inp["dtype"].__name__, " quantization:", inp["quantization"])
print("output dtype:", out["dtype"].__name__, " quantization:", out["quantization"])


def predict(x):
    data = np.array([[x]], dtype=np.float32)
    if inp["dtype"] == np.int8:                       # quantized model
        scale, zero = inp["quantization"]
        data = np.clip(np.round(data / scale + zero), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], data)
    interp.invoke()
    y = interp.get_tensor(out["index"])[0][0]
    if out["dtype"] == np.int8:
        scale, zero = out["quantization"]
        y = (float(y) - zero) * scale
    return float(y)


print("\n   x      model     tan(x)    error")
for x in [-1.2, -1.0, -0.5, 0.0, 0.5, 0.7, 1.0, 1.1, 1.2]:
    y = predict(x)
    print(f"{x:6.2f}  {y:8.4f}  {np.tan(x):8.4f}  {y - np.tan(x):+8.4f}")

xs = np.linspace(-1.2, 1.2, 200)
mae = np.mean([abs(predict(x) - np.tan(x)) for x in xs])
print(f"\nMean absolute error over 200 points: {mae:.4f}")