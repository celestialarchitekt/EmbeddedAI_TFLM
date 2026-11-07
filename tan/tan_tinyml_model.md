# tan(x) TinyML Model

A small neural network that approximates tan(x) for x in [-1.2, 1.2]. It is trained in Python on Google Colab, exported with TensorFlow Lite as a float32 model, and run for inference on a Raspberry Pi 5 and an Arduino GIGA R1.

## Model

| Property | Value |
|---|---|
| Task | Regression: x → tan(x) |
| Input range | -1.2 to 1.2 (tan diverges at ±π/2, about ±1.57) |
| Architecture | Input (1) → Dense 16, ReLU → Dense 16, ReLU → Dense 1 |
| Data type | float32 (no quantization) |
| Optimizer | Adam |
| Loss | Mean squared error (MSE) |
| Metric | Mean absolute error (MAE) |

## Training

- **Data:** 2000 random x values drawn uniformly from [-1.2, 1.2], with targets `tan(x)` plus Gaussian noise (standard deviation 0.05).
- **Settings:** 300 epochs, batch size 32, 20% held out for validation.
- **Framework:** Keras (TensorFlow), trained on Google Colab.

## Export pipeline

```
Keras model → TFLite converter → tan_model.tflite → tan_model.h
```

1. The Keras model is converted with `tf.lite.TFLiteConverter`, giving a float32 `tan_model.tflite` file.
2. The `.tflite` bytes are then written as a C array (`g_model[]`, `g_model_len`) in `tan_model.h`.

The `.tflite` file and the header contain exactly the same bytes: the serialized model graph and its trained weights.

## Inference

| Device | Model file | Runtime | Code |
|---|---|---|---|
| Arduino GIGA R1 | `tan_model.h` | TFLM (TensorFlow Lite for Microcontrollers), via the Chirale library | `.ino` sketch |
| Raspberry Pi 5 | `tan_model.tflite` | LiteRT (`ai-edge-litert`) | Python script |

On the Arduino, the sketch loads the model from the header, allocates a tensor arena, reads x from the Serial Monitor, runs the interpreter and prints the result next to the true `tanf(x)`.

## Results (Raspberry Pi 5, LiteRT)

| x | Model | tan(x) | Error |
|---|---|---|---|
| -1.20 | -2.5353 | -2.5722 | +0.0369 |
| -0.50 | -0.5399 | -0.5463 | +0.0064 |
| 0.00 | 0.0072 | 0.0000 | +0.0072 |
| 0.50 | 0.5519 | 0.5463 | +0.0056 |
| 1.00 | 1.5558 | 1.5574 | -0.0016 |
| 1.20 | 2.5131 | 2.5722 | -0.0590 |

- **Mean absolute error:** 0.011 over 200 points in [-1.2, 1.2].
- Accuracy is highest in the middle of the range (errors under 0.01) and lowest near ±1.2, where tan is steepest.
- Outputs match the Keras model from Colab (for example 0.5519 at x = 0.5), so the exported `.tflite` file reproduces the trained model.

## Notes

- Inputs outside [-1.2, 1.2] should be clamped, because the model was not trained there.
- The Arduino sketch must match the model's data type. A float32 model uses `input->data.f[]`; an int8 model needs quantize and dequantize steps.
- A stale header from an earlier int8 export caused wrong outputs on the GIGA, so always check which `.h` file is in the sketch folder.
