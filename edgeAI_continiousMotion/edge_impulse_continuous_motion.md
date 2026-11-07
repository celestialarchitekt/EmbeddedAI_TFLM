# Edge Impulse: Continuous Motion Recognition on Arduino GIGA R1

## About Edge Impulse

Edge Impulse is a web platform for building and deploying machine learning on edge devices such as microcontrollers. It also hosts public projects that anyone can browse and clone.

Its workflow covers:

1. **Collect or upload data:** sensor readings, images or audio.
2. **Extract features** from the raw data with a signal-processing block.
3. **Train a model** in Edge Impulse's cloud.
4. **Test, optimize and quantize** the model.
5. **Check resource use:** estimated latency, RAM and flash for a chosen target chip.
6. **Deploy** it as an Arduino library, a C++ library, or firmware for supported boards.

## Project

A public gesture-recognition project was cloned into my own Edge Impulse account instead of building a dataset from scratch.

| Item | Details |
|---|---|
| Input | 3-axis accelerometer data (accX, accY, accZ) |
| Classes | idle, snake, updown, wave |
| Processing block | Spectral features |
| Learning blocks | Neural network classifier, anomaly detection |
| Reported accuracy | 99.67% (in Edge Impulse Studio) |
| Latency estimate | about 4 ms on Cortex-M7 at 216 MHz |
| RAM / flash estimate | about 4.7 KB RAM, about 19 KB flash (int8) |

## Deployment

1. Open the **Deployment** tab.
2. Choose **Arduino library** as the deployment target.
3. Choose the **EON Compiler** as the inference engine (less RAM and flash at the same accuracy).
4. Select **Quantized (int8)** and click **Build**. This produces a `.zip` file.
5. In the Arduino IDE, install it with **Sketch > Include Library > Add .ZIP Library**.
6. Open the `static_buffer` example from **File > Examples**.

## Getting test input

The GIGA R1 has no accelerometer, so a recorded sample is used instead of live data.

1. Open **Live classification** in Edge Impulse Studio.
2. Load an existing test sample (here `wave.33ii1nhk`, label `wave`).
3. Copy its **Raw features**.
4. Paste them into the `features[]` array in `static_buffer.ino`.

## Result on the GIGA R1

Uploaded with the Arduino GIGA R1 WiFi selected, read at 115200 baud in the Serial Monitor:

| Output | GIGA R1 (int8) | Studio (float32) |
|---|---|---|
| wave | 0.996 | 1.00 |
| idle, snake, updown | 0.000 | 0 |
| Anomaly score | -0.270 | -0.27 |
| Timing | DSP 1 ms, inference under 1 ms | |

The prediction and the anomaly score match Edge Impulse Studio, so the feature extraction and the neural network behave on the board as they do in the cloud. The small gap between 0.996 and 1.00 comes from int8 versus float32.

## Notes and limitations

- The GIGA R1 is not an officially supported Edge Impulse board. The Arduino library export worked on it anyway, since it uses the same STM32H747 chip as the supported Portenta H7.
- Only the `wave` sample has been tested so far. The other classes still need to be checked with idle, snake and updown samples, preferably from the test set.
- Live classification would need an external accelerometer connected to the GIGA.
- The timing is reported in whole milliseconds, so "0 ms" means under 1 ms.
