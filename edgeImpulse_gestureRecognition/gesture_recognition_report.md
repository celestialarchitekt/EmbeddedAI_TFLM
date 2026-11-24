# Gesture Recognition Project

## Overview

This is a motion recognition demo. A small device with an accelerometer is held in the hand and moved in different ways. The accelerometer measures acceleration along three axes (X, Y and Z), and the model decides which movement just happened.

The project is a public Edge Impulse project that was cloned into my own account. The goal of this test is **deployment fidelity**: to check that a model built in the Edge Impulse cloud gives the same answers when it runs on the Arduino GIGA R1. The GIGA R1 is not an officially supported Edge Impulse board.

## Classes

The model chooses between four labels:

| Label | Movement |
|---|---|
| idle | the device is held still or barely moving |
| snake | the device moves along a zig-zag path |
| updown | the device moves up and down |
| wave | the device waves from side to side |

## Model

| Item | Details |
|---|---|
| Input | about 2 seconds of 3-axis accelerometer data (raw features) |
| Processing block | Spectral features |
| Learning blocks | Neural network classifier, anomaly detection |
| Size (int8, EON Compiler) | about 19 KB flash and 4.7 KB RAM |
| Reported accuracy | 99.67% in Edge Impulse Studio |
| Estimated latency | about 4 ms on Cortex-M7 at 216 MHz |

The model is TinyML: small enough to run on a microcontroller.

## How it works

1. **Raw features** are the accelerometer readings for one window of time, written as one list of numbers (x, y, z, x, y, z, ...).
2. The **Spectral features** block turns the raw window into frequency-based features.
3. The **neural network** gives a score for each label. The four scores add up to about 1, and the highest score is the predicted class.
4. The **anomaly block** scores how far the window's features lie from the clusters of training data.

On the GIGA, the Arduino library performs steps 2 to 4. The sketch only supplies the raw features.

## Clusters and anomaly score

Data points that lie together form a **cluster**. The training data forms one group per kind of movement.

The **anomaly score** is computed for each input window. It measures how far that window's features are from the nearest cluster learned during training. A higher score means the input is more unfamiliar. It does not say the input is wrong, and it is independent of the classifier: a window can be unfamiliar and still be classified with high confidence.

## Training and test data

The model learns from the **training set**. Labelled samples that were held back (the **test set**) are used to check it. Samples for idle, snake, updown and wave were passed to the model as test inputs.

Accuracy is the share of windows classified correctly, compared with their known labels. Studio reports it on validation data after training, and on the test set under **Model testing**.

## Method

1. In the Edge Impulse Studio, open **Deployment**, choose **Arduino library**, select **Quantized (int8)** and the **EON Compiler**, and click **Build**.
2. Add the `.zip` in the Arduino IDE and open the `static_buffer` example.
3. In Studio, open **Live classification**, load a sample, and copy its **Raw features**.
4. Paste them into the `features[]` array, select the Arduino GIGA R1 WiFi, and upload.
5. Read the predictions in the Serial Monitor (115200 baud) and compare them with the **Detailed result** table in Studio.

The difference between the GIGA output and the Studio table is the delta.

## Results

| Sample | GIGA R1 (int8) | Studio | Result |
|---|---|---|---|
| wave | wave 0.996, anomaly -0.270 | wave 1.00, anomaly -0.27 | match |
| idle | idle 0.996, anomaly -0.267 | not compared | correct label |
| snake | snake 0.996, anomaly -0.150 | not compared | correct label |
| updown (second sample) | updown 0.996, anomaly 0.369 | updown 1.00, anomaly 0.33 | match (anomaly differs slightly) |
| updown.33ihcn7n | idle 0.465, snake 0.191, updown 0.344, anomaly -0.002 | updown 1.00, anomaly -0.03 | **mismatch** |

- Timing on the board: DSP about 1 ms, inference under 1 ms.
- The mismatch on `updown.33ihcn7n` was not investigated. Possible causes are the pasted features or int8 quantization, but this was not confirmed.

## Conclusion

The Edge Impulse library compiled and ran on the GIGA R1. Four of the five samples classified correctly, and the wave sample reproduced Studio's scores. The mismatch on one updown sample is an open item.

## Limitations

- Five samples are far too few to measure accuracy. The board test checks agreement with Studio, not model accuracy.
- It is not known whether every sample used came from the test set.
- No live sensor data was used, because the GIGA R1 has no accelerometer.
- The timing is reported in whole milliseconds, so "0 ms" means under 1 ms.


## Architecture
- Input layer receives a 1D vector , with 33 features. 
- Dense layer 20 neurons, each neuron gets a weigheted sum of 33 features with a weight + bias.
- 20 outputs from the previous layer get fed into each of the 10 neuron with a weight + bias.
- Output has 4 classes: takes 10 output values and classifies them into 4 classes.




