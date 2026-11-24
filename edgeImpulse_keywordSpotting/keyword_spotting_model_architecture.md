# Keyword Spotting Model Architecture

This report documents a **1D Convolutional Neural Network (1D-CNN)** classifier for keyword spotting on edge hardware. *(Note: While the dataset includes three command/ambient categories—`turnon`, `turnoff`, and `background`—the configured model architecture features **2 output classes**: `background` and `turnoff`.)*

## Layer-by-Layer Breakdown

* **Input layer (650 features):** Receives a flat 1D vector of 650 extracted values from the Digital Signal Processing (DSP) block.
* **Reshape layer (13 columns):** Unflattens the 650-element vector into a 2D sequence of **50 time steps $\times$ 13 features (columns)** ($650 / 13 = 50$) so temporal patterns can be processed along the time axis.
* **1D conv / pool layer (8 filters, kernel size 3, 1 layer):** Applies 1D convolutions across adjacent time steps using a kernel window of length 3, followed by pooling to downsample along the time axis.
* **Dropout layer (rate 0.25):** Randomly zeroes out 25% of activations during training to prevent overfitting on small datasets.
* **1D conv / pool layer (16 filters, kernel size 3, 1 layer):** Extracts higher-level temporal features across the incoming channels using 16 filters with a kernel size of 3, followed by pooling.
* **Dropout layer (rate 0.25):** Applies a second 25% dropout stage for regularization.
* **Flatten layer:** Converts the multi-channel feature map back into a 1D vector before the final classification layer.
* **Output layer (2 classes):** Produces confidence scores for the two configured output classes (`background` and `turnoff`).

## How 1D Convolution & Channels Work

* **Direction of convolution:** Unlike a 2D convolution that slides across both height and width, a **1D convolution** slides a filter in one direction—along the **time steps**—while reading across all feature channels at each step.
* **Shape & channel progression:**
  $$\text{650 features} \rightarrow \text{50 time steps} \times \text{13 features per step}$$
  A kernel window of size 3 slides across those time steps. Running **8 filters** in parallel acts as **8 independent pattern detectors** over the same input, producing **8 output channels** per step.
* **Core relationship:**
  $$\text{Number of filters} = \text{Number of output channels}$$
  *(Similarly, the second convolutional layer's 16 filters read across those 8 incoming channels to produce 16 output channels.)*




