/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <edgeAI_continousMotionDeployment_inferencing.h>

static const float features[] = {
    // copy raw features here (for example from the 'Live classification' page)
    // see https://docs.edgeimpulse.com/docs/running-your-impulse-arduino
    1.0300, 7.3800, 6.5800, 1.1800, 7.3400, 6.6300, 0.9900, 7.4600, 6.7800, 0.9200, 7.5900, 6.8200, 0.9200, 7.5900, 6.8200, 0.9900, 7.5100, 6.7200, 0.9400, 7.4800, 6.7200, 1.1300, 7.4000, 6.7300, 1.3200, 7.4700, 6.7000, 1.2500, 7.5500, 6.7900, 1.2500, 7.5500, 6.7900, 1.0100, 7.6200, 6.9100, 0.8700, 7.6600, 6.9600, 0.8800, 7.4500, 6.8800, 1.1600, 7.2400, 6.7100, 1.2200, 7.2900, 6.6000, 1.2200, 7.2900, 6.6000, 1.2300, 7.3100, 6.6000, 1.1400, 7.3100, 6.6500, 1.1100, 7.3700, 6.6400, 1.1500, 7.2800, 6.6300, 1.2000, 7.2700, 6.6300, 1.2000, 7.2700, 6.6300, 1.2700, 7.3600, 6.6500, 1.1600, 7.3700, 6.7100, 1.2000, 7.4400, 6.7400, 1.1600, 7.5000, 6.6900, 1.0800, 7.4200, 6.7300, 1.0800, 7.4200, 6.7300, 1.1700, 7.3400, 6.6700, 1.2000, 7.2800, 6.6200, 1.2700, 7.2400, 6.6500, 1.2400, 7.3700, 6.6700, 1.3100, 7.5400, 6.6600, 1.1800, 7.4700, 6.6600, 1.1800, 7.4700, 6.6600, 1.1400, 7.2900, 6.5500, 0.8700, 7.0700, 6.4900, 0.9900, 6.6900, 6.3300, 1.0000, 6.5300, 6.0300, 0.8600, 6.6200, 6.0400, 0.8600, 6.6200, 6.0400, 0.6500, 6.9200, 6.3700, 0.5300, 7.6000, 7.0500, 0.9700, 8.5100, 7.7100, 2.6400, 9.6800, 8.3200, 4.0600, 11.5500, 9.0800, 4.0600, 11.5500, 9.0800, 2.5600, 14.1400, 11.0700, 0.6900, 15.6000, 12.6700, -0.1300, 14.7200, 12.9500, 1.8700, 12.3400, 11.2700, 2.0700, 11.6400, 9.9900, 2.0700, 11.6400, 9.9900, 1.4000, 11.7200, 9.9400, 1.1100, 12.3200, 9.4900, 0.1200, 12.3900, 9.3600, -0.8500, 10.8600, 9.3200, -0.0100, 9.3400, 8.7500, -0.0100, 9.3400, 8.7500, 0.3900, 7.9700, 7.1600, -0.0200, 7.8100, 5.5200, -0.2700, 6.6600, 4.2600, -2.2600, 4.1100, 3.4200, -2.0600, 3.6100, 2.4000, -2.0600, 3.6100, 2.4000, -2.1500, 4.2500, 1.8200, -1.7700, 3.7100, 0.6000, -1.3600, 3.4900, -0.0600, -2.9400, 2.4700, -0.4800, -4.4600, 1.8800, -0.1700, -4.9700, 1.7900, -0.5300, -4.9700, 1.7900, -0.5300, -5.4300, 2.9300, -1.4900, -4.9200, 3.3500, -1.7400, -4.9000, 0.4400, 0.5600, -4.2000, -2.1100, 1.2100, -3.6300, -2.4500, -0.6800, -3.6300, -2.4500, -0.6800, -3.9100, -3.2100, -2.0300, -2.6100, -3.8700, -1.8400, 0.2000, -4.9700, -3.4700, 0.3100, -6.2700, -4.7200, -0.5200, -5.0900, -4.9200, -0.5200, -5.0900, -4.9200, 0.2300, -5.3600, -4.6700, 1.0400, -8.6300, -4.5400, 1.6500, -9.2200, -5.2400, 2.2800, -9.7600, -5.8100, 2.8600, -10.2600, -6.3900, 2.8600, -10.2600, -6.3900, 2.3200, -11.6600, -6.9100, 1.1700, -11.7100, -7.0100, 1.2800, -11.1400, -6.5400, 2.3000, -11.5600, -6.6200, 2.9200, -11.7200, -6.8900, 2.9200, -11.7200, -6.8900, 2.5200, -12.4200, -7.3400, 3.8600, -12.8000, -7.5000, 4.7500, -13.2100, -8.6700, 5.2500, -13.9600, -9.0200, 5.6300, -14.4400, -9.2100, 5.6300, -14.4400, -9.2100, 5.8300, -15.4700, -9.1800, 6.8100, -16.1000, -9.6700, 6.3900, -15.8500, -9.6100, 5.2500, -14.5500, -8.9700, 4.6200, -13.2400, -7.6100, 4.6200, -13.2400, -7.6100, 3.9500, -12.6600, -6.2000, 3.6700, -12.2900, -5.6800, 2.9400, -11.2900, -5.0000, 1.4900, -10.3100, -4.1000, 1.1700, -9.2800, -3.4900, 1.4200, -7.6900, -3.4000, 1.4200, -7.6900, -3.4000, 1.0000, -5.5600, -3.2600, -1.3600, -4.2300, -1.7000, -3.1800, -3.0300, -0.3000, -3.5600, -1.1600, -0.2800, -2.9800, 0.3300, -1.5200, -2.9800, 0.3300, -1.5200, -2.6300, 0.5700, -2.1700, -3.0700, 0.3300, -1.7800, -3.4000, 0.7500, -0.9300
};

/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");
}

/**
 * @brief      Arduino main function
 */
void loop()
{
    ei_printf("Edge Impulse standalone inferencing (Arduino)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("The size of your 'features' array is not correct. Expected %lu items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        delay(1000);
        return;
    }

    ei_impulse_result_t result = { 0 };

    // the features are stored into flash, and we don't want to load everything into RAM
    signal_t features_signal;
    features_signal.total_length = sizeof(features) / sizeof(features[0]);
    features_signal.get_data = &raw_feature_get_data;

    // invoke the impulse
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return;
    }

    // print inference return code
    ei_printf("run_classifier returned: %d\r\n", res);
    print_inference_result(result);

    delay(1000);
}

void print_inference_result(ei_impulse_result_t result) {

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#endif

}