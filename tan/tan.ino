// tan(x) inference on Arduino GIGA R1 WiFi using Chirale TensorFlowLite (TFLM)
// Works with BOTH float32 and int8-quantized models exported as tan_model.h
// Serial Monitor: 9600 baud, line ending "Newline" or "No line ending"

#include <Chirale_TensorFlowLite.h>
#include "tan.h"   // defines g_model[] and g_model_len

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr float kMinX = -1.2f;
constexpr float kMaxX = 1.2f;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("tan(x) inference example");
  Serial.println("Initializing TensorFlow Lite Micro...");

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match the library!");
    while (true);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed - increase kTensorArenaSize");
    while (true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input->type == kTfLiteFloat32) {
    Serial.println("Model type: float32");
  } else if (input->type == kTfLiteInt8) {
    Serial.println("Model type: int8 (quantized)");
  } else {
    Serial.println("Unsupported model input type.");
    while (true);
  }

  Serial.println("Initialization done.");
  Serial.println();
  Serial.println("Enter a number between -1.2 and 1.2:");
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  float x = line.toFloat();   // 0 if the text is not a valid number
  Serial.print("Your input: ");
  Serial.println(x, 3);

  if (x < kMinX) x = kMinX;
  if (x > kMaxX) x = kMaxX;
  Serial.print("Used input: ");
  Serial.println(x, 3);

  // Write the input (quantize first if the model is int8)
  if (input->type == kTfLiteFloat32) {
    input->data.f[0] = x;
  } else {
    int q = (int)lroundf(x / input->params.scale) + input->params.zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    input->data.int8[0] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Read the output (dequantize if the model is int8)
  float y;
  if (output->type == kTfLiteFloat32) {
    y = output->data.f[0];
  } else {
    y = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  }

  Serial.print("Inferred tan(");
  Serial.print(x, 3);
  Serial.print(") = ");
  Serial.println(y, 3);

  Serial.print("Actual   tan(");
  Serial.print(x, 3);
  Serial.print(") = ");
  Serial.println(tanf(x), 3);
  Serial.println();
}
