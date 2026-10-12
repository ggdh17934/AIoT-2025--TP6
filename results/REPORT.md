# TP6 report

| Model | Keras H5 MB | Float TFLite KB | Int8 TFLite KB | Size ratio | Keras accuracy | Int8 accuracy | Accuracy drop (percentage points) | Blob <512 KB? |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.913 | 921.641 | 243.461 | 3.79 | 0.8693 | 0.8664 | 0.29 | True |
| CNN | 0.241 | 225.816 | 63.375 | 3.56 | 0.8807 | 0.8790 | 0.17 | True |

The two TP5 architectures are converted to full integer TFLite: int8 model
input/output and integer kernels; biases and some metadata may use other integer
widths. Calibration uses 500 **training** images, seed 42. Accuracy is measured
on all 10,000 independent Fashion-MNIST test images. Quantization computes
`q = round(real / scale + zero_point)`, clipped to [-128,127]. Dequantization
is `real = (q - zero_point) * scale`. Uniform output quantization preserves argmax.
Size ratios compare float and int8 TFLite files, avoiding optimizer-state overhead.
The script first uses from_keras_model as requested. If the installed TensorFlow/
Keras combination rejects direct freezing, it exports the identical trained
graph and converts from_saved_model instead; no retraining or model change occurs.

TFLite/LiteRT stores a compact graph for a lighter runtime. PTQ calibrates the
activation ranges and lowers numerical precision, often reducing storage at the
cost of some accuracy. Runtime speedups depend on the actual board and kernels.

XIAO ESP32-S3: dual-core LX7, up to 240 MHz, 512 KB internal SRAM. The standard
XIAO ESP32-S3/Sense described by Seeed has **8 MB Flash and 8 MB PSRAM**; PSRAM
is external working memory and is not Flash. Board revisions/variants differ.
The table's blob comparison answers the assignment's approximate size question,
but does **not** prove runtime fit. Model weights can remain in Flash while
the tensor arena, stacks and activations occupy RAM. Measure arena allocation
and free heap on the target. The CNN's smaller blob makes it a useful starting
point; under-100-ms inference cannot be promised from clock speed alone.

No ESP32 timing or peak RAM measurements are claimed. TP7 supplies firmware
which prints arena usage and actual inference time when run.

Sources:
- https://github.com/ahmed3991/AIoT-2025/blob/main/TP6/TP6.md
- https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/
- https://ai.google.dev/edge/litert/models/post_training_integer_quant
