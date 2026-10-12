"""TP6: convert both trained TP5 models to fully integer TFLite models."""
# %% 1. Imports and locate the TP5 models
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_DIR = ROOT / "models"
OUT = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# On Colab upload the two .keras files from TP5 when prompted.
missing = [name for name in ["mlp_model.keras", "cnn_model.keras"] if not (MODEL_DIR / name).exists()]
if missing:
    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError("Copy TP5 results/mlp_model.keras and cnn_model.keras into models/.") from error
    print("Upload:", missing)
    for name, content in files.upload().items():
        if Path(name).name in ["mlp_model.keras", "cnn_model.keras"]:
            (MODEL_DIR / Path(name).name).write_bytes(content)

# %% 2. Training-only representative data and independent test images
(train_images, _), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_x = train_images.astype(np.float32) / 255.0
test_x = test_images.astype(np.float32) / 255.0
rng = np.random.default_rng(42)
calibration_indices = rng.choice(54000, size=500, replace=False)
rows = []
tensor_info = {}

# %% 3. Convert MLP and CNN with full int8 input, output, and operators
for name in ["mlp", "cnn"]:
    model = keras.models.load_model(MODEL_DIR / f"{name}_model.keras", compile=False)
    calibration = train_x[..., None] if name == "cnn" else train_x
    evaluation_x = test_x[..., None] if name == "cnn" else test_x

    def representative_data():
        for index in calibration_indices:
            yield [calibration[index:index + 1]]

    # Save the same model without optimizer state for an H5 storage comparison.
    h5_path = OUT / f"{name}_model.h5"
    model.save(h5_path, include_optimizer=False)
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    saved_model_path = OUT / f"_{name}_saved_model"
    use_saved_model = False
    try:
        float_model = float_converter.convert()
    except (TypeError, AttributeError):
        # Some TensorFlow/Keras combinations cannot freeze Keras directly.
        # Export the same trained graph; this does not retrain the model.
        model.export(saved_model_path, format="tf_saved_model")
        use_saved_model = True
        float_model = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path)).convert()
    (OUT / f"{name}_float32.tflite").write_bytes(float_model)

    converter = (tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
                 if use_saved_model else tf.lite.TFLiteConverter.from_keras_model(model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_path = OUT / f"{name}_int8.tflite"
    model_path.write_bytes(converter.convert())

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    assert input_tensor["dtype"] == np.int8 and output_tensor["dtype"] == np.int8
    scale, zero_point = input_tensor["quantization"]
    assert scale > 0
    print(name, "input shape", input_tensor["shape"], "scale", scale, "zero point", zero_point)

    # Use the actual scale and zero point; do not assume pixel - 128.
    correct = 0
    for index, image in enumerate(evaluation_x):
        quantized = np.clip(np.rint(image / scale + zero_point), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_tensor["index"], quantized.reshape(input_tensor["shape"]))
        interpreter.invoke()
        output = interpreter.get_tensor(output_tensor["index"])
        correct += int(np.argmax(output[0]) == test_labels[index])
    keras_predictions = model.predict(evaluation_x, batch_size=128, verbose=0).argmax(axis=1)
    keras_accuracy = float(np.mean(keras_predictions == test_labels))
    int8_accuracy = correct / len(test_labels)
    float_size = (OUT / f"{name}_float32.tflite").stat().st_size
    int8_size = model_path.stat().st_size
    rows.append({
        "Model": name.upper(), "Keras_H5_MB": h5_path.stat().st_size / 1024**2,
        "Float32_TFLite_KB": float_size / 1024, "Int8_TFLite_KB": int8_size / 1024,
        "Float_to_int8_ratio": float_size / int8_size,
        "Keras_accuracy": keras_accuracy, "Int8_accuracy": int8_accuracy,
        "Accuracy_drop_percentage_points": (keras_accuracy - int8_accuracy) * 100,
        "Model_blob_below_512_KB": int8_size < 512 * 1024,
    })
    tensor_info[name] = {
        "input_shape": input_tensor["shape"].tolist(), "input_dtype": "int8",
        "input_scale": float(scale), "input_zero_point": int(zero_point),
        "output_shape": output_tensor["shape"].tolist(), "output_dtype": "int8",
        "output_quantization": list(output_tensor["quantization"]),
    }

# %% 4. Comparison and deployment-readiness report
results = pd.DataFrame(rows)
results.to_csv(OUT / "results.csv", index=False)
(OUT / "tensor_info.json").write_text(json.dumps(tensor_info, indent=2), encoding="utf-8")
print(results.to_string(index=False))
table = "| Model | Keras H5 MB | Float TFLite KB | Int8 TFLite KB | Size ratio | Keras accuracy | Int8 accuracy | Accuracy drop (percentage points) | Blob <512 KB? |\n|---|---|---|---|---|---|---|---|---|\n"
for row in rows:
    table += f"| {row['Model']} | {row['Keras_H5_MB']:.3f} | {row['Float32_TFLite_KB']:.3f} | {row['Int8_TFLite_KB']:.3f} | {row['Float_to_int8_ratio']:.2f} | {row['Keras_accuracy']:.4f} | {row['Int8_accuracy']:.4f} | {row['Accuracy_drop_percentage_points']:.2f} | {row['Model_blob_below_512_KB']} |\n"
report = f"""# TP6 report

{table}
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
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")
print("Saved both full-int8 models and REPORT.md in", OUT)
