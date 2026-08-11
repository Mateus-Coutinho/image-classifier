# 🌿 Plant Disease Classification with Deep Learning

An end-to-end Computer Vision system for classifying plant leaf diseases using Deep Learning and **Transfer Learning** with the **MobileNetV2** architecture. The entire data pipeline and training process are optimized for high-performance GPU execution on an **NVIDIA GeForce RTX 3050** via **WSL2**.

---

## 🚀 Highlights & Results

* **Validation Accuracy:** **~93.1%** on the *PlantVillage* dataset (15 classes).
* **Transfer Learning:** Leveraged pre-trained **MobileNetV2** (ImageNet weights) for rapid convergence and high accuracy.
* **Hardware & Memory Optimization:**
  * Enabled **Mixed Precision (`float16`)** training to cut VRAM usage in half and utilize NVIDIA Tensor Cores.
  * Built an asynchronous `tf.data.Dataset` pipeline (`prefetch` and dynamic buffer management).
  * Implemented dynamic GPU memory growth (`set_memory_growth`) to prevent VRAM allocation overflows.

---

## 📐 Repository Structure

```text
image-classifier/
├── data/                  # Raw and processed dataset directory (Git-ignored)
├── models/                # Saved model checkpoints (.keras)
├── src/
│   ├── config.py          # Global hyperparameter and path configurations
│   ├── data_loader.py     # Asynchronous data loading and optimization pipeline
│   ├── train.py           # Model assembly, compilation, and training script
│   └── predict.py         # Inference script for single-image classification
├── .gitignore             # Version control exclusion rules
├── README.md              # Project documentation
└── requirements.txt       # Python environment dependencies
