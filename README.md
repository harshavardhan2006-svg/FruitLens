# 🍎 FOOD FRESHNESS DETECTION USING DEEP LEARNING (MOBILENETV2 TRANSFER LEARNING)

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />

  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />

  <img src="https://img.shields.io/badge/Model-MobileNetV2-009688?style=for-the-badge&logo=google&logoColor=white" alt="MobileNetV2" />

  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />

  <img src="https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab" />

  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT" />

</p>

<p align="center">

  <strong>FruitLens</strong> is an intelligent Computer Vision and Deep Learning system designed to accurately classify produce as <strong>Fresh</strong> or <strong>Rotten</strong> in real-time. Built using <strong>MobileNetV2 transfer learning</strong>, FruitLens features an intuitive <strong>Streamlit web application</strong>, live camera/webcam capture, confidence scores, probability gauge meters, and an end-to-end <strong>Google Colab</strong> training pipeline with Hugging Face integration.

</p>

---

## 📑 Table of Contents

- [🌟 Overview & Motivation](#-overview--motivation)

- [⚡ Key Features](#-key-features)

- [🍎 Supported Fruit Categories](#-supported-fruit-categories)

- [🧠 Deep Learning Architecture & Pipeline](#-deep-learning-architecture--pipeline)

- [📊 Dataset & Data Leakage Prevention](#-dataset--data-leakage-prevention)

- [📁 Repository Structure](#-repository-structure)

- [🚀 Quick Start & Installation](#-quick-start--installation)

  - [Prerequisites](#prerequisites)

  - [1. Local Setup](#1-local-setup)

  - [2. Google Colab Training Workflow](#2-google-colab-training-workflow)

- [🌐 Free Cloud Deployment (Streamlit Cloud)](#-free-cloud-deployment-streamlit-cloud)

- [📈 Evaluation Metrics & Performance](#-evaluation-metrics--performance)

- [🛠️ Troubleshooting](#️-troubleshooting)

- [🎓 Course Project Details](#-course-project-details)

- [📜 License & References](#-license--references)

---

## 🌟 Overview & Motivation

Post-harvest food loss and microbial spoilage represent critical global economic, environmental, and food safety challenges:

- **Food Spoilage in Supply Chains:** Produce that begins to decay produces ethylene gas and releases fungal spores, accelerating decay in neighboring produce and contaminating entire crates.

- **Subjective Human Inspection:** Traditional manual quality checks rely on human visual inspection, which is slow, labor-intensive, fatigue-prone, and inconsistent across different observers and lighting conditions.

- **Expensive Hardware Alternatives:** Specialized systems such as hyperspectral imaging, electronic noses, and chemical sensors offer high precision but require expensive, proprietary equipment that is inaccessible to everyday consumers, small-scale grocers, and community food pantries.

**FruitLens solves this problem** by providing a lightweight, accessible, AI-powered computer vision pipeline. Using a smartphone camera or uploaded photograph, users can quickly check whether produce looks fresh or spoiled and see a confidence score for the prediction.

---

## ⚡ Key Features

- **📸 Dual Input Modes:**

  - **File Upload:** Drag-and-drop or upload high-resolution photos (`.jpg`, `.jpeg`, `.png`).

  - **Live Camera Capture:** Capture photos directly from a laptop webcam or mobile smartphone camera using Streamlit's native `st.camera_input`.

- **🧠 Transfer Learning:** Employs **MobileNetV2** pre-trained on 1.4 million ImageNet samples, utilizing inverted residual blocks and depthwise separable convolutions for high accuracy with minimal computational overhead.

- **⚡ Lightweight:** The MobileNetV2 backbone has about 2.26 million parameters (roughly 9 MB of float32 weights), which keeps load times and inference fast on CPU-only servers. The exact size of the saved `.keras` file depends on what is stored with it and should be checked after training.

- **📊 Real-Time Confidence Gauges:** Delivers both the final classification label and granular probability breakdown meters for both Fresh and Rotten classes.

- **⚠️ Low-Confidence Safety Safeguards:** Automatically alerts the user if confidence drops below 70%, which may indicate unclear lighting, occluded views, or an unsupported fruit type. This is only a heuristic: a neural network can be confidently wrong on inputs it was not trained on, so it is not a reliable out-of-distribution detector.

- **🔄 Two-Phase Progressive Fine-Tuning:**

  - *Phase 1:* Frozen backbone with a high learning rate (`1e-3`) to train the classification head.

  - *Phase 2:* Top 30 backbone layers unfrozen with a low learning rate (`1e-5`) for precise domain adaptation while preventing catastrophic forgetting.

- **🛡️ Strict Anti-Leakage Protocol:** Stratified train/validation/test split performed *prior* to data augmentation, ensuring no synthetic or augmented variants cross split boundaries.

- **💻 Dual User Interfaces:** Includes an interactive web app powered by **Streamlit** and a **Gradio** web preview embedded inside the Google Colab notebook.

---

## 🍎 Supported Fruit Categories

The model was trained on **8 popular fruit types** containing both pristine fresh specimens and real-world rot variations (spots, fungal growth, discoloration, wrinkled skin, bruising):

| Fruit | Fresh Indicators | Rotten Indicators |

|---|---|---|

| 🍏 **Apple** | Firm peel, uniform coloration, smooth skin | Soft spots, browning, microbial decay |

| 🍌 **Banana** | Bright yellow peel, intact stem | Dark sunken patches, extensive blackening, mushiness |

| 🍊 **Orange** | Vibrant citrus peel, firm texture | Mold spores, fungal patches, soft collapse |

| 🍇 **Grape** | Plump turgor, bloom intact, firm clusters | Wrinkled skin, juice leakage, mold clusters |

| 🍈 **Guava** | Crisp light green/yellow skin, firm body | Deep brown lesions, soft bruising, rot spots |

| 🫐 **Jujube** | Smooth glossy reddish-brown skin | Severe shriveling, localized decay, darkening |

| 🍎 **Pomegranate** | Leathery taut crimson rind, heavy crown | Discolored blotches, cracked soft rind, fungal spots |

| 🍓 **Strawberry** | Vivid red luster, firm seed structure | Gray mold (*Botrytis*), collapse, discoloration |

---

## 🧠 Deep Learning Architecture & Pipeline

FruitLens leverages transfer learning with **MobileNetV2**, an efficient convolutional neural network architecture tailored for edge and mobile vision tasks.

```

                  ┌──────────────────────────────┐

                  │   Input Photo (JPG / PNG)    │

                  └──────────────┬───────────────┘

                                 │

                                 ▼

                  ┌──────────────────────────────┐

                  │ Image Preprocessing Pipeline │

                  │  - EXIF Auto-Orientation     │

                  │  - Resize 224x224 (PIL)      │

                  │  - Rescaling to [-1, 1]      │

                  └──────────────┬───────────────┘

                                 │

                                 ▼

                  ┌──────────────────────────────┐

                  │     MobileNetV2 Backbone     │

                  │  (ImageNet Pre-trained Base) │

                  │  - Inverted Residual Blocks  │

                  │  - Depthwise Separable Convs │

                  │  - Linear Bottlenecks        │

                  └──────────────┬───────────────┘

                                 │

                                 ▼

                  ┌──────────────────────────────┐

                  │  GlobalAveragePooling2D()    │

                  │   (Spatial Dimensionality)   │

                  └──────────────┬───────────────┘

                                 │

                                 ▼

                  ┌──────────────────────────────┐

                  │        Dropout (0.3)         │

                  │   (Regularization Layer)     │

                  └──────────────┬───────────────┘

                                 │

                                 ▼

                  ┌──────────────────────────────┐

                  │   Dense(1, Sigmoid Output)   │

                  │   P(Rotten) ∈ [0.0, 1.0]     │

                  └──────────────┬───────────────┘

                                 │

                   ┌─────────────┴─────────────┐

                   ▼                           ▼

        P(Rotten) < 0.50            P(Rotten) ≥ 0.50

        ┌───────────────────┐       ┌───────────────────┐

        │  ✅ Class: FRESH  │       │  🚫 Class: ROTTEN │

        └───────────────────┘       └───────────────────┘

```

### Architectural Specifications

- **Input Dimension:** \(224 \times 224 \times 3\) RGB.

- **Normalization:** Scaled to \([-1, 1]\) by a `Rescaling(1/127.5, offset=-1)` layer inside the model (numerically equivalent to MobileNetV2 `preprocess_input`).

- **Pooling Layer:** `GlobalAveragePooling2D()` converts the \(7 \times 7 \times 1280\) feature maps into a 1280-dimensional feature vector.

- **Dropout Rate:** \(0.30\) to reduce co-adaptation of features.

- **Output Layer:** Single neuron with Sigmoid activation function outputting \(P(\text{Rotten})\).

- **Loss Function:** Binary Cross-Entropy:

  \(\mathcal{L} = -\frac{1}{N}\sum\_{i=1}^{N} \Big[ y_i \log(\hat{y}\_i) + (1 - y_i)\log(1 - \hat{y}\_i) \Big]\)

### Two-Phase Training Schedule

*(Values below match the configuration cells of `notebooks/Food_Freshness_Detection.ipynb`; `ReduceLROnPlateau` is not used.)*

| Parameter | Phase 1 (Feature Extraction) | Phase 2 (Fine-Tuning) |

|---|---|---|

| **Trainable Layers** | Top custom classifier head only (Backbone frozen) | Top 30 MobileNetV2 layers (BatchNorm layers kept frozen) + Classifier head |

| **Optimizer** | Adam (\(\beta_1 = 0.9, \beta_2 = 0.999\)) | Adam (\(\beta_1 = 0.9, \beta_2 = 0.999\)) |

| **Learning Rate** | \(1 \times 10^{-3}\) (\(0.001\)) | \(1 \times 10^{-5}\) (\(0.00001\)) |

| **Max Epochs** | 10 epochs | 8 epochs |

| **Batch Size** | 32 | 32 |

| **Callbacks** | `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)` | `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)` |

| **Class Weights** | Balanced, computed from the training split | Balanced, computed from the training split |

After both phases, the notebook compares the two saved checkpoints on the **validation** set (lower validation loss wins) and saves that model as `food_freshness_model.keras`. The test set is not used for this choice.

---

## 📊 Dataset & Data Leakage Prevention

FruitLens uses the peer-reviewed **Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality** (*Sultana, Jahan & Uddin, 2022, Data in Brief*), sourced directly from Hugging Face:

- **Hugging Face Repository:** [`Project-AgML/fresh_rotten_fruit_classification`](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification)

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

- **Subset Selected:** `raw` subset containing **3,200 pristine original photos** (1,600 Fresh + 1,600 Rotten).

- **Balance:** Exactly 50% Fresh and 50% Rotten across all 8 fruit categories.

### Strict Anti-Leakage Protocol

Some workflows augment the entire dataset *before* splitting, which can place augmented copies of the same photograph in both the training and test splits. FruitLens strictly implements **stratified splitting prior to augmentation**:

```

                       3,200 Raw Images

                              │

               ┌──────────────┼──────────────┐

               ▼              ▼              ▼

           Training       Validation        Test

            (70%)           (15%)          (15%)

         2,240 Images     480 Images     480 Images

               │              │              │

        [Augmentation]     [Clean]        [Clean]

        - Random Flip     No Augment     No Augment

        - Rotation ±36° (factor 0.1)

        - Zoom ±15%

        - Contrast ±10%

```

> **Caveat:** the split is made by image file, not by individual fruit. If the dataset contains several photographs of the same physical fruit, near-duplicates could still fall into different splits and make test scores optimistic.

---

## 📁 Repository Structure

```

FruitLens/

│

├── .streamlit/

│   └── config.toml               # Streamlit theme branding (Primary green, layout config)

│

├── model/

│   ├── README.md                 # Instructions for placing trained model weights

│   ├── food_freshness_model.keras# Trained Keras model (generated by the notebook; NOT yet committed)

│   └── metrics.json              # Test metrics for the sidebar (generated by the notebook; NOT yet committed)

│

├── notebooks/

│   └── Food_Freshness_Detection.ipynb # Complete Google Colab end-to-end training notebook

│

├── report/

│   └── FOOD FRESHNESS DETECTION USING DL.pdf # Full course academic project documentation report

│

├── .gitignore                    # Git exclusions (virtualenvs, cache, checkpoints)

├── app.py                        # Streamlit web application frontend & inference engine

├── LICENSE                       # MIT License

├── README.md                     # Comprehensive project documentation

└── requirements.txt              # Production Python package dependencies

```

---

## 🚀 Quick Start & Installation

### Prerequisites

- Python **3.10, 3.11, or 3.12** installed on your system.

- `pip` package manager.

- A modern web browser (Chrome, Firefox, Safari, Edge).

### 1. Local Setup

Clone the repository and set up a virtual environment:

```bash

# Clone the repository

git clone https://github.com/JAN-VIKASH/FruitLens.git

cd FruitLens

# Create a virtual environment

python -m venv .venv

# Activate the virtual environment

# On Windows (PowerShell):

.venv\Scripts\Activate.ps1

# On Windows (CMD):

.venv\Scripts\activate.bat

# On macOS / Linux:

source .venv/bin/activate

# Install required dependencies

pip install -r requirements.txt

# Run the Streamlit web application

streamlit run app.py

```

The app will automatically launch in your browser at:

👉 **`http://localhost:8501`**

---

### 2. Google Colab Training Workflow

You can train or fine-tune the model with free GPU acceleration directly on Google Colab **without needing to manually download or upload any dataset files**:

1. Open [Google Colab](https://colab.research.google.com).

2. Click **File** ➔ **Upload notebook**, and upload `notebooks/Food_Freshness_Detection.ipynb`.

3. In the Colab menu, select:

   **Runtime** ➔ **Change runtime type** ➔ **T4 GPU** ➔ **Save**.

4. Click **Runtime** ➔ **Run all** (or execute cells sequentially).

5. The notebook will:

   - Automatically download the dataset from Hugging Face Hub.

   - Perform stratified train/validation/test splitting.

   - Run Phase 1 training (head training) and Phase 2 fine-tuning.

   - Generate confusion matrices, ROC curves, and classification metrics.

   - Launch an inline **Gradio** web app.

   - Package `food_freshness_model.keras`, `metrics.json`, the figures (`training_curves.png`, `confusion_matrix.png`, `roc_curve.png`) and `per_fruit_accuracy.csv` into a downloadable `project_outputs.zip` (optional last cell).

6. Unzip `project_outputs.zip` and place `food_freshness_model.keras` and `metrics.json` into your local `model/` directory.

---

## 🌐 Free Cloud Deployment (Streamlit Cloud)

You can deploy FruitLens live on the web for free using **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account (`JAN-VIKASH/FruitLens`).

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

3. Click **Create app** (or **New app**).

4. Configure the deployment settings:

   - **Repository:** `JAN-VIKASH/FruitLens`

   - **Branch:** `main`

   - **Main file path:** `app.py`

5. Click **Advanced settings**:

   - Set **Python version:** `3.11` *(Recommended for optimal TensorFlow compatibility)*.

6. Click **Deploy!**

7. Within 2-3 minutes, your live web application will be accessible worldwide via a custom `.streamlit.app` URL.

---

## 📈 Evaluation Metrics & Performance

Model performance is rigorously measured on the independent, unseen test split (\(N = 480\) images):

\(\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}\)

\(\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)

> **Status:** this repository snapshot contains no executed notebook output, trained model or `metrics.json`, so no result values are quoted here. Run the notebook to generate them; any numbers in the project report must match that output.

### Why Rotten Recall Matters

In food freshness detection, **false negatives** (predicting a rotten fruit as fresh) pose food safety hazards and cause fungal decay to spread across storage batches. Conversely, a false positive simply prompts manual human verification. Recall for the Rotten class is therefore reported alongside accuracy. Note that training itself monitors **validation loss** (with balanced class weights) rather than recall, and the decision threshold is fixed at 0.5; tuning the threshold for higher Rotten recall is listed as future work.

When `model/metrics.json` is generated, the Streamlit app sidebar dynamically parses and renders the official test accuracy, precision, recall, and F1 metrics.

---

## 🛠️ Troubleshooting

| Issue / Error | Probable Cause | Recommended Resolution |

|---|---|---|

| **"No trained model found"** | `model/food_freshness_model.keras` has not been placed in the folder. | Train the model using the notebook or upload the `.keras` file directly through the app's upload interface. |

| **"Model could not be loaded"** | Discrepancy between the TensorFlow version used during training and inference. | In `requirements.txt`, pin `tensorflow-cpu` to the exact version printed by the notebook's "Save the trained model" step (section 12). |

| **"Low confidence warning"** | The uploaded image is blurry, poorly lit, or contains an unsupported fruit type. | Ensure the fruit is centered, well-lit, and belongs to one of the 8 trained categories. |

| **`use_container_width` warning or error** | Newer Streamlit releases deprecate/remove this argument. | `app.py` no longer passes it to `st.image`; keep `app.py` and `requirements.txt` in sync. |

| **Streamlit Cloud Build Error** | Selected an unsupported Python version (e.g. Python 3.13). | In Streamlit Cloud **Advanced settings**, set the Python runtime explicitly to **Python 3.11**. |

---

## 🎓 Course Project Details

- **Institution:** Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT)

- **Department:** Department of Computer Science & Engineering

- **Academic Year:** 2026-2027

- **Course:** Deep Learning Laboratory / Course Project

- **Project Title:** Food Freshness Detection Using Deep Learning (MobileNetV2 Transfer Learning)

- **Faculty Guide / Mentor:** Ms. G. Lalitha Kumari

- **Team Members:**

  1. K. Jan Vikash (23501A0571)

  2. K. Sruthi (23501A0573)

  3. K. Yoshita Navya (23501A0581)

  4. K. Harsha Vardhan (23501A0584)

---

## 📜 License & References

### License

This project is open-source and released under the [MIT License](LICENSE).

### Academic Citations

1. **Dataset:**

   > Sultana, N., Jahan, M., & Uddin, M. S. (2022). *An extensive dataset for successful recognition of fresh and rotten fruits.* **Data in Brief**, 44, 108552. DOI: [10.1016/j.dib.2022.108552](https://doi.org/10.1016/j.dib.2022.108552)

2. **Backbone Architecture:**

   > Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 4510-4520.

3. **Streamlit Framework:**

   > Streamlit Documentation. https://docs.streamlit.io/

---