# 👗 StyleSense-Multimodal

> **Where Vision Meets Style**
> An end-to-end multimodal deep learning project for fashion product classification using both product images and textual metadata.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Summary

**StyleSense-Multimodal** is a complete machine learning project that combines **computer vision** and **natural language processing** to classify fashion products into meaningful categories and styles.

Traditional fashion classification systems often rely on either product images or textual descriptions. In real-world e-commerce platforms, however, both modalities contain important information:

* Images reveal shape, color, texture, and visual design.
* Text provides product names, descriptions, attributes, and category hints.

This project demonstrates how combining both sources through a multimodal learning pipeline can improve product understanding and classification performance.

---

## 🎯 Problem Statement

Fashion product classification is challenging because many products can look visually similar while belonging to different categories, and textual descriptions can be noisy, incomplete, duplicated, or inconsistent.

The goal of this project is to build a robust multimodal classifier that can predict fashion product categories using:

* Product images
* Product titles
* Product descriptions
* Structured attributes such as color, style, or category metadata

---

## 🚀 Key Features

* 📥 Web scraping pipeline for collecting product images and text metadata
* 🖼️ Automatic image downloading and local image storage
* 🧹 Data cleaning and preprocessing for both image and text data
* 🗂️ Structured dataset generation using CSV/JSON formats
* 🔠 Text preprocessing, tokenization, and sequence construction
* 🧠 Multiple multimodal deep learning architectures
* 🖼️ Image encoders based on CNNs and vision transformer-style models
* 📝 Text encoders based on BERT, GRU, and LSTM
* 🔗 Feature fusion using late-fusion neural layers
* 🎯 Multi-class fashion category classification
* 📊 Evaluation using accuracy, precision, recall, F1-score, confusion matrix, and classification reports
* ♻️ Reproducible training with seeded experiments
* ⚡ Modular notebooks for model experimentation and comparison

---

## 🧠 Project Architecture

The project follows a modular multimodal machine learning architecture. Each folder in the repository represents a specific stage of the pipeline, starting from raw scraped data and ending with trained models, evaluation results, and an optional frontend interface.

```text
StyleSense-Multimodal/
│
├── assets/images/                 # Downloaded fashion product images
│   ├── hats_headwear/
│   ├── hoodies_sweatshirts/
│   ├── pants_trousers/
│   ├── shoes/
│   ├── shorts/
│   └── tshirts_tops/
│
├── data/                          # Raw and processed dataset files
│   └── multimodal_dataset.csv
│
├── Datacleaning/                  # Data cleaning and preprocessing workflow
│   └── Multimodal_Dataset_Preprocessing.ipynb
│
│
│
├── Models/                        # Model notebooks, checkpoints, and variants
│   ├── Model A: BERT + EfficientNet-B0
│   ├── Model B: GRU + ResNet18
│   └── Model C: LSTM + ViT-style image branch
│
├── Result/                        # Model outputs and evaluation artifacts
│   ├── classification reports
│   ├── confusion matrices
│   ├── training curves
│   ├── validation metrics
│   └── prediction examples
│
├── frontend/                      # Optional demo interface
│   ├── image upload
│   ├── text input
│   ├── model prediction
│   └── predicted category display
│
├── scripts/                       # Helper scripts
│   ├── data preparation scripts
│   ├── training scripts
│   ├── evaluation scripts
│   └── inference scripts
│
├── tests/                         # Testing and validation
│   ├── dataset tests
│   ├── preprocessing tests
│   └── model input/output tests
│
├── docs/                          # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Ignored files and folders
```

### End-to-End Pipeline

```text
1. Web Scraping
   └── Collect product image URLs, product text, labels, and product links

2. Image Storage
   └── Download and organize product images inside assets/images by category

3. Data Cleaning
   └── Clean text, remove invalid records, validate images, and standardize labels

4. Dataset Construction
   └── Build multimodal_dataset.csv with image, text, label, metadata, and URLs

5. Feature Extraction
   ├── Image branch extracts visual features from product images
   └── Text branch extracts semantic features from product descriptions

6. Multimodal Fusion
   └── Concatenate image and text embeddings into one shared representation

7. Classification
   └── Predict the final fashion category using fully connected classifier layers

8. Evaluation
   └── Generate accuracy, F1-score, classification report, confusion matrix, and plots

9. Deployment / Demo
   └── Use the frontend to upload an image, enter product text, and display predictions
```

### Multimodal Learning Flow

```text
Product Image ──► Image Preprocessing ──► Image Encoder ──┐
                                                          │
                                                          ├──► Fusion Layer ──► Classifier ──► Predicted Category
                                                          │
Product Text  ──► Text Preprocessing  ──► Text Encoder  ──┘
```

### Architecture Logic

The system is designed around two parallel input branches:

| Branch        | Input                                      | Processing                                      | Output                           |
| ------------- | ------------------------------------------ | ----------------------------------------------- | -------------------------------- |
| Image Branch  | Product image from `assets/images/`        | Resize, normalize, augment, encode with CNN/ViT | Visual feature vector            |
| Text Branch   | Product text from `multimodal_dataset.csv` | Clean, tokenize, pad/encode with NLP model      | Text feature vector              |
| Fusion Branch | Image + text vectors                       | Concatenate and pass through dense layers       | Shared multimodal representation |
| Classifier    | Fused features                             | Fully connected layers + softmax                | Final product category           |

This architecture allows the model to learn from both visual appearance and textual product information, making predictions more reliable than using only one modality.

---

## 📊 Dataset

The dataset is built from scraped Nike fashion product listings and converted into a structured multimodal dataset stored as:

```text
data/processed/multimodal_dataset.csv
```

### Dataset Size

The final dataset contains:

```text
6,079 fashion product samples
8 structured columns
6 raw product categories
6 grouped model labels
```

### Dataset Columns

| Column        | Description                                    |
| ------------- | ---------------------------------------------- |
| `image`       | Original image URL collected during scraping   |
| `text`        | Cleaned textual input used by the text encoder |
| `label_raw`   | Original scraped product category              |
| `label`       | Final grouped class label used for training    |
| `text_length` | Character length of the cleaned text           |
| `word_count`  | Number of words in the cleaned text            |
| `image_name`  | Local saved image filename                     |
| `product_url` | Original product page URL                      |

### Example Row Format

| image                            | text                   | label_raw             | label       | image_name                    |
| -------------------------------- | ---------------------- | --------------------- | ----------- | ----------------------------- |
| `https://static.nike.com/...jpg` | `men fleece hoodie...` | `hoodies_sweatshirts` | `outerwear` | `hoodies_sweatshirts_001.jpg` |

### Class Distribution

| Raw Category          | Final Label   | Number of Samples |
| --------------------- | ------------- | ----------------: |
| `shoes`               | `footwear`    |             1,366 |
| `hoodies_sweatshirts` | `outerwear`   |             1,355 |
| `tshirts_tops`        | `tops`        |             1,185 |
| `hats_headwear`       | `accessories` |             1,139 |
| `shorts`              | `bottoms`     |               517 |
| `pants_trousers`      | `other`       |               517 |

### Text Statistics

| Statistic | Text Length | Word Count |
| --------- | ----------: | ---------: |
| Minimum   |          29 |          6 |
| Mean      |       96.83 |      15.20 |
| Median    |          93 |         14 |
| Maximum   |         225 |         30 |

### Image Folder Classes

The local image dataset is organized by raw product category:

```text
assets/images/
├── hats_headwear/
├── hoodies_sweatshirts/
├── pants_trousers/
├── shoes/
├── shorts/
└── tshirts_tops/
```

---

## 🏗️ Model Architectures

This project experiments with multiple multimodal model variants.

---

### 🅰️ Model A — BERT + EfficientNet-B0

**Notebook:** `Model_A(2).ipynb`

Model A uses a strong transformer-based text encoder and a CNN-based image encoder.

```text
Text Input  → BERT Encoder          ┐
                                      ├→ Fusion Layer → Fully Connected Classifier → Category
Image Input → EfficientNet-B0 Encoder┘
```

#### Components

| Component       | Description                                            |
| --------------- | ------------------------------------------------------ |
| Text Encoder    | BERT tokenizer + BERT model                            |
| Image Encoder   | EfficientNet-B0                                        |
| Fusion Strategy | Late fusion by concatenating image and text embeddings |
| Classifier      | Fully connected layers                                 |
| Framework       | PyTorch + Hugging Face Transformers                    |

#### Highlights

* Uses pretrained BERT for semantic text representation
* Uses EfficientNet-B0 for efficient visual feature extraction
* Applies leakage-aware splitting to reduce duplicate-text leakage
* Includes best-checkpoint saving based on validation performance
* Evaluates using accuracy, F1-score, classification report, and confusion matrix

---

### 🅱️ Model B — GRU + ResNet18

**Notebook:** `Model_B.ipynb`

Model B uses a custom recurrent text encoder and a CNN image encoder.

```text
Text Input  → Embedding Layer → GRU Encoder ┐
                                             ├→ Fusion MLP → Classifier → Category
Image Input → ResNet18 Encoder              ┘
```

#### Components

| Component       | Description                          |
| --------------- | ------------------------------------ |
| Text Encoder    | Token embedding + GRU                |
| Image Encoder   | ResNet18                             |
| Fusion Strategy | Concatenation followed by MLP layers |
| Classifier      | Fully connected neural network       |
| Framework       | PyTorch                              |

#### Highlights

* Lightweight alternative to transformer-based text modeling
* Uses pretrained ResNet18 for image understanding
* Supports weighted sampling for class imbalance handling
* Uses early stopping and cosine learning-rate scheduling
* Evaluates with weighted F1-score and confusion matrix

---

### 🅲 Model C — LSTM + ViT-Style Image Pipeline

**Notebook:** `Model_C(1).ipynb`

Model C uses a Keras/TensorFlow pipeline with an LSTM-based text encoder and a vision-transformer-style image branch.

```text
Text Input  → Tokenizer → Embedding → LSTM ┐
                                           ├→ Fusion Dense Layers → Softmax Classifier
Image Input → Patch Embedding / ViT Branch ┘
```

#### Components

| Component       | Description                                      |
| --------------- | ------------------------------------------------ |
| Text Encoder    | Tokenizer + padded sequences + LSTM              |
| Image Encoder   | ViT-style patch-based image branch               |
| Fusion Strategy | Concatenation of learned image and text features |
| Classifier      | Dense layers with softmax output                 |
| Framework       | TensorFlow / Keras                               |

#### Highlights

* Uses TensorFlow data pipelines for batching and prefetching
* Applies stratified 70/15/15 train-validation-test split
* Uses callbacks such as early stopping, model checkpointing, and learning-rate reduction
* Saves final model in `.keras` format

---

## 🔗 Multimodal Fusion Strategy

The general fusion approach used in this project is **late fusion**.

1. The image encoder converts product images into dense visual embeddings.
2. The text encoder converts product titles/descriptions into dense semantic embeddings.
3. The two embeddings are concatenated.
4. The fused vector is passed through fully connected layers.
5. A final classifier predicts the product category.

```text
image_features = ImageEncoder(image)
text_features  = TextEncoder(text)

combined_features = concat(image_features, text_features)
prediction = Classifier(combined_features)
```

Late fusion is practical because it allows each modality to be processed by a specialized pretrained or custom encoder before combining the learned representations.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Deep Learning Frameworks

* PyTorch
* TensorFlow / Keras

### Computer Vision

* Torchvision
* ResNet18
* EfficientNet-B0
* Vision Transformer-style patch embeddings
* PIL

### NLP / Text Modeling

* Hugging Face Transformers
* BERT
* GRU
* LSTM
* Tokenizer-based sequence modeling

### Data Processing & Evaluation

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📁 Project Structure

The repository is organized as follows:

```text
StyleSense-Multimodal/
│
├── .git/
├── .qodo/
├── .venv/
├── assets/
│   └── images/
│       ├── hats_headwear/
│       ├── hoodies_sweatshirts/
│       ├── pants_trousers/
│       ├── shoes/
│       ├── shorts/
│       └── tshirts_tops/
│
├── config/
├── data/
├── Datacleaning/
├── docs/
├── frontend/
├── Models/
├── Result/
├── scripts/
├── src/
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Folder Descriptions

| Folder / File      | Purpose                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `assets/images/`   | Stores downloaded product images grouped by category              |
| `config/`          | Configuration files for paths, hyperparameters, and experiments   |
| `data/`            | Raw and processed dataset files                                   |
| `Datacleaning/`    | Data cleaning scripts or notebooks                                |
| `docs/`            | Documentation, reports, and project notes                         |
| `frontend/`        | Optional user interface or demo application                       |
| `Models/`          | Saved model files, notebooks, or model definitions                |
| `Result/`          | Evaluation outputs, plots, reports, and prediction results        |
| `scripts/`         | Utility scripts for scraping, preprocessing, training, or testing |
| `src/`             | Main reusable source code                                         |
| `tests/`           | Unit tests and validation tests                                   |
| `requirements.txt` | Python dependencies                                               |
| `.gitignore`       | Files and folders excluded from Git tracking                      |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/StyleSense-Multimodal.git
cd StyleSense-Multimodal
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
tqdm
requests
beautifulsoup4
torch
torchvision
tensorflow
transformers
opencv-python
```

> Adjust the dependency versions depending on whether you run the PyTorch notebooks, TensorFlow notebooks, or both.

---

## 📥 Data Collection Pipeline

The project starts with scraping fashion product listings from online sources.

### Collected Fields

The scraper should collect fields such as:

* Product title
* Product image URL
* Product category
* Product description
* Color
* Style
* Price or brand, if available

### Example Scraping Output

```json
{
  "title": "Men's Running Shoes",
  "category": "shoes",
  "color": "Black",
  "image_url": "https://example.com/image.jpg",
  "description": "Lightweight running shoes with breathable mesh upper."
}
```

---

## 🧹 Data Preprocessing

The preprocessing stage prepares raw scraped data for model training.

### Text Processing

* Lowercasing
* Removing duplicated punctuation
* Removing extra spaces
* Handling missing text values
* Combining title and description into a single text field
* Optional removal of category-revealing keywords to avoid leakage

### Image Processing

* Download images from URLs
* Validate image files
* Remove broken or unreadable images
* Resize images to model-specific input size
* Normalize pixel values
* Apply image augmentations during training

### Label Processing

* Encode string labels into integer labels
* Save label mappings for inference
* Use stratified splitting to preserve class distribution

---

## ✂️ Train / Validation / Test Split

A recommended split is:

```text
Training set:   70%
Validation set: 15%
Test set:       15%
```

For better evaluation quality, the project should avoid data leakage by ensuring that near-duplicate or identical text samples do not appear across different splits.

---

## 🏋️ Training Workflow

### General Training Steps

1. Load the processed dataset.
2. Encode labels.
3. Split the dataset into train, validation, and test sets.
4. Build image transformations.
5. Tokenize or vectorize text.
6. Create multimodal dataloaders.
7. Initialize the multimodal model.
8. Train using cross-entropy loss.
9. Monitor validation performance.
10. Save the best checkpoint.
11. Evaluate on the test set.

### Example Training Command

```bash
python src/training/train.py \
  --model model_a \
  --data data/processed/multimodal_dataset.csv \
  --image_dir assets/images \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 2e-5
```

---

## 🔍 Inference

After training, the model can predict the category of a new product using an image and a text description.

### Example Prediction Command

```bash
python src/inference/predict.py \
  --image assets/images/shoes/example.jpg \
  --text "Black lightweight running shoes with breathable upper"
```

### Example Output

```text
Predicted category: shoes
Confidence: 0.94
```

---

## 📈 Evaluation Metrics

The project evaluates model performance using:

| Metric                | Purpose                                       |
| --------------------- | --------------------------------------------- |
| Accuracy              | Overall prediction correctness                |
| Precision             | Correctness of positive predictions per class |
| Recall                | Ability to retrieve all samples of a class    |
| F1-score              | Balance between precision and recall          |
| Macro F1              | Equal importance to all classes               |
| Weighted F1           | Accounts for class imbalance                  |
| Confusion Matrix      | Shows class-level mistakes                    |
| Classification Report | Detailed per-class evaluation                 |

---

### Recommended Visualizations

Include these artifacts in the `results/` directory:

* Training and validation loss curves
* Training and validation accuracy curves
* Confusion matrix heatmap
* Per-class F1-score comparison
* Sample correct predictions
* Sample incorrect predictions

---

## 🧪 Experiment Tracking

For a professional workflow, track each experiment with:

* Model name
* Encoder types
* Batch size
* Learning rate
* Number of epochs
* Image size
* Maximum text length
* Optimizer
* Scheduler
* Validation metric
* Test metric
* Checkpoint path

Example:

```text
Experiment: Model A
Text Encoder: BERT
Image Encoder: EfficientNet-B0
Batch Size: 32
Epochs: 10
Optimizer: AdamW
Scheduler: CosineAnnealingLR
Best Checkpoint: results/checkpoints/model_a_best.pt
```

---

## 🧯 Common Challenges & Fixes

| Challenge            | Explanation                             | Suggested Fix                                    |
| -------------------- | --------------------------------------- | ------------------------------------------------ |
| Broken images        | Some scraped image URLs may fail        | Validate images before training                  |
| Missing descriptions | Product text may be incomplete          | Fill missing values with empty strings or titles |
| Class imbalance      | Some categories may dominate            | Use class weights or weighted sampling           |
| Data leakage         | Duplicate text can appear across splits | Split by unique text groups where possible       |
| Overfitting          | Model memorizes training samples        | Use dropout, augmentation, early stopping        |
| Noisy labels         | Scraped categories may be inconsistent  | Clean labels and inspect samples manually        |
| Large models         | BERT/CNN models need GPU memory         | Reduce batch size or freeze encoders             |

---

## ✅ Reproducibility

To make experiments reproducible:

* Fix random seeds for Python, NumPy, PyTorch, and TensorFlow
* Use deterministic train/validation/test splits
* Save label encoders and tokenizers
* Save model checkpoints
* Log model hyperparameters
* Keep dataset versions consistent

Example:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## 🧾 Notebooks

| Notebook           | Description                                  |
| ------------------ | -------------------------------------------- |
| `Model_A.ipynb` | BERT + EfficientNet-B0 multimodal classifier |
| `Model_B.ipynb`    | GRU + ResNet18 multimodal classifier         |
| `Model_C.ipynb` | LSTM + ViT-style multimodal classifier       |

---

## 🧭 Future Improvements

* Add CLIP-based image-text embeddings
* Fine-tune larger vision transformers
* Use attention-based fusion instead of simple concatenation
* Add brand, price, and color as structured features
* Build a Streamlit or Gradio demo app
* Deploy the model as a REST API
* Add experiment tracking with MLflow or Weights & Biases
* Improve scraping robustness with retry logic and validation
* Add automated tests for preprocessing and inference
* Convert notebooks into reusable Python modules

---

## 🖥️ Demo App Idea

A simple user interface can allow users to upload a product image and enter a title or description.

```text
Input:
- Product image
- Product title
- Optional description

Output:
- Predicted category
- Confidence score
- Top-k predictions
```

Recommended tools:

* Streamlit
* Gradio
* FastAPI

---

## 📌 Example README Preview Image

You can add screenshots later:

```text
assets/demo/demo_prediction.png
assets/results/confusion_matrix_model_a.png
assets/results/training_curve_model_b.png
```



---

## 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your work.
5. Open a pull request.

```bash
git checkout -b feature/new-model
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙋 Author

**Project:** StyleSense-Multimodal
**Focus:** Multimodal deep learning for fashion product classification
**Domain:** Computer Vision, Natural Language Processing, E-commerce AI

---

## ⭐ Acknowledgements

This project uses concepts and tools from:

* PyTorch
* TensorFlow / Keras
* Hugging Face Transformers
* Torchvision pretrained models
* Scikit-learn
* Modern multimodal deep learning research

---

## 📚 Citation

If you use this project for academic or learning purposes, you may cite it as:

```bibtex
@misc{stylesense_multimodal,
  title        = {StyleSense-Multimodal: Fashion Product Classification using Image and Text Fusion},
  author       = {Your Name},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Multimodal deep learning project for fashion classification}
}
```

---

## ✅ Project Status

```text
Status: Active / Experimental
Current Stage: Model experimentation and evaluation
Next Step: Add final metrics, demo app, and deployment pipeline
```
