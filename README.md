# 👗 StyleSense-Multimodal
 
### Where Vision Meets Style

FashionFusion-AI is an end-to-end multimodal deep learning project that combines image and text data to classify fashion items into categories and styles.

The project demonstrates how integrating visual features (images) with textual information (product attributes and descriptions) can significantly improve classification performance in real-world fashion applications.

---

## 🚀 Project Overview

This project follows a complete pipeline:

1. Web Scraping fashion data (images + attributes)
2. Data Cleaning and Preprocessing
3. Dataset Construction (CSV/JSON)
4. Multimodal Feature Extraction
5. Model Training (Image + Text)
6. Evaluation and Prediction

---

## 🧠 Key Features

- 📥 Web scraping for image and text data
- 🖼️ Automatic image downloading and storage
- 🗂️ Structured dataset generation (CSV/JSON)
- 🔗 Multimodal learning (Image + Text fusion)
- 🤖 Pretrained models (CNN + NLP)
- 📊 Training and evaluation pipeline
- ⚡ Scalable and modular architecture

---

---

## 📊 Dataset

The dataset is created through web scraping and includes:

- Image paths
- Product titles
- Categories
- Attributes (color, style, etc.)

### Example:

| image_path | title | category | color |
|-----------|------|---------|-------|
| img1.jpg  | Dress | Casual | Red |

---

## 🤖 Model Architecture

The model combines:

- 🖼️ **Image Encoder**  
  Pretrained CNN (ResNet / ViT)

- 📝 **Text Encoder**  
  NLP model (BERT / embeddings)

- 🔗 **Fusion Layer**  
  Concatenates image + text features

- 🎯 **Classifier**  
  Fully connected layers for prediction

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/StyleSense-Multimodal.git
cd StyleSense-Multimodal
pip install -r requirements.txt
