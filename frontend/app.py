import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from transformers import BertModel, BertTokenizer
import json
from PIL import Image
import os

st.set_page_config(page_title="Multimodal Fashion Classifier", page_icon="👗", layout="wide")

# Custom CSS for a beautiful, premium design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        color: white;
    }
    
    h1, h2, h3 {
        color: #e0e0ff !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.6);
        color: white;
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        animation: fadeIn 0.5s ease-out;
    }
    
    .label-text {
        font-size: 24px;
        font-weight: 700;
        color: #00f2fe;
        margin-bottom: 10px;
        text-transform: capitalize;
    }
    
    .confidence-text {
        font-size: 18px;
        color: #a0a0c0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Define Model Architecture
class MultimodalModel(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(MultimodalModel, self).__init__()
        
        # Image Encoder (EfficientNet-B0)
        efficientnet = models.efficientnet_b0(pretrained=False)
        self.image_encoder = nn.Sequential(
            efficientnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Text Encoder (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Fusion Network
        self.fusion = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images, input_ids, attention_mask):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        img_feat = self.image_encoder(images).flatten(1)
        combined_features = torch.cat([text_feat, img_feat], dim=1)
        output = self.fusion(combined_features)
        return output

# App Configuration
MODEL_PATH = "result/result/model_A_best.pth"
LABEL_MAP_PATH = "result/result/label_map.json"

@st.cache_resource
def load_model():
    # Load Label Map
    with open(LABEL_MAP_PATH, "r") as f:
        labels_info = json.load(f)
    id2label = {int(k): v for k, v in labels_info['id2label'].items()}
    
    num_classes = len(id2label)
    
    # Initialize Model
    model = MultimodalModel(num_classes=num_classes)
    
    # Load Weights
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, id2label

@st.cache_data
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

# Header
st.title("👗 Multimodal Fashion Classifier")
st.markdown("Upload an image of a fashion item and provide a brief text description. Our deep learning model will analyze both to accurately classify the product category.")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1. Image Input")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### 2. Text Input")
    text_input = st.text_area("Describe the item (optional but recommended):", placeholder="e.g. 'A red cotton t-shirt with short sleeves'")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_button = st.button("Classify Item", use_container_width=True)

# Prediction Logic
if predict_button:
    if uploaded_file is None:
        st.error("Please upload an image first!")
    else:
        with st.spinner("Analyzing image and text..."):
            try:
                model, tokenizer, id2label = load_model()
                transform = get_image_transforms()
                
                # Prepare Inputs
                image_tensor = transform(image).unsqueeze(0)
                
                text_to_encode = text_input if text_input.strip() else "fashion item"
                encoded_text = tokenizer(text_to_encode, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
                
                # Inference
                with torch.no_grad():
                    outputs = model(
                        images=image_tensor,
                        input_ids=encoded_text['input_ids'],
                        attention_mask=encoded_text['attention_mask']
                    )
                    
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    predicted_label = id2label[predicted_idx.item()]
                    confidence_score = confidence.item() * 100
                
                st.markdown(f"""
                    <div class="result-card">
                        <div class="label-text">{predicted_label}</div>
                        <div class="confidence-text">Confidence: {confidence_score:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show probability distribution
                st.markdown("### Class Probabilities")
                probs = probabilities[0].numpy()
                import pandas as pd
                
                # Create a nice looking dataframe for probabilities
                df_probs = pd.DataFrame({
                    "Category": [id2label[i] for i in range(len(id2label))],
                    "Probability": probs * 100
                })
                df_probs = df_probs.sort_values(by="Probability", ascending=False)
                
                st.bar_chart(df_probs.set_index("Category"))
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
