"""
Model A - Multimodal Fashion Classifier
========================================
Architecture: BERT (text) + EfficientNet-B0 (image) -> Fusion -> Classifier
Dataset: final_dataset.csv + processed images folder
Output: trained_model_A.pth + evaluation_results_A.txt

Requirements:
    pip install torch torchvision transformers scikit-learn pandas pillow tqdm
"""

import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "csv_path":        "final_dataset.csv",
    "images_dir":      "images/",
    "output_dir":      "model_A_outputs/",
    "bert_model":      "bert-base-uncased",
    "efficientnet":    "efficientnet_b0",
    "max_text_len":    128,
    "image_size":      224,
    "batch_size":      16,
    "epochs":          10,
    "learning_rate":   2e-5,
    "dropout":         0.3,
    "hidden_dim":      512,
    "seed":            42,
    "val_size":        0.15,
    "test_size":       0.15,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Merge title + description into a single text field
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

    required = ["text", "image_name", "label"]
    for col in required:
        assert col in df.columns, f"Column '{col}' not found in CSV!"

    df = df.dropna(subset=required).reset_index(drop=True)

    label_list = sorted(df["label"].unique().tolist())
    label2id   = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label   = {idx: lbl for lbl, idx in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    print(f"Dataset size: {len(df)} samples")
    print(f"Classes ({len(label_list)}): {label_list}")
    return df, label2id, id2label

df, label2id, id2label = load_data(CONFIG["csv_path"])

# Train / Val / Test split
train_df, temp_df = train_test_split(
    df, test_size=(CONFIG["val_size"] + CONFIG["test_size"]),
    stratify=df["label_id"], random_state=CONFIG["seed"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=CONFIG["test_size"] / (CONFIG["val_size"] + CONFIG["test_size"]),
    stratify=temp_df["label_id"], random_state=CONFIG["seed"]
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ─────────────────────────────────────────────
# 3. DATASET CLASS
# ─────────────────────────────────────────────
class FashionDataset(Dataset):
    def __init__(self, dataframe, images_dir, tokenizer, max_len, transform, is_train=True):
        self.df         = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.transform  = transform
        self.is_train   = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text
        text = str(row["text"])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Image
        img_path = os.path.join(self.images_dir, row["image_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (CONFIG["image_size"], CONFIG["image_size"]), color=255)
        image = self.transform(image)

        # Label
        label = torch.tensor(row["label_id"], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image":          image,
            "label":          label
        }

# ─────────────────────────────────────────────
# 4. TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 5. TOKENIZER & DATALOADERS
# ─────────────────────────────────────────────
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])

train_dataset = FashionDataset(train_df, CONFIG["images_dir"], tokenizer, CONFIG["max_text_len"], train_transform)
val_dataset   = FashionDataset(val_df,   CONFIG["images_dir"], tokenizer, CONFIG["max_text_len"], eval_transform, is_train=False)
test_dataset  = FashionDataset(test_df,  CONFIG["images_dir"], tokenizer, CONFIG["max_text_len"], eval_transform, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

print(f"DataLoaders ready | Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

# ─────────────────────────────────────────────
# 6. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class MultimodalFashionClassifier(nn.Module):
    """
    Text branch  : BERT-base  -> 768-d CLS token
    Image branch : EfficientNet-B0 -> 1280-d pooled feature
    Fusion       : concat -> [2048] -> FC -> ReLU -> Dropout -> FC -> num_classes
    """
    def __init__(self, num_classes, bert_model_name, dropout=0.3, hidden_dim=512):
        super().__init__()

        # Text Encoder (BERT)
        print("Loading BERT model...")
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = 768

        # Image Encoder (EfficientNet-B0)
        print("Loading EfficientNet-B0...")
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.effnet_dim = 1280

        # Fusion + Classifier
        fusion_input_dim = self.bert_dim + self.effnet_dim  # 768 + 1280 = 2048
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        # Text features
        bert_output  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat    = bert_output.last_hidden_state[:, 0, :]  # CLS token -> [B, 768]

        # Image features
        img_feat     = self.image_encoder(image)               # -> [B, 1280, 1, 1]
        img_feat     = img_feat.flatten(1)                     # -> [B, 1280]

        # Fusion
        combined     = torch.cat([text_feat, img_feat], dim=1) # -> [B, 2048]
        logits       = self.fusion(combined)                   # -> [B, num_classes]

        return logits


num_classes = len(label2id)
model = MultimodalFashionClassifier(
    num_classes=num_classes,
    bert_model_name=CONFIG["bert_model"],
    dropout=CONFIG["dropout"],
    hidden_dim=CONFIG["hidden_dim"]
).to(DEVICE)

print(f"Model ready | Classes: {num_classes}")

# ─────────────────────────────────────────────
# 7. TRAINING SETUP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=2, factor=0.5
)

# ─────────────────────────────────────────────
# 8. TRAIN & EVAL FUNCTIONS
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images         = batch["image"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images         = batch["image"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask, images)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Starting Training")
print("="*55)

best_val_acc = 0.0
history      = []

for epoch in range(1, CONFIG["epochs"] + 1):
    print(f"\nEpoch {epoch}/{CONFIG['epochs']}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)

    scheduler.step(val_loss)

    history.append({
        "epoch":      epoch,
        "train_loss": round(train_loss, 4),
        "train_acc":  round(train_acc, 4),
        "val_loss":   round(val_loss, 4),
        "val_acc":    round(val_acc, 4),
    })

    print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
    print(f"  Val   -> Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "trained_model_A.pth"))
        print(f"  Best model saved (val_acc={val_acc*100:.2f}%)")

# ─────────────────────────────────────────────
# 10. FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Final Evaluation on Test Set")
print("="*55)

model.load_state_dict(torch.load(os.path.join(CONFIG["output_dir"], "trained_model_A.pth"), map_location=DEVICE))

test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)

label_names = [id2label[i] for i in range(num_classes)]
report      = classification_report(test_labels, test_preds, target_names=label_names)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")
print("\nClassification Report:")
print(report)

# ─────────────────────────────────────────────
# 11. SAVE RESULTS
# ─────────────────────────────────────────────
results_path = os.path.join(CONFIG["output_dir"], "evaluation_results_A.txt")
with open(results_path, "w") as f:
    f.write("=" * 55 + "\n")
    f.write("Model A - Evaluation Results\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
    f.write(f"Test Loss     : {test_loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Training History:\n")
    for h in history:
        f.write(str(h) + "\n")

history_path = os.path.join(CONFIG["output_dir"], "training_history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

label_map_path = os.path.join(CONFIG["output_dir"], "label_map.json")
with open(label_map_path, "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

print(f"\nResults saved to: {CONFIG['output_dir']}")
print(f"   - trained_model_A.pth")
print(f"   - evaluation_results_A.txt")
print(f"   - training_history.json")
print(f"   - label_map.json")
print("\nDone!")


# ─────────────────────────────────────────────
# 12. INFERENCE FUNCTION
# ─────────────────────────────────────────────
def predict(text, image_path, model, tokenizer, transform, device, id2label):
    """
    Takes a text string and image path, returns the predicted class label.
    Example:
        label = predict("blue running shoes", "images/shoes/img.jpg", model, tokenizer, eval_transform, DEVICE, id2label)
    """
    model.eval()
    with torch.no_grad():
        # Text
        encoding = tokenizer(
            text, max_length=CONFIG["max_text_len"],
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Forward
        logits     = model(input_ids, attention_mask, image)
        pred_id    = logits.argmax(dim=1).item()
        pred_label = id2label[pred_id]

    return pred_label
