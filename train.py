# ============================================================
# INSTALL DEPENDENCIES
# ============================================================
!pip install -q transformers timm torchvision pillow opencv-python

# ============================================================
# IMPORTS
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SwinModel
from PIL import Image
import torchvision.transforms as T
import os
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# CHARACTER TOKENIZER (ENGLISH + KANNADA)
# ============================================================
class CharTokenizer:
    def __init__(self):
        english = list("abcdefghijklmnopqrstuvwxyz")
        kannada = list("ಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ")
        chars = english + kannada

        self.vocab = {c: i+4 for i, c in enumerate(chars)}
        self.vocab["<pad>"] = 0
        self.vocab["<s>"] = 1
        self.vocab["</s>"] = 2
        self.vocab["<unk>"] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, char):
        return torch.tensor([1, self.vocab.get(char, 3), 2])

    def decode(self, ids):
        return "".join([self.inv_vocab.get(i, "") for i in ids])

    def __len__(self):
        return len(self.vocab)

tokenizer = CharTokenizer()
print("Vocab size:", len(tokenizer))

# ============================================================
# RAW DATASET (NO PREPROCESSING HERE)
# ============================================================
class FolderLabelDataset(Dataset):
    def __init__(self, root_dir, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        for label in os.listdir(root_dir):
            path = os.path.join(root_dir, label)
            if not os.path.isdir(path):
                continue
            for img in os.listdir(path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(path, img), label))

        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        tokens = self.tokenizer.encode(label)
        return image, tokens

# ============================================================
# MANUSCRIPT PREPROCESSING (INSIDE MODEL)
# ============================================================
def preprocess_tensor_manuscript(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 10
    )

    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    processed = torch.from_numpy(processed).permute(2, 0, 1).float() / 255.0
    return processed

# ============================================================
# SWIN ENCODER WITH INTERNAL PREPROCESSING
# ============================================================
class DonutSwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        self.hidden_dim = self.swin.config.hidden_size

    def forward(self, images):
        with torch.no_grad():
            images = torch.stack(
                [preprocess_tensor_manuscript(img) for img in images]
            ).to(images.device)

        return self.swin(pixel_values=images).last_hidden_state

# ============================================================
# TRANSFORMER DECODER
# ============================================================
class DonutDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )

        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, memory):
        x = self.embedding(input_ids)
        out = self.decoder(x, memory)
        return self.fc(out)

# ============================================================
# FULL DONUT MODEL
# ============================================================
class HandwrittenDonut(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = DonutSwinEncoder()
        self.decoder = DonutDecoder(vocab_size, self.encoder.hidden_dim)

    def forward(self, images, input_ids):
        memory = self.encoder(images)
        return self.decoder(input_ids, memory)

# ============================================================
# LOAD DATA
# ============================================================
dataset = FolderLabelDataset("/content/dataset/training_images", tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ============================================================
# TRAINING
# ============================================================
model = HandwrittenDonut(len(tokenizer)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, tokens in loader:
        images, tokens = images.to(device), tokens.to(device)
        outputs = model(images, tokens[:, :-1])

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            tokens[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# ============================================================
# INFERENCE
# ============================================================
def predict(image_path):
    model.eval()
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    memory = model.encoder(image)
    ids = torch.tensor([[1]]).to(device)

    for _ in range(5):
        out = model.decoder(ids, memory)
        next_id = out[:, -1].argmax(-1).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == 2:
            break

    print("Prediction:", tokenizer.decode(ids.squeeze().tolist()))
