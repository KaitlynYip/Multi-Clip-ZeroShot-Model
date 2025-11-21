# clipCIFAR.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
import open_clip
from tqdm import tqdm

# ==================== CONFIG ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16   # smaller for 4GB GPU
EPOCHS = 5
LR = 5e-5
CHECKPOINT_DIR = "clip_checkpoints"
DATA_DIR = r"D:\New folder\clip_Project\cifar_images"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

cifar_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class CIFARDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        caption = f"a photo of a {cifar_classes[label]}"
        return img, caption

def main():
    # ==================== LOAD CIFAR-10 ====================
    train_dataset_raw = CIFAR10(root=DATA_DIR, train=True, download=True)
    test_dataset_raw = CIFAR10(root=DATA_DIR, train=False, download=True)

    # ==================== LOAD CLIP ====================
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)
    model.train()

    # ==================== TRANSFORMS ====================
    train_dataset = CIFARDataset(train_dataset_raw, preprocess)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True  # safe on Windows with __main__ guard
    )

    # ==================== OPTIMIZER ====================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # ==================== TRAIN LOOP ====================
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images = images.to(DEVICE).float()
            texts = tokenizer(list(captions)).to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                img_feat, txt_feat, logit_scale = model(images, texts)
                logits_img = img_feat @ txt_feat.T * logit_scale.exp()
                logits_txt = logits_img.T
                labels = torch.arange(len(images), device=DEVICE)
                loss = (F.cross_entropy(logits_img, labels) + F.cross_entropy(logits_txt, labels)) / 2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/clip_cifar_epoch{epoch}.pt")

    print("\nTraining complete!")

if __name__ == "__main__":
    main()
