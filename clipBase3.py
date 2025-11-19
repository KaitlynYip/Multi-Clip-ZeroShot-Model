# clipBase2.py — Multi-Domain CLIP Zero-Shot Model
# Based on https://github.com/mlfoundations/open_clip
# Trains from scratch; 90% train / 10% test split for COVID; space for other domains

import open_clip
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# ==================== CONFIG ====================
DEVICE = "cuda"
BATCH_SIZE = 16  # Safe for 4GB GPU
EPOCHS = 20
LR = 5e-5
CHECKPOINT_DIR = "clip_checkpoints"
TRAIN_CSV = "train_multi_domain.csv"
TEST_CSV = "test_multi_domain.csv"
# ===============================================

print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")

# ==================== DATASET ====================
class MultiDomainDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df):,} samples from {csv_file}")
        self.transform = transform

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['filepath']).convert("RGB")
            img = self.transform(img)
        except:
            img = torch.zeros(3, 224, 224)
        text = open_clip.tokenize([row['caption']])[0]
        return img, text

# ==================== BUILD TRAIN/TEST CSV ====================
def build_multi_domain_csv():
    train_rows = []
    test_rows = []

    # DOMAIN 1: COVID X-RAY (90/10 split)
    meta_path = Path('covid-chestxray-dataset/metadata.csv')
    if meta_path.exists():
        df = pd.read_csv(meta_path)
        covid = df[df['finding'].str.contains('COVID-19', case=False, na=False)].copy()
        covid['filepath'] = 'covid-chestxray-dataset/images/' + covid['filename']
        covid = covid[covid['filepath'].apply(lambda x: Path(x).exists() and str(x).lower().endswith(('.png', '.jpg', '.jpeg')))]
        covid = covid.drop_duplicates(subset=['filepath'])
        covid['caption'] = 'a chest x-ray showing COVID-19 pneumonia'
        train_df, test_df = train_test_split(covid, test_size=0.1, random_state=42)
        train_rows.extend(train_df[['filepath', 'caption']].to_dict('records'))
        test_rows.extend(test_df[['filepath', 'caption']].to_dict('records'))
        print(f"COVID: {len(train_df)} train / {len(test_df)} test")

    # SPACE FOR DOMAIN 2: Add MIMIC-CXR, CheXpert, etc.
    # Example:
    # mimic_train = Path('mimic-cxr/train')
    # for img in mimic_train.rglob('*.jpg'):
    #     train_rows.append({'filepath': str(img), 'caption': 'a chest x-ray'})
    # mimic_test = Path('mimic-cxr/val')
    # for img in mimic_test.rglob('*.jpg'):
    #     test_rows.append({'filepath': str(img), 'caption': 'a chest x-ray'})

    # SPACE FOR DOMAIN 3: Add CIFAR-10 or natural images
    # Example:
    # cifar_dir = Path('cifar_images')
    # for img in cifar_dir.glob('*.png'):
    #     train_rows.append({'filepath': str(img), 'caption': 'a photo of an airplane'})

    # SPACE FOR DOMAIN 4: Add your custom domain
    # Example:
    # custom_dir = Path('my_dataset')
    # for img in custom_dir.glob('*.jpg'):
    #     train_rows.append({'filepath': str(img), 'caption': 'your caption'})

    pd.DataFrame(train_rows).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(test_rows).to_csv(TEST_CSV, index=False)
    print(f"Saved train/test CSVs")

# ==================== MAIN ====================
if __name__ == '__main__':
    build_multi_domain_csv()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
    model.to(DEVICE)
    model.train()

    dataset = MultiDomainDataset(TRAIN_CSV, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for images, texts in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, texts = images.to(DEVICE), texts.to(DEVICE)

            img_feat, txt_feat, logit_scale = model(images, texts)
            logits_img = img_feat @ txt_feat.T * logit_scale.exp()
            logits_txt = logits_img.T

            labels = torch.arange(len(images), device=DEVICE)
            loss = (torch.nn.functional.cross_entropy(logits_img, labels) +
                    torch.nn.functional.cross_entropy(logits_txt, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/multi_domain_clip_epoch{epoch}.pt")

    print("Training complete!")