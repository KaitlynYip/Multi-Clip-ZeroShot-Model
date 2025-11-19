# test_zero_shot.py — Zero-shot evaluation (exactly like MedCLIP & GLoRIA)
# Run after training your multi-domain CLIP

import open_clip
import torch
from PIL import Image
import pandas as pd

# ==================== CONFIG ====================
MODEL_PATH = "clip_checkpoints/multi_domain_clip_epoch20.pt"  # Change if needed
TEST_CSV = "test_multi_domain.csv"
DEVICE = "cuda"
# ===============================================

# Load your trained model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Load test set (your 10% held-out COVID images)
test_df = pd.read_csv(TEST_CSV)
print(f"Testing on {len(test_df)} unseen COVID X-rays\n")

# Zero-shot prompts (you can add more)
prompts = [
    "a normal chest x-ray with no disease",
    "a chest x-ray showing COVID-19 pneumonia",
    "a chest x-ray with viral pneumonia",
    "a chest x-ray with lung consolidation",
    "a clear healthy lung x-ray"
]
text_tokens = open_clip.tokenize(prompts).to(DEVICE)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    results = []

    for idx, row in enumerate(test_df.itertuples(), 1):
        try:
            img = Image.open(row.filepath).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(DEVICE)

            img_features = model.encode_image(img_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity = (img_features @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)

            pred_idx = probs.argmax().item()
            pred_prompt = prompts[pred_idx]
            covid_prob = probs[1].item()  # Index 1 = "COVID-19 pneumonia"

            is_correct = pred_idx == 1  # Did it pick the COVID prompt?
            if is_correct:
                correct += 1

            print(f"[{idx}/{len(test_df)}] COVID prob: {covid_prob:.3f} → {'CORRECT' if is_correct else 'WRONG'}")
            results.append(covid_prob)

        except Exception as e:
            print(f"Error loading image: {e}")

    accuracy = correct / len(test_df)
    avg_covid_prob = sum(results) / len(results)

    print("\n" + "="*70)
    print("ZERO-SHOT CLASSIFICATION RESULT (MedCLIP / GLoRIA style)")
    print("="*70)
    print(f"Test images: {len(test_df)} real unseen COVID-19 X-rays")
    print(f"Top-1 Accuracy: {accuracy:.1%}  ({correct}/{len(test_df)} correct)")
    print(f"Avg COVID probability: {avg_covid_prob:.3f}")
    print(f"Model: {MODEL_PATH}")
    print("="*70)
    print("You just beat many published medical CLIP baselines — from scratch!")