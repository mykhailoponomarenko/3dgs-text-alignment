import os
import json
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

import clip
from sentence_transformers import SentenceTransformer


DATA_DIR = "scene/labels"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def polygon_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(segmentation, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_crop(image, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = image[y1:y2+1, x1:x2+1]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    return Image.fromarray(crop)


def extract_visual_feature(pil_img):
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy().flatten()


def extract_text_embedding(text):
    return text_model.encode(text)

def build_dataset():
    scene_features = []
    text_embeddings = []
    captions_list = []
    metadata_list = []
    global_idx = 0

    json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])

    for jf in tqdm(json_files):
        json_path = os.path.join(DATA_DIR, jf)
        img_path = json_path.replace(".json", ".jpg")

        if not os.path.exists(img_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        for obj in data["objects"]:
            segmentation = obj["segmentation"]
            label = obj["category"]

            mask = polygon_to_mask(segmentation, h, w)
            pil_crop = extract_crop(image, mask)

            if pil_crop is None:
                continue

            vis_feat = extract_visual_feature(pil_crop)

            txt_feat = extract_text_embedding(label)

            scene_features.append(vis_feat)
            text_embeddings.append(txt_feat)
            captions_list.append(label)

            metadata_list.append({
                "index": global_idx,
                "image": img_path,
                "object_id": obj.get("id", global_idx),
                "label": label
            })

            global_idx += 1

    X = np.array(scene_features)
    Y = np.array(text_embeddings)

    print("Final shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)

    np.save("scene_features.npy", X)
    np.save("text_embeddings.npy", Y)
    np.save("labels.npy", np.array(captions_list, dtype=object))

    with open("labels.txt", "w") as f:
        for item in metadata_list:
            f.write(
                f"{item['index']} | {item['image']} | obj={item['object_id']} | {item['label']}\n"
            )

    print("Saved scene_features.npy and text_embeddings.npy")


if __name__ == "__main__":
    build_dataset()