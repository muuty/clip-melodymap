import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm

DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)


def extract_place_features():
    # extract features from images and save them to a json file
    features = {}
    images_dir = os.path.join(os.getcwd(), "images")

    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        try:
          image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
        except:
          print(f'Error: {image_name}')
          continue

        with torch.no_grad():
            image_features = model.encode_image(image)
            features[image_name] = image_features.cpu().numpy().tolist()

    with open("place_features.json", "w", encoding='utf-8') as fp:
        json.dump(features, fp)

def extract_music_features():
    # extract features from images and save them to a json file
    features = {}
    file = open('music.csv', 'r')
    # extract feature from lyrics, row[3]
    for row in tqdm(file):
        if '"' in row:
            lyrics = row.split('"')[1]
            row = row.replace(f'"{lyrics}"', lyrics.replace(',', ' '))
        try:
            artist, title, lyrics, genre = row.split(',')
        except:
            print(f'Error: {row}')
            continue

        # split lyrics by 40 length, and average the embeddings

        for i in range(0, len(lyrics), 40):
            text = clip.tokenize([lyrics[i:i+40]]).to("cpu")
            embeddings = []
            with torch.no_grad():
                text_features = model.encode_text(text)
                embeddings.append(text_features.cpu().numpy().tolist())
            text_features = torch.tensor(embeddings).mean(dim=0)
            name = f'{artist}-{title}'.replace(' ', '_')
            features[name] = text_features.cpu().numpy().tolist()

    with open("music_features.json", "w", encoding='utf-8') as fp:
        json.dump(features, fp)

def chech_feature_counts():
    with open("music_features.json", "r", encoding='utf-8') as fp:
        features = json.load(fp)

    for key, value in features.items():
        print(key)
    print(len(features))


if __name__ == "__main__":
    # extract_place_features()
    extract_music_features()
    # chech_feature_counts()