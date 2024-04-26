import json
import torch
from torch.nn import CosineSimilarity

def recommend_music(place_name: str) -> str:
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
        print(place_features.keys())
    with open("music_features.json", "r", encoding='utf-8') as fp:
        music_features = json.load(fp)
        print(music_features.keys())

    place_feature = place_features[place_name]
    similarities = {}
    # use distance as similarity
    for music_name, music_feature in music_features.items():
        # cosine similarity after flatten
        similarity = CosineSimilarity(dim=0)(torch.tensor(place_feature).flatten(), torch.tensor(music_feature).flatten())
        similarities[music_name] = similarity.item()
        # RuntimeError: a Tensor with 512 elements cannot be converted to Scalar

    similarities = sorted(similarities.items(), key=lambda x: x[1])
    return similarities[:5]

if __name__ == "__main__":
    print(recommend_music("여수_스카이타워.jpg"))
    print(recommend_music("여수_바다김밥.jpg"))
