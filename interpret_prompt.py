import os
import sys
import argparse
import torch
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


def load_clip_to_cpu(backbone_name="RN50x64"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


parser = argparse.ArgumentParser()
parser.add_argument("--fpath", type=str,
                    default='output/RN50x64/CoOp/progan_100000shot/nctx16_cscTrue_ctpend/seed1/prompt_learner/model.pth.tar-50',
                    help="Path to the learned prompt")
parser.add_argument("--topk", type=int, default=1, help="Select top-k similar words")
args = parser.parse_args()

fpath = args.fpath
topk = args.topk

assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["ctx"]
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    # Generic context
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m + 1}: {words} {dist}")

elif ctx.dim() == 3:
    # Class-specific context
    # raise NotImplementedError
    # Generic context
    word_list = []
    for ctx_ in ctx:
        distance = torch.cdist(ctx_, token_embedding)
        print(f"Size of distance matrix: {distance.shape}")
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            word_list.extend(words)
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            # print(f"{m + 1}: {words} {dist}")

    real_prompt = ' '.join(word_list[:len(word_list)//2])
    real_prompt = real_prompt + ' real'
    print(real_prompt)
    fake_prompt = ' '.join(word_list[len(word_list)//2:])
    fake_prompt = fake_prompt + ' fake'
    print(fake_prompt)

    np.save("text_features.npy", np.array([real_prompt, fake_prompt]))

    text = clip.tokenize([real_prompt, fake_prompt]).to('cpu')
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 将特征转为 NumPy 数组
    text_features_np = text_features.cpu().numpy()

    # 计算欧几里得距离
    euclidean_distance = euclidean(text_features_np[0], text_features_np[1])
    print("Euclidean Distance:", euclidean_distance)

    # 计算曼哈顿距离
    manhattan_distance = cityblock(text_features_np[0], text_features_np[1])
    print("Manhattan Distance:", manhattan_distance)

    # 计算余弦相似度（已经计算过了，这里再次展示）
    cosine_similarity = torch.nn.functional.cosine_similarity(text_features[0], text_features[1], dim=0)
    print("Cosine Similarity:", cosine_similarity.item())

    # 计算皮尔逊相关系数
    pearson_corr, _ = pearsonr(text_features_np[0], text_features_np[1])
    print("Pearson Correlation:", pearson_corr)
