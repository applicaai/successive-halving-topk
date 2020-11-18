from models.successive_halving_topk import TopKOperator, TopKConfig
import torch

# Input your settings
k = 256     # your k
n = 8192    # your n
depth = 32  # depth of the representations(vectors, embeddings etc.)

# Build operator and configure it
topk = TopKOperator()
cfg = TopKConfig(input_len=n,
                 pooled_len=k,
                 base=20,       # the bigger the better approximation, but can be unstable
                 )
topk.set_config(cfg)

# Prepare data (Note: sample embeddings from range [-1, 1], so that cosine similarity is fairly unbiased)
embeddings = torch.rand((1, n, depth)) * 2 - 1
scores = torch.rand((1, n, 1))

# Select with Soft TopK operator we proposed
out_embs, out_scores = topk(embeddings, scores)
out_scores.unsqueeze_(2)

# Make sure the shapes matches
assert out_embs.shape == torch.Size([1, k, depth])
assert out_scores.shape == torch.Size([1, k, 1])

# Assess quality fast by comparing hard top-1 select to our soft method
top1_hard = embeddings[0, scores.argmax(1).squeeze(), :]
top1_soft = out_embs[0, 0, :]
assert top1_hard.shape == top1_soft.shape
cosine_sim = torch.cosine_similarity(top1_hard, top1_soft, dim=0)   # this should be ~1.0
print(f'Approximation quality of Successive Halving TopK for top-1,'
      f' as measured by cosine similarity is {cosine_sim.item()}.')
