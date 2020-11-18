from torch.utils.data import Dataset
import torch


class SyntheticDataset(Dataset):
    """Synthetic dataset to test how well Soft TopK approximates Hard TopK."""

    def __init__(self, input_len, pooled_len, emb_size, data_size=10000, ordered=True,
                 device='cpu'):
        """
        Args:
            input_len (int): Length of the source BEFORE pooling
            pooled_len (int): Length of the source AFTER pooling.
            emb_size (int): Depth of the embeddings
        """
        self.input_len = input_len
        self.pooled_len = pooled_len
        self.emb_size = emb_size
        self.data_size = data_size
        self.ordered = ordered
        self.device = device

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """ generating synthetic data on the fly:
                input_embeddings:  input_len x emb_size
                input_scores:  input_len x 1
                output_scores:  pooled_len x 1    <- max(input_scores)
                output_embeddings:  pooled_len x emd_size      <- input_embeddings[input_scores.topk().indices]
         """
        input_embeddings = torch.rand((self.input_len, self.emb_size))*2-1

        input_scores = torch.rand(self.input_len)
        if self.ordered:
            input_scores = input_scores.sort(descending=True).values

        output_scores = input_scores[input_scores.topk(self.pooled_len).indices]
        output_embeddings = input_embeddings[input_scores.topk(self.pooled_len).indices]

        sample = {
            'input_embeddings': input_embeddings.to(self.device),
            'output_embeddings': output_embeddings.to(self.device),
            'input_scores': input_scores.unsqueeze(1).to(self.device),
            'output_scores': output_scores.unsqueeze(1).to(self.device),
        }

        return sample