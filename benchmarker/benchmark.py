import torch

from models.successive_halving_topk import built_topk_selectors
from data.synthetic_dataset import SyntheticDataset
from time import time
from itertools import product
from pandas import DataFrame


class SingleResult:
    def __init__(self, input_len, pooled_len, emb_depth, pooler_name,
                 time_result, score_loss, emb_loss, cosine_sim, cosine_row, l2_row):
        self.n = input_len
        self.k = pooled_len
        self.emb_depth = emb_depth
        self.pooler_name = pooler_name
        self.time = time_result
        self.score_loss = score_loss
        self.emb_loss = emb_loss
        self.cosine_sim = cosine_sim
        self.cosine_row = cosine_row
        self.l2_row = l2_row

    def __getitem__(self, item):
        return SingleResult(
            self.n,\
            self.k,\
            self.emb_depth,\
            self.pooler_name,\
            self.time[item],\
            self.score_loss[item],\
            self.emb_loss[item],\
            self.cosine_sim[item],\
            self.cosine_row[item],\
            self.l2_row[item],
        )

    def __repr__(self):
        return f'====\n{self.pooler_name}\n==== {self.n} -> {self.k}  ' \
               f' (x{self.emb_depth})\n' \
               f' score_loss={self.score_loss},' \
               f' \n emb_loss={self.emb_loss},' \
               f' \n cos_sim={self.cosine_sim},' \
               f' \n cos_row={self.cosine_row},' \
               f' \n l2_row={self.l2_row},' \
               f' \n time_result={self.time} \n'

def max_rowwise(a, b, type='l2'):
    """Helper function to calculate metrics"""
    a = a.unsqueeze(1).expand(-1, a.shape[1], -1, -1)      # my approx
    b = b.unsqueeze(2).expand(-1, -1, a.shape[2], -1)      #
    if type == 'cosine':
        s = torch.cosine_similarity(a, b, dim=-1).max(dim=2).values     # must be dim 2, regarding matches to 'b', which is a target
    elif type == 'l2':
        s = ((a - b) ** 2).sum(-1).min(dim=2).values.mean()
    else:
        raise NotImplemented
    return s

def get_one_results(trainloader, topk, results):
    start_t = 0.0
    end_t = 0.0
    cosine_sim = []
    score_loss = []
    emb_loss = []
    cosine_row = []
    l2_row = []
    time_results = []
    for el_i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        sample = data

        # forward
        start_t = time()
        outputs = topk(sample['input_embeddings'], sample['input_scores'])
        end_t = time()

        #
        out_si = outputs[1].sort(descending=True).indices
        emb_sorted_by_scores = []
        for i in range(out_si.shape[0]):
            tmp = outputs[0][i, out_si[i]]
            emb_sorted_by_scores.append(tmp)
        emb_sorted_by_scores = torch.stack(emb_sorted_by_scores, dim=0)
        sorted_out_scores = outputs[1].sort(descending=True).values

        # Calculate losses
        cosine_sim.append(torch.cosine_similarity(emb_sorted_by_scores, sample['output_embeddings']).mean().item())
        score_loss.append(((sorted_out_scores - sample['output_scores'].squeeze()) ** 2).mean().item())
        emb_loss.append(((emb_sorted_by_scores - sample['output_embeddings']) ** 2).mean().item())
        cosine_row.append(max_rowwise(emb_sorted_by_scores, sample['output_embeddings'], 'cosine').mean().item())
        l2_row.append(max_rowwise(emb_sorted_by_scores, sample['output_embeddings'], 'l2').mean().item())
        time_results.append(end_t - start_t)

    # assign to results
    results.time = time_results
    results.score_loss = score_loss
    results.emb_loss = emb_loss
    result.cosine_sim = cosine_sim
    result.cosine_row = cosine_row
    result.l2_row = l2_row
    return results


def print_ugly_logs(results):
    """This is logging different metrics to the console.
    It is a temporary & ugly debugging solution, but is actually very handy."""
    our = list(filter(lambda x: x.pooler_name == 'our_topk', results))
    their = list(filter(lambda x: x.pooler_name == 'iter_topk', results))
    df = None
    for el in results:
        for j in range(len(el.l2_row)):
            if df is None:
                df = DataFrame.from_dict(el[j].__dict__, orient='index').transpose()
            else:
                df = df.append(DataFrame.from_dict(el[j].__dict__, orient='index').transpose())
    tstamp = int(time() * 10)
    df.to_csv(f'./benchmark_log_{tstamp}_{DEVICE}.csv')
    our_tt = sum([sum(el.time) / len(el.time) for el in our]) / len(our)
    our_sc = sum([sum(el.score_loss) / len(el.score_loss) for el in our]) / len(our)
    our_emb = sum([sum(el.emb_loss) / len(el.emb_loss) for el in our]) / len(our)
    our_cos = sum([sum(el.cosine_sim) / len(el.cosine_sim) for el in our]) / len(our)
    our_cosr = sum([sum(el.cosine_row) / len(el.cosine_row) for el in our]) / len(our)
    our_l2_row = sum([sum(el.l2_row) / len(el.l2_row) for el in our]) / len(our)
    their_tt = sum([sum(el.time) / len(el.time) for el in their]) / len(their)
    their_sc = sum([sum(el.score_loss) / len(el.score_loss) for el in their]) / len(their)
    their_emb = sum([sum(el.emb_loss) / len(el.emb_loss) for el in their]) / len(their)
    their_cos = sum([sum(el.cosine_sim) / len(el.cosine_sim) for el in their]) / len(their)
    their_cosr = sum([sum(el.cosine_row) / len(el.cosine_row) for el in their]) / len(their)
    their_l2_row = sum([sum(el.l2_row) / len(el.l2_row) for el in their]) / len(their)
    sumlog = ''
    sumlog += f'time\n OURS: {our_tt},  ITER: {their_tt}\n'
    sumlog += f'score_loss\n OURS: {our_sc},  ITER: {their_sc}\n'
    sumlog += f'emb_loss\n OURS: {our_emb},  ITER: {their_emb}\n'
    sumlog += f'cosine_sim\n OURS: {our_cos},  ITER: {their_cos}\n'
    sumlog += f'cosine_rowwise\n OURS: {our_cosr},  ITER: {their_cosr}\n'
    sumlog += f'l2_rowwise\n OURS: {our_l2_row},  ITER: {their_l2_row}\n'
    with open(f'./summary_log_{tstamp}.csv', 'w+') as log:
        log.write(sumlog)
    print(sumlog)
    print(f'Finished Benchmarking of Poolers and saved with timestamp '
          f': {tstamp} and :device {DEVICE}')


if __name__ == '__main__':
    INPUT_LENS = [2**i for i in range(4, 6, 1)]
    POOLED_LENS = [2**i for i in range(1, 3, 1)]
    EMB_DEPTHS = [2**i for i in range(5, 6)]       # The size seems not to influence results that much
    DATA_SIZE = 256  # number of data items in each test
    BATCH_SIZE = 16   # number of elements to be measured at once and averaged
    DEVICE = 'cpu'
    results = []
    i = 0
    for input_len, pooled_len, emb_depth in product(INPUT_LENS, POOLED_LENS, EMB_DEPTHS):
        if pooled_len >= input_len:     # sanitization
            continue
        our_topk, iter_topk = built_topk_selectors(input_len, pooled_len)
        trainset = SyntheticDataset(input_len, pooled_len, emb_depth, DATA_SIZE, device=DEVICE)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=0)
        for topk in [our_topk, iter_topk]:
            result = SingleResult(input_len, pooled_len, emb_depth, topk.name,
                         time_result=0.0, score_loss=0.0, emb_loss=0.0, cosine_sim=0.0,
                                  cosine_row=0.0, l2_row=0.0)
            result = get_one_results(trainloader, topk, result)
            results.append(result)
            print(result)
        i += 1
        print(f'************ finished tests: {i}')

    print_ugly_logs(results)
