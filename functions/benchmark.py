from math import floor
from copy import deepcopy
import numpy as np
from tqdm.autonotebook import tqdm
from functions.easy_ensemble import EasyEnsemble


def generate_splits(N, k_folds, n_repeats, rng):
    # array_split spreads the N % k_folds leftover customers over the folds,
    # so every customer is in exactly one test fold per repeat.
    res = []
    for i in range(n_repeats):
        res.extend(np.array_split(rng.permutation(N), k_folds))
    return res

def get_splits_from_results(past_results):
    return list([split["test_indices"] for split in past_results])

def benchmark(dataset, models, k_folds=5, n_repeats=4, seed=None, verbose=False, fit_params={}, predict_params={}, past_results=None):
    rng = np.random.default_rng(seed=seed)
    N = dataset.X.shape[0]
    all_indices = np.arange(N)
    if past_results is None:
        splits = generate_splits(N, k_folds, n_repeats, rng)
    else:
        splits = get_splits_from_results(past_results)
    n_splits = len(splits)
    
    res = []
    
    for i, split in tqdm(enumerate(splits), total=n_splits):
        if verbose:
            print("Split {}/{}".format(i, n_splits))
                  
        data_test = dataset[split]
        # -> Shuffle also the train indices just to be sure
        train_indices = rng.permutation(np.setdiff1d(all_indices, split))
        data_train = dataset[train_indices]
        
        res_split = {"test_indices": split, "results": {}}
        
        for model_name, model in models.items():
            model = deepcopy(model)
            if verbose:
                print("Fitting", model_name)
            model.fit(data_train, **fit_params.get(model_name, {}))
            if verbose:
                print("Predicting", model_name)
            pred = model.predict(data_test, **predict_params.get(model_name, {}))
            
            res_split["results"][model_name] = {}
            res_split["results"][model_name]["pred"] = pred
            res_split["results"][model_name]["model"] = model
        res.append(res_split)
    return res




# import torch

# def awq_quantize(W, activations, bits=4, group_size=128, dtype=torch.float16):
#     W = W.to(dtype)
#     activations = activations.to(dtype)

#     # calibration math in fp32 for stability, even if weights are fp16
#     scale = activations.float().abs().mean(dim=0).clamp(min=1e-5)
#     W_scaled = (W.float() * scale)

#     qmax = 2 ** (bits - 1) - 1
#     out = torch.empty_like(W, dtype=torch.float32)

#     for i in range(0, W.shape[1], group_size):
#         w_group = W_scaled[:, i:i+group_size]
#         s = w_group.abs().max() / qmax
#         q = torch.clamp((w_group / s).round(), -qmax, qmax)
#         out[:, i:i+group_size] = (q * s) / scale[i:i+group_size]

#     return out.to(dtype)  # cast back to fp16 (or fp32) at the end

# # fp16 example
# W16 = torch.randn(512, 512, dtype=torch.float16)
# acts16 = torch.randn(64, 512, dtype=torch.float16)
# W_q16 = awq_quantize(W16, acts16, dtype=torch.float16)

# # fp32 example
# W32 = torch.randn(512, 512, dtype=torch.float32)
# acts32 = torch.randn(64, 512, dtype=torch.float32)
# W_q32 = awq_quantize(W32, acts32, dtype=torch.float32)

