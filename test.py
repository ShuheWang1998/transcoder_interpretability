# import torch


# x = torch.randn(2,2,requires_grad=True)

# print("x:",x)

# y = torch.randn(2,2,requires_grad=True)

# print("y:",y)

# w = x@y

# print("w:",w)

# z = w + x + y

# print("z:",z)

# features_grad = 0.

# def extract(g):
#     global features_grad
#     features_grad = g

# w.register_hook(extract)

# z[0][0].backward()

# print(x.grad)
# print(y.grad)
# print(features_grad)

# print(x+1)
# print(y+1)

# x_t = x.transpose(0,1)
# print("x_t:",x_t)
# y_t = y.transpose(0,1)
# print("y_t:",y_t)

# print(x_t + 1)
# print(y_t + 1)

# # answer = y@x.grad.inverse() + x@y.grad.inverse()
# # print("answer:",answer)
# # print("answer.inverse():",answer.inverse())

# part_1 = x.grad@x
# part_2 = y@y.grad

# print("part_1:",part_1)
# print("part_2:",part_2)

# part_3 = x.grad@y.grad

# print("part_3:",part_3)

# answer = part_3@((part_1 + part_2).inverse())

# print("answer:",answer)

# answer = ((part_1 + part_2).inverse())@part_3

# print("answer:",answer)


# x = torch.randn(2,3,requires_grad=True)

# print("x:",x)

# y = torch.randn(3,2,requires_grad=True)

# print("y:",y)

# z = x@y

# print("z:",z)

# z[0][0].backward()

# print(x.grad)

# print(y.grad)

# x = torch.randn(2,3,requires_grad=True)

# print("x:",x)

# print(x.inverse())

# previous_vector = torch.randn(1,2,3)

# now_vector = torch.randn(1,2,3)

# print("previous_vector:",previous_vector)
# print("now_vector:",now_vector)

# previous_vector = torch.nn.functional.log_softmax(previous_vector, dim=-1)
# now_vector = torch.nn.functional.softmax(now_vector, dim=-1)

# print("previous_vector:",previous_vector)
# print("now_vector:",now_vector)
# print("previous_vector.shape:",previous_vector.shape)
# print("now_vector.shape:",now_vector.shape)

# kl_div = torch.nn.functional.kl_div(previous_vector, now_vector, log_target=False, reduction='mean', reduce=False)

# print("kl_div:",kl_div)
# print("kl_div.shape:",kl_div.shape)

# mean_kl_div = torch.mean(kl_div, dim=-1)

# print("mean_kl_div:",mean_kl_div)
# print("mean_kl_div.shape:",mean_kl_div.shape)



# a = torch.randn(2,3,4)

# print("a:",a)

# index_i = torch.tensor([0, 1])

# index_j = torch.tensor([1, 2])

# print("a[index_i, index_j, :]:", a[index_i, index_j, :])








from transcoder_circuits.circuit_analysis import make_sae_feature_vector, greedy_get_top_paths, print_all_paths, get_paths_via_filter, FeatureFilter, FeatureType, FilterType, paths_to_graph, add_error_nodes_to_graph, ComponentType
from transcoder_circuits.feature_dashboards import get_deembeddings_for_feature_vector, get_deembeddings_for_transcoder_feature, get_transcoder_pullback_features, get_all_feature_scores, get_feature_scores
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
from tqdm import tqdm






def find_top_k_features(model, transcoder, token_ids, token_index, top_k=10):
    score = get_all_feature_scores(model, transcoder, token_ids, batch_size=128)

    value, index = torch.topk(score, top_k, dim=-1)

    return value, index





model = HookedTransformer.from_pretrained('gpt2')












from datasets import load_dataset
from utils import tokenize_and_concatenate

dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10_000)
tokenized_owt = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128, streaming=True)
tokenized_owt = tokenized_owt.shuffle(42)
tokenized_owt = tokenized_owt.take(12800*2)
owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
owt_tokens_torch = torch.from_numpy(owt_tokens).cuda()





transcoder_template = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
for i in range(12):
    transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())


# prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded Emma"
prompt = "Oh, that rifle model is a 6M"
token_strs = model.to_str_tokens(prompt)

print("token_strs", list(enumerate(token_strs)))



token_ids = model.tokenizer(prompt, return_tensors='pt').input_ids

print("token_ids", token_ids)


features, indices = find_top_k_features(model, transcoders[8], token_ids, token_index=-1, top_k=10)

print("features.shape:", features.shape)
print("indices.shape:", indices.shape)




feature_vector = make_sae_feature_vector(transcoders[8], indices[-1][0], token=-1)
print(feature_vector)
print(feature_vector.vector.shape)

print("indices[idx_][0]:", indices[-1])
print("features[idx_][0]:", features[-1])


pulledback_feature, deembeddings = get_deembeddings_for_feature_vector(model, feature_vector, k=7)
deembeddings = list(deembeddings)



positive_deembeddings = [(d[0], d[1]) for d in deembeddings]
negative_deembeddings = [(d[2], d[3]) for d in deembeddings]

print("positive_deembeddings:", positive_deembeddings)
print("negative_deembeddings:", negative_deembeddings)

# print("pulledback_feature:", pulledback_feature)
# print("pulledback_feature.shape:", pulledback_feature.shape)



cur_scores = get_feature_scores(model, transcoders[8], owt_tokens_torch[:128*100], 89, batch_size=128, use_raw_scores=False)
display_activating_examples_dash(model, owt_tokens_torch, cur_scores, header_level=None) # don't show dashboard with html headers