from transcoder_circuits.circuit_analysis import make_sae_feature_vector, greedy_get_top_paths, print_all_paths, get_paths_via_filter, FeatureFilter, FeatureType, FilterType, paths_to_graph, add_error_nodes_to_graph, ComponentType
from transcoder_circuits.feature_dashboards import get_deembeddings_for_feature_vector, get_deembeddings_for_transcoder_feature, get_transcoder_pullback_features, get_all_feature_scores
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
from tqdm import tqdm






def fine_top_k_features(model, transcoder, token_ids, token_index, top_k=10):
    score = get_all_feature_scores(model, transcoder, token_ids, batch_size=128)[token_index, :]

    value, index = torch.topk(score, top_k)

    return value, index





model = HookedTransformer.from_pretrained('gpt2')


transcoder_template = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
for i in range(12):
    transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())


prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded Emma"
# prompt = "Oh, that rifle model is a 6M"
token_strs = model.to_str_tokens(prompt)

print("token_strs", list(enumerate(token_strs)))



token_ids = model.tokenizer(prompt, return_tensors='pt').input_ids

print("token_ids", token_ids)



top_k_features, top_k_features_idx = fine_top_k_features(model, transcoders[11], token_ids, token_index=-1, top_k=10)

print("top_k_features", top_k_features)
print("top_k_features_idx", top_k_features_idx)





feature_vector = make_sae_feature_vector(transcoders[11], top_k_features_idx[0])
print("feature_vector", feature_vector)
print("feature_vector.vector.shape", feature_vector.vector.shape)



_, cache = model.run_with_cache(prompt) # cache the model activations on this prompt


all_paths = greedy_get_top_paths(model, transcoders, cache, feature_vector, num_iters=3, num_branches=15, do_raw_attribution=True)

print_all_paths(all_paths)


print("======================")




# # ignore paths that go through MLP2 transcoder
# filtered_paths = get_paths_via_filter(all_paths, not_infix_path=[
#     FeatureFilter(
#         layer=2, layer_filter_type=FilterType.EQ,
#         feature_type=FeatureType.TRANSCODER
#     )
# ])

# ignore paths that end in last token
filtered_paths = get_paths_via_filter(all_paths, suffix_path=[
    FeatureFilter(token=len(token_strs)-1, token_filter_type=FilterType.NE)
])

# look at paths that end in layer 0
filtered_paths = get_paths_via_filter(filtered_paths, suffix_path=[
    FeatureFilter(layer=0)
])

# # ignore paths that end in last token
# filtered_paths = get_paths_via_filter(filtered_paths, suffix_path=[
#     FeatureFilter(component_type=ComponentType.MLP, component_type_filter_type=FilterType.EQ)
# ])

print_all_paths(filtered_paths)



print(filtered_paths[1][-1])
print(filtered_paths[1])
pulledback_feature, deembeddings = get_deembeddings_for_feature_vector(model, filtered_paths[1][-1], k=7)
deembeddings = list(deembeddings)



positive_deembeddings = [(d[0], d[1]) for d in deembeddings]
negative_deembeddings = [(d[2], d[3]) for d in deembeddings]

print("positive_deembeddings:", positive_deembeddings)
print("negative_deembeddings:", negative_deembeddings)



token_strs [(0, '<|endoftext|>'), (1, 'In'), (2, ' the'), (3, ' hotel'), (4, ' laundry'), (5, ' room'), (6, ','), (7, ' Emma'), (8, ' burned'), (9, ' Mary'), (10, "'s"), (11, ' shirt'), (12, ','), (13, ' so'), (14, ' the'), (15, ' manager'), (16, ' sc'), (17, 'olded'), (18, ' Emma')]




# live_features = np.arange(len(frequencies[8]))[utils.to_numpy(frequencies[8] > -4)]

# feature_idx = live_features[77]
# feature_vector = make_sae_feature_vector(transcoders[8], feature_idx)
# print(feature_vector)
# print(feature_vector.vector.shape)


# prompt = "Oh, that rifle model is a 6M"
# token_strs = model.to_str_tokens(prompt)

# print(list(enumerate(token_strs)))


# prompt = "Oh, that rifle model is a 6M"
# _, cache = model.run_with_cache(prompt) # cache the model activations on this prompt

# all_paths = greedy_get_top_paths(model, transcoders, cache, feature_vector, num_iters=3, num_branches=15, do_raw_attribution=True)

# print_all_paths(all_paths)





# # ignore paths that go through MLP2 transcoder
# filtered_paths = get_paths_via_filter(all_paths, not_infix_path=[
#     FeatureFilter(
#         layer=2, layer_filter_type=FilterType.EQ,
#         feature_type=FeatureType.TRANSCODER
#     )
# ])

# # ignore paths that end in last token
# filtered_paths = get_paths_via_filter(filtered_paths, suffix_path=[
#     FeatureFilter(token=9, token_filter_type=FilterType.NE)
# ])

# # look at paths that end in layer 0
# filtered_paths = get_paths_via_filter(filtered_paths, suffix_path=[
#     FeatureFilter(layer=0)
# ])

# print_all_paths(filtered_paths)

# edges, nodes = paths_to_graph(filtered_paths)


# for edge, contrib in edges.items():
#     print(edge, contrib)


# for node, node_feature_obj in nodes.items():
#     # each node is associated with a FeatureVector object
#     # and we can access the contribution of a FeatureVector by using its .contrib member
#     print(node, node_feature_obj.contrib)  


# edges_with_error, nodes_with_error = add_error_nodes_to_graph(model, cache, transcoders, edges, nodes)


# for node, node_feature_obj in nodes_with_error.items():
#     print(node, node_feature_obj.contrib) 







# print(all_paths[1][3][-1])
# print(all_paths[1][3])
# pulledback_feature, deembeddings = get_deembeddings_for_feature_vector(model, all_paths[1][3][-1], k=7)
# deembeddings = list(deembeddings)

# print(deembeddings)


# pulledback_feature, deembeddings = get_deembeddings_for_transcoder_feature(model, transcoders[0], 7829, attn_head=None, attn_layer=0, k=7)
# deembeddings = list(deembeddings)

# print(deembeddings)


# logits = list(get_transcoder_pullback_features(model, all_paths[0][4][0], transcoders[2], k=5,
#         input_tokens=None, input_example=None, input_token_idx=None)
#     )

# print(len(logits))
# print(logits[:5])