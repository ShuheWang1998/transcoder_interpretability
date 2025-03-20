from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import numpy as np




model = HookedTransformer.from_pretrained('gpt2')

sub_transcoders_num = 12
transcoder_template = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
for i in range(sub_transcoders_num):
    transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())


# Clean up memory
import gc

gc.collect()
torch.cuda.empty_cache()



# prompt = "This is a test string!"

prompt = "When John and Mary went to the shops, John gave the bag to"
prompt_answer = ("Mary", "John")

tokens_arr = model.to_tokens(prompt)

print("tokens_arr:", tokens_arr)

logits_before_replacement = model(tokens_arr)

print("logits_before_replacement shape:", logits_before_replacement.shape)


    
for transcoder in transcoders[:sub_transcoders_num]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)
    # print("=========================")
    # print("transcoder.cfg.hook_point_layer:", transcoder.cfg.hook_point_layer)
    

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


logits_after_replacement = model(tokens_arr)

for transcoder in transcoders[:sub_transcoders_num]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp.hidden_acts.register_hook(save_grad(transcoder.cfg.hook_point_layer))





print("logits_after_replacement shape:", logits_after_replacement.shape)


print("top 20 logits after replacement:", torch.topk(logits_after_replacement[0, -1], k=20))


print("top 20 logits before replacement:", torch.topk(logits_before_replacement[0, -1], k=20))


max_difference = (logits_after_replacement[0, -1] - logits_before_replacement[0, -1]).max()

print("max difference (after - before):", max_difference)

min_difference = (logits_after_replacement[0, -1] - logits_before_replacement[0, -1]).min()

print("min difference (after - before):", min_difference)

different_numbers = (logits_after_replacement[0, -1] != logits_before_replacement[0, -1]).sum()

print("different numbers (total):", different_numbers)


for top_k in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]:
    top_k_value_after_replacement, top_k_index_after_replacement = torch.topk(logits_after_replacement[0, -1], k=top_k)
    top_k_value_before_replacement, top_k_index_before_replacement = torch.topk(logits_before_replacement[0, -1], k=top_k)

    different_indices = (top_k_index_after_replacement != top_k_index_before_replacement).sum()

    print("top", top_k, "different indices:", different_indices)

    intersection = np.intersect1d(top_k_index_after_replacement.cpu().numpy(), top_k_index_before_replacement.cpu().numpy())

    # print("intersection:", intersection)

    print("top", top_k, "intersection shape:", intersection.shape)

    # print("intersection value:", intersection)



# layer 1
# max difference (after - before): tensor(0.5034, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-0.5642, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(0, device='cuda:0')
# top 2 intersection shape: (2,)
# top 3 different indices: tensor(0, device='cuda:0')
# top 3 intersection shape: (3,)
# top 5 different indices: tensor(0, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(2, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(8, device='cuda:0')
# top 20 intersection shape: (20,)
# top 50 different indices: tensor(32, device='cuda:0')
# top 50 intersection shape: (47,)
# top 100 different indices: tensor(81, device='cuda:0')
# top 100 intersection shape: (96,)
# top 200 different indices: tensor(181, device='cuda:0')
# top 200 intersection shape: (192,)
# top 500 different indices: tensor(477, device='cuda:0')
# top 500 intersection shape: (486,)
# top 1000 different indices: tensor(973, device='cuda:0')
# top 1000 intersection shape: (969,)


# layer 2
# max difference (after - before): tensor(0.6183, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-0.7542, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(0, device='cuda:0')
# top 2 intersection shape: (2,)
# top 3 different indices: tensor(0, device='cuda:0')
# top 3 intersection shape: (3,)
# top 5 different indices: tensor(0, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(4, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(12, device='cuda:0')
# top 20 intersection shape: (18,)
# top 50 different indices: tensor(37, device='cuda:0')
# top 50 intersection shape: (47,)
# top 100 different indices: tensor(86, device='cuda:0')
# top 100 intersection shape: (95,)
# top 200 different indices: tensor(185, device='cuda:0')
# top 200 intersection shape: (189,)
# top 500 different indices: tensor(476, device='cuda:0')
# top 500 intersection shape: (478,)
# top 1000 different indices: tensor(975, device='cuda:0')
# top 1000 intersection shape: (965,)

# layer 3
# max difference (after - before): tensor(1.0174, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-0.9917, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(0, device='cuda:0')
# top 2 intersection shape: (2,)
# top 3 different indices: tensor(0, device='cuda:0')
# top 3 intersection shape: (3,)
# top 5 different indices: tensor(0, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(2, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(11, device='cuda:0')
# top 20 intersection shape: (18,)
# top 50 different indices: tensor(39, device='cuda:0')
# top 50 intersection shape: (46,)
# top 100 different indices: tensor(88, device='cuda:0')
# top 100 intersection shape: (94,)
# top 200 different indices: tensor(186, device='cuda:0')
# top 200 intersection shape: (185,)
# top 500 different indices: tensor(485, device='cuda:0')
# top 500 intersection shape: (477,)
# top 1000 different indices: tensor(982, device='cuda:0')
# top 1000 intersection shape: (950,)


# layer 4
# max difference (after - before): tensor(1.3069, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-1.1925, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(2, device='cuda:0')
# top 3 intersection shape: (3,)
# top 5 different indices: tensor(2, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(5, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(11, device='cuda:0')
# top 20 intersection shape: (18,)
# top 50 different indices: tensor(40, device='cuda:0')
# top 50 intersection shape: (46,)
# top 100 different indices: tensor(90, device='cuda:0')
# top 100 intersection shape: (89,)
# top 200 different indices: tensor(190, device='cuda:0')
# top 200 intersection shape: (176,)
# top 500 different indices: tensor(489, device='cuda:0')
# top 500 intersection shape: (460,)
# top 1000 different indices: tensor(987, device='cuda:0')
# top 1000 intersection shape: (934,)


# layer 5
# max difference (after - before): tensor(1.6049, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-1.4414, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(0, device='cuda:0')
# top 2 intersection shape: (2,)
# top 3 different indices: tensor(0, device='cuda:0')
# top 3 intersection shape: (3,)
# top 5 different indices: tensor(2, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(4, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(10, device='cuda:0')
# top 20 intersection shape: (18,)
# top 50 different indices: tensor(40, device='cuda:0')
# top 50 intersection shape: (45,)
# top 100 different indices: tensor(90, device='cuda:0')
# top 100 intersection shape: (86,)
# top 200 different indices: tensor(189, device='cuda:0')
# top 200 intersection shape: (176,)
# top 500 different indices: tensor(487, device='cuda:0')
# top 500 intersection shape: (457,)
# top 1000 different indices: tensor(986, device='cuda:0')
# top 1000 intersection shape: (927,)


# layer 6
# max difference (after - before): tensor(2.4885, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-1.8756, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(0, device='cuda:0')
# top 2 intersection shape: (2,)
# top 3 different indices: tensor(1, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(3, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(6, device='cuda:0')
# top 10 intersection shape: (9,)
# top 20 different indices: tensor(15, device='cuda:0')
# top 20 intersection shape: (19,)
# top 50 different indices: tensor(44, device='cuda:0')
# top 50 intersection shape: (42,)
# top 100 different indices: tensor(91, device='cuda:0')
# top 100 intersection shape: (84,)
# top 200 different indices: tensor(189, device='cuda:0')
# top 200 intersection shape: (164,)
# top 500 different indices: tensor(486, device='cuda:0')
# top 500 intersection shape: (431,)
# top 1000 different indices: tensor(985, device='cuda:0')
# top 1000 intersection shape: (874,)


# layer 7
# max difference (after - before): tensor(2.6286, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-2.2458, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(1, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(3, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(8, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(15, device='cuda:0')
# top 20 intersection shape: (18,)
# top 50 different indices: tensor(45, device='cuda:0')
# top 50 intersection shape: (41,)
# top 100 different indices: tensor(95, device='cuda:0')
# top 100 intersection shape: (79,)
# top 200 different indices: tensor(194, device='cuda:0')
# top 200 intersection shape: (156,)
# top 500 different indices: tensor(494, device='cuda:0')
# top 500 intersection shape: (423,)
# top 1000 different indices: tensor(992, device='cuda:0')
# top 1000 intersection shape: (850,)


# layer 8
# max difference (after - before): tensor(3.7874, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-2.9069, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(2, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(4, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(8, device='cuda:0')
# top 10 intersection shape: (9,)
# top 20 different indices: tensor(18, device='cuda:0')
# top 20 intersection shape: (16,)
# top 50 different indices: tensor(48, device='cuda:0')
# top 50 intersection shape: (36,)
# top 100 different indices: tensor(98, device='cuda:0')
# top 100 intersection shape: (71,)
# top 200 different indices: tensor(198, device='cuda:0')
# top 200 intersection shape: (143,)
# top 500 different indices: tensor(497, device='cuda:0')
# top 500 intersection shape: (393,)
# top 1000 different indices: tensor(997, device='cuda:0')
# top 1000 intersection shape: (798,)


# layer 9
# max difference (after - before): tensor(4.3799, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-3.5874, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(2, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(3, device='cuda:0')
# top 5 intersection shape: (5,)
# top 10 different indices: tensor(7, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(17, device='cuda:0')
# top 20 intersection shape: (15,)
# top 50 different indices: tensor(47, device='cuda:0')
# top 50 intersection shape: (37,)
# top 100 different indices: tensor(97, device='cuda:0')
# top 100 intersection shape: (61,)
# top 200 different indices: tensor(197, device='cuda:0')
# top 200 intersection shape: (131,)
# top 500 different indices: tensor(496, device='cuda:0')
# top 500 intersection shape: (347,)
# top 1000 different indices: tensor(995, device='cuda:0')
# top 1000 intersection shape: (732,)


# layer 10
# max difference (after - before): tensor(4.7783, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-4.1207, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(2, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(4, device='cuda:0')
# top 5 intersection shape: (4,)
# top 10 different indices: tensor(8, device='cuda:0')
# top 10 intersection shape: (9,)
# top 20 different indices: tensor(17, device='cuda:0')
# top 20 intersection shape: (14,)
# top 50 different indices: tensor(47, device='cuda:0')
# top 50 intersection shape: (31,)
# top 100 different indices: tensor(97, device='cuda:0')
# top 100 intersection shape: (58,)
# top 200 different indices: tensor(197, device='cuda:0')
# top 200 intersection shape: (116,)
# top 500 different indices: tensor(497, device='cuda:0')
# top 500 intersection shape: (304,)
# top 1000 different indices: tensor(995, device='cuda:0')
# top 1000 intersection shape: (660,)


# layer 11
# max difference (after - before): tensor(4.7272, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-4.6841, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(2, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(4, device='cuda:0')
# top 5 intersection shape: (4,)
# top 10 different indices: tensor(8, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(18, device='cuda:0')
# top 20 intersection shape: (13,)
# top 50 different indices: tensor(47, device='cuda:0')
# top 50 intersection shape: (30,)
# top 100 different indices: tensor(97, device='cuda:0')
# top 100 intersection shape: (52,)
# top 200 different indices: tensor(196, device='cuda:0')
# top 200 intersection shape: (116,)
# top 500 different indices: tensor(496, device='cuda:0')
# top 500 intersection shape: (297,)
# top 1000 different indices: tensor(996, device='cuda:0')
# top 1000 intersection shape: (646,)


# layer 12
# max difference (after - before): tensor(4.7272, device='cuda:0', grad_fn=<MaxBackward1>)
# min difference (after - before): tensor(-4.6841, device='cuda:0', grad_fn=<MinBackward1>)
# different numbers (total): tensor(50257, device='cuda:0')
# top 1 different indices: tensor(0, device='cuda:0')
# top 1 intersection shape: (1,)
# top 2 different indices: tensor(1, device='cuda:0')
# top 2 intersection shape: (1,)
# top 3 different indices: tensor(2, device='cuda:0')
# top 3 intersection shape: (2,)
# top 5 different indices: tensor(4, device='cuda:0')
# top 5 intersection shape: (4,)
# top 10 different indices: tensor(8, device='cuda:0')
# top 10 intersection shape: (10,)
# top 20 different indices: tensor(18, device='cuda:0')
# top 20 intersection shape: (13,)
# top 50 different indices: tensor(47, device='cuda:0')
# top 50 intersection shape: (30,)
# top 100 different indices: tensor(97, device='cuda:0')
# top 100 intersection shape: (52,)
# top 200 different indices: tensor(196, device='cuda:0')
# top 200 intersection shape: (116,)
# top 500 different indices: tensor(496, device='cuda:0')
# top 500 intersection shape: (297,)
# top 1000 different indices: tensor(996, device='cuda:0')
# top 1000 intersection shape: (646,)