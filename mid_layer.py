from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import json



def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)



model = HookedTransformer.from_pretrained('gpt2')

sub_transcoders_num = 12

# change_layer = 11

# change_layer_multiplier = 10000


# prompt = "This is a test string!"

# prompt = "When John and Mary went to the shops, John gave the bag to"
# prompt_answer = (" Mary", " John")
# prompt_answer = (" Mary", " John")

# prompt = "When John and Mary went to the shops, Mary gave the bag to"
# prompt_answer = (" John", " Mary")


# prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded"
# prompt_answer = (" Emma", " Mary")

prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded"
prompt_answer = (" Mary", " Emma")





vocab_path = "/data/projects/punim2522/models/gpt2/vocab.json"
vocab = read_json(vocab_path)
vocab = {v: k for k, v in vocab.items()}



transcoder_template = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
for i in range(sub_transcoders_num):
    transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())


# Clean up memory
import gc

gc.collect()
torch.cuda.empty_cache()





# print(get_feature_scores(model, transcoders[8], model.tokenizer(
#     "This is a Test String!",
# return_tensors='pt').input_ids, feature_idx))

# results = []
# for idx_ in range(768):
#     results.append((get_feature_scores(model, transcoders[8], model.tokenizer(
#         "This is a Test String!",
#     return_tensors='pt').input_ids[-3:], idx_).max(), idx_))


# results = sorted(results, key=lambda x: x[0], reverse=True)

# print(results)


# logits, cache = model.run_with_cache("This is a test string!") # "model" is a HookedTransformer from TransformerLens
# activations = cache[transcoders[7].cfg.hook_point] # transcoders[8].cfg.hook_point tells us where transcoders[8] gets its input from
# feature_activations = transcoders[7](activations)[1]
# feature_activations = feature_activations[0, -1] # batch 0, last token
# print("Top 20 features activated on the last token:", torch.topk(feature_activations, k=20)) # what are the top features activated on the last token?
# print("logits:", logits)
# print("logits shape:", logits.shape)
# # print("cache:", cache)



print("=========================")




# print("length of prompt:", len(prompt.split(" ")))

# print("transcoders[8].cfg:", transcoders[8].cfg)

tokens_arr = model.to_tokens(prompt)


    
for transcoder in transcoders[:sub_transcoders_num]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)
    # print("=========================")
    # print("transcoder.cfg.hook_point_layer:", transcoder.cfg.hook_point_layer)
    

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


logits = model(tokens_arr)

# for transcoder in transcoders[:sub_transcoders_num]:
#     model.blocks[transcoder.cfg.hook_point_layer].mlp.hidden_acts.register_hook(save_grad(transcoder.cfg.hook_point_layer))



print("tokens_arr:", tokens_arr)

# print("logits:", logits)
print("logits shape:", logits.shape)

print("top 20 logits:", torch.topk(logits[0, -1], k=20))

print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

# print("model.blocks[0].mlp:", model.blocks[0].mlp)

# print("model.blocks[0].mlp.hidden_acts:", model.blocks[0].mlp.hidden_acts)
# print("model.blocks[0].mlp.hidden_acts shape:", model.blocks[0].mlp.hidden_acts.shape)
    
# feature_activations = model.blocks[change_layer].mlp.hidden_acts[0, -1] # batch 0, last token
# print(f"layer {change_layer} Top 20 features activated on the last token:", torch.topk(feature_activations, k=20)) # what are the top features activated on the last token?








def get_logits_diff(logits, answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]

    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))

    print("correct_logits: ", correct_logits)

    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))

    print("incorrect_logits: ", incorrect_logits)

    print("logits diff: ", correct_logits - incorrect_logits)

    # return (correct_logits - incorrect_logits).mean()
    return correct_logits - incorrect_logits


answer_token_indices = torch.tensor([[model.to_single_token(prompt_answer[0]), model.to_single_token(prompt_answer[1])]], device=model.cfg.device)

print("answer_token_indices: ", answer_token_indices)

# logits_diff = get_logits_diff(logits, answer_token_indices)

# print("logits_diff: ", logits_diff)
# print("logits_diff shape: ", logits_diff.shape)

# logits_diff.backward()

print("logits[:, -1, answer_token_indices[0, 0]]: ", logits[:, -1, answer_token_indices[0, 0]])

print("logits[:, -1, answer_token_indices[0, 1]]: ", logits[:, -1, answer_token_indices[0, 1]])

# logits[:, -1, answer_token_indices[0, 1]].backward()

# # print("model.blocks[7].mlp.hidden_acts.grad: ", model.blocks[7].mlp.hidden_acts.grad)

# # print("grads: ", grads)


# # print(f"grads[{change_layer}]: ", grads[change_layer])

# print(f"grads[{change_layer}].shape: ", grads[change_layer].shape)

# # print("top 20 feature gradients on the last token: ", torch.topk(grads[0][0, -1], k=20))


# print(f"model.blocks[{change_layer}].mlp.hidden_acts.shape: ", model.blocks[change_layer].mlp.hidden_acts.shape)

# print(f"model.blocks[{change_layer}].mlp.out.shape: ", model.blocks[change_layer].mlp.out.shape)

# model.blocks[change_layer].mlp.additional_gradients = grads[change_layer] * change_layer_multiplier

# logits = model(tokens_arr)

# # logits[:, -1, 1757].backward()

# # print("grads[0]: ", grads[0])
# # print("logits:", logits)
# print("logits shape:", logits.shape)

# print("top 20 logits:", torch.topk(logits[0, -1], k=20))

# print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

# print("correct logits: ", logits[:, -1, answer_token_indices[0, 0]])

# print("incorrect logits: ", logits[:, -1, answer_token_indices[0, 1]])

# print("logits diff: ", logits[:, -1, answer_token_indices[0, 0]] - logits[:, -1, answer_token_indices[0, 1]])

# print("model: ", model)

# print("model.blocks[11].hook_attn_in: ", model.blocks[11].hook_attn_in)

# print("model.cfg: ", model.cfg)

# print("model.residual: ", model.residual)


view_layer = 11

print("model.blocks[view_layer].residual shape: ", model.blocks[view_layer].residual.shape)

mid_layer_token = model.unembed(model.ln_final(model.blocks[view_layer].residual))

print("mid_layer_token shape: ", mid_layer_token.shape)

print("top 20 mid_layer_token: ", torch.topk(mid_layer_token[0, -1], k=20))

print("top 20 words: ", [vocab[int(i)] for i in torch.topk(mid_layer_token[0, -1], k=20).indices])

print("mid_layer_token[:, -1, answer_token_indices[0, 0]]: ", mid_layer_token[:, -1, answer_token_indices[0, 0]])

print("mid_layer_token[:, -1, answer_token_indices[0, 1]]: ", mid_layer_token[:, -1, answer_token_indices[0, 1]])


# layer 11
# values=tensor([15.5266, 13.3825, 13.1565, 12.8904, 12.8815, 12.8694, 12.6025, 12.4857,
#         12.2756, 12.0788, 11.9020, 11.8905, 11.7196, 11.6295, 11.3702, 11.3282,
#         11.1670, 11.1551, 10.7514, 10.6902], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([  607,   683,   262,    11,    13,  5335, 18966,   502,   290,   606,
#           326,   673,   340,   514,   284,   345,   465,   257,   329,  9074],
#        device='cuda:0'))
# top 20 words:  ['Ġher', 'Ġhim', 'Ġthe', ',', '.', 'ĠMary', 'ĠEmma', 'Ġme', 'Ġand', 'Ġthem', 'Ġthat', 'Ġshe', 'Ġit', 'Ġus', 'Ġto', 'Ġyou', 'Ġhis', 'Ġa', 'Ġfor', 'ĠMrs']
# mid_layer_token[:, -1, answer_token_indices[0, 0]]:  tensor([12.8694], device='cuda:0', grad_fn=<SelectBackward0>)
# mid_layer_token[:, -1, answer_token_indices[0, 1]]:  tensor([12.6025], device='cuda:0', grad_fn=<SelectBackward0>)

# layer 10
# values=tensor([21.4720, 20.4978, 18.6292, 18.3504, 17.6629, 16.7362, 16.4214, 16.2560,
#         16.1740, 16.1373, 15.7858, 15.7215, 15.4707, 15.3962, 15.3255, 14.8414,
#         14.3543, 14.2681, 14.1767, 14.0806], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([  683,   607,   502,   606,   514,  5335,   262,    11,    13, 18966,
#           465,   340,   326,   345,   290,  5223,   673,   284,  5850,  9074],
#        device='cuda:0'))
# top 20 words:  ['Ġhim', 'Ġher', 'Ġme', 'Ġthem', 'Ġus', 'ĠMary', 'Ġthe', ',', '.', 'ĠEmma', 'Ġhis', 'Ġit', 'Ġthat', 'Ġyou', 'Ġand', 'Ġherself', 'Ġshe', 'Ġto', 'ĠHarry', 'ĠMrs']
# mid_layer_token[:, -1, answer_token_indices[0, 0]]:  tensor([16.7362], device='cuda:0', grad_fn=<SelectBackward0>)
# mid_layer_token[:, -1, answer_token_indices[0, 1]]:  tensor([16.1373], device='cuda:0', grad_fn=<SelectBackward0>)

# layer 9
# values=tensor([25.5988, 22.1138, 21.7901, 21.3687, 20.5372, 18.0081, 17.7646, 17.5504,
#         17.3934, 17.1667, 16.8742, 16.5807, 16.5184, 16.1165, 16.0726, 15.8565,
#         15.5577, 15.4451, 14.9794, 14.9603], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([  683,   502,   606,   607,   514,   465,   345,   340,   262,    13,
#            11,   326,  2241,  5223,   290,  5335,  2506,  5850,   616, 18966],
#        device='cuda:0'))
# top 20 words:  ['Ġhim', 'Ġme', 'Ġthem', 'Ġher', 'Ġus', 'Ġhis', 'Ġyou', 'Ġit', 'Ġthe', '.', ',', 'Ġthat', 'Ġhimself', 'Ġherself', 'Ġand', 'ĠMary', 'Ġeveryone', 'ĠHarry', 'Ġmy', 'ĠEmma']
# mid_layer_token[:, -1, answer_token_indices[0, 0]]:  tensor([15.8565], device='cuda:0', grad_fn=<SelectBackward0>)
# mid_layer_token[:, -1, answer_token_indices[0, 1]]:  tensor([14.9603], device='cuda:0', grad_fn=<SelectBackward0>)

# layer 8
# values=tensor([24.6307, 20.4069, 19.6077, 18.6731, 18.0037, 17.1616, 17.0327, 16.8545,
#         16.7324, 16.5268, 16.1578, 16.1118, 15.5392, 15.1575, 15.0140, 14.9165,
#         14.8737, 14.7990, 14.6091, 14.4826], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([ 683,  606,  607,  502,  465,  514,  340,  326,  262,   11,  345,   13,
#          290, 2506,  284, 2241, 8453,  503,  616, 5850], device='cuda:0'))
# top 20 words:  ['Ġhim', 'Ġthem', 'Ġher', 'Ġme', 'Ġhis', 'Ġus', 'Ġit', 'Ġthat', 'Ġthe', ',', 'Ġyou', '.', 'Ġand', 'Ġeveryone', 'Ġto', 'Ġhimself', 'Ġapolog', 'Ġout', 'Ġmy', 'ĠHarry']
# mid_layer_token[:, -1, answer_token_indices[0, 0]]:  tensor([12.7765], device='cuda:0', grad_fn=<SelectBackward0>)
# mid_layer_token[:, -1, answer_token_indices[0, 1]]:  tensor([11.3373], device='cuda:0', grad_fn=<SelectBackward0>)