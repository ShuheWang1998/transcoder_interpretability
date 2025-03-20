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


prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded"
prompt_answer = (" Emma", " Mary")





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


view_layer = 3

print("model.blocks[view_layer].residual shape: ", model.blocks[view_layer].residual.shape)

mid_layer_token = model.unembed(model.ln_final(model.blocks[view_layer].residual))

print("mid_layer_token shape: ", mid_layer_token.shape)

print("top 20 mid_layer_token: ", torch.topk(mid_layer_token[0, -1], k=20))

print("top 20 words: ", [vocab[int(i)] for i in torch.topk(mid_layer_token[0, -1], k=20).indices])

print("mid_layer_token[:, -1, answer_token_indices[0, 0]]: ", mid_layer_token[:, -1, answer_token_indices[0, 0]])

print("mid_layer_token[:, -1, answer_token_indices[0, 1]]: ", mid_layer_token[:, -1, answer_token_indices[0, 1]])



# layer 1
# values=tensor([13.6632, 13.2253, 12.7099, 12.4926, 12.2588, 12.0741, 12.0574, 11.8105,
#         11.7825, 11.7482, 11.6168, 11.5019, 11.4875, 11.3864, 11.3534, 11.2308,
#         11.2061, 11.1259, 11.0051, 10.9555], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([  683,    11,   416,   465, 21948,   262, 46838,   422, 40618,   607,
#           284,   290,   329, 34313,   534,  1028,   326,   351,   340,   379],
#        device='cuda:0'))
# top 20 words:  ['Ġhim', ',', 'Ġby', 'Ġhis', 'ĠRarity', 'Ġthe', 'ĠCartoon', 'Ġfrom', 'Ġmerciless', 'Ġher', 'Ġto', 'Ġand', 'Ġfor', 'Ġpolitely', 'Ġyour', 'Ġagainst', 'Ġthat', 'Ġwith', 'Ġit', 'Ġat']
# mid_layer_token[:, -1, answer_token_indices[0, 0]]:  tensor([4.1657], device='cuda:0', grad_fn=<SelectBackward0>)
# mid_layer_token[:, -1, answer_token_indices[0, 1]]:  tensor([5.6849], device='cuda:0', grad_fn=<SelectBackward0>)



# layer 2
# values=tensor([15.0466, 14.8317, 14.8309, 13.6953, 13.3739, 13.1521, 13.1280, 12.7836,
#         12.7302, 12.6896, 12.6270, 12.5902, 12.4140, 12.2320, 12.2137, 12.1976,
#         12.1174, 11.9687, 11.9514, 11.9022], device='cuda:0',
#        grad_fn=<TopkBackward0>),
# indices=tensor([  683,   465,    11,   416,   607,   422,   262,   326,   284,   329,
#           351,   290, 34425,   625,   287, 34313,  2241,    13, 48959,  1028],
#        device='cuda:0'))
# top 20 words:  ['Ġhim', 'Ġhis', ',', 'Ġby', 'Ġher', 'Ġfrom', 'Ġthe', 'Ġthat', 'Ġto', 'Ġfor', 'Ġwith', 'Ġand', 'Ġangrily', 'Ġover', 'Ġin', 'Ġpolitely', 'Ġhimself', '.', 'olded', 'Ġagainst']
# mid_layer_token[:, -1, answer_token_indices[0, 0]]:  tensor([5.5611], device='cuda:0', grad_fn=<SelectBackward0>)
# mid_layer_token[:, -1, answer_token_indices[0, 1]]:  tensor([6.4851], device='cuda:0', grad_fn=<SelectBackward0>)