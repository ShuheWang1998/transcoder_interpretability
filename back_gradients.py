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

change_layer = 0

change_layer_multiplier = 1


# prompt = "This is a test string!"

# prompt = "When John and Mary went to the shops, John gave the bag to"
# prompt_answer = (" Mary", " John")
# prompt_answer = (" Mary", " John")

prompt = "When John and Mary went to the shops, Mary gave the bag to"
prompt_answer = (" John", " Mary")


# prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded"
# prompt_answer = (" Mary", " Emma")





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

for transcoder in transcoders[:sub_transcoders_num]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp.hidden_acts.register_hook(save_grad(transcoder.cfg.hook_point_layer))



print("tokens_arr:", tokens_arr)

# print("logits:", logits)
print("logits shape:", logits.shape)

print("top 20 logits:", torch.topk(logits[0, -1], k=20))

print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

# print("model.blocks[0].mlp:", model.blocks[0].mlp)

# print("model.blocks[0].mlp.hidden_acts:", model.blocks[0].mlp.hidden_acts)
# print("model.blocks[0].mlp.hidden_acts shape:", model.blocks[0].mlp.hidden_acts.shape)
    
feature_activations = model.blocks[change_layer].mlp.hidden_acts[0, -1] # batch 0, last token
print(f"layer {change_layer} Top 20 features activated on the last token:", torch.topk(feature_activations, k=20)) # what are the top features activated on the last token?






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

print("logits[:, -1, answer_token_indices[0, 1]]: ", logits[:, -1, answer_token_indices[0, 1]])

logits[:, -1, answer_token_indices[0, 1]].backward()

# print("model.blocks[7].mlp.hidden_acts.grad: ", model.blocks[7].mlp.hidden_acts.grad)

# print("grads: ", grads)


# print(f"grads[{change_layer}]: ", grads[change_layer])

print(f"grads[{change_layer}].shape: ", grads[change_layer].shape)

# print("top 20 feature gradients on the last token: ", torch.topk(grads[0][0, -1], k=20))


print(f"model.blocks[{change_layer}].mlp.hidden_acts.shape: ", model.blocks[change_layer].mlp.hidden_acts.shape)

print(f"model.blocks[{change_layer}].mlp.out.shape: ", model.blocks[change_layer].mlp.out.shape)

model.blocks[change_layer].mlp.additional_gradients = grads[change_layer] * change_layer_multiplier

logits = model(tokens_arr)


print("layer top 20 features on the last token: ", torch.topk(model.blocks[change_layer].mlp.hidden_acts[0, -1], k=20))

# logits[:, -1, 1757].backward()

# print("grads[0]: ", grads[0])
# print("logits:", logits)
print("logits shape:", logits.shape)

print("top 20 logits:", torch.topk(logits[0, -1], k=20))

print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

print("correct logits: ", logits[:, -1, answer_token_indices[0, 0]])

print("incorrect logits: ", logits[:, -1, answer_token_indices[0, 1]])

print("logits diff: ", logits[:, -1, answer_token_indices[0, 0]] - logits[:, -1, answer_token_indices[0, 1]])


normalized_gradient = []

for i in range(sub_transcoders_num):
    normalized_gradient.append(torch.norm(grads[i], p=1))
    print(f"normalized_gradient[{i}]: ", normalized_gradient[i])









# layer 0 Top 20 features activated on the last token: torch.return_types.topk(
# values=tensor([7.1131e+00, 3.9266e+00, 2.1755e+00, 8.4439e-01, 7.3629e-01, 7.1201e-01,
#         6.5119e-01, 6.1520e-01, 4.5307e-01, 4.4939e-01, 3.4332e-01, 2.9460e-01,
#         2.0887e-01, 1.6692e-01, 1.3713e-01, 1.3547e-01, 4.3553e-02, 8.0219e-03,
#         5.9294e-03, 0.0000e+00], device='cuda:0', grad_fn=<TopkBackward0>),
# indices=tensor([  630,  7866, 19861,  5209, 23747, 24214, 11025, 10066, 20820, 11570,
#          1143, 12001,  5388, 12496, 24393,  8063, 14799,  3758,  3815,     0],
#        device='cuda:0'))



# values=tensor([7.3525, 4.3235, 2.4235, 1.4162, 1.3763, 1.2094, 0.9388, 0.8626, 0.7862,
#         0.6247, 0.6170, 0.4514, 0.4307, 0.4082, 0.4072, 0.3974, 0.3325, 0.3312,
#         0.2344, 0.2239], device='cuda:0', grad_fn=<TopkBackward0>),
# indices=tensor([  630,  7866, 19861, 24214, 20820, 12001, 23747, 24393, 10066,  5209,
#          8063, 11248,  4331,  2448, 11025, 14799, 23464, 11570, 15125, 20435],
#        device='cuda:0'))



# 630,  7866, 19861,  5209, 23747, 24214, 11025, 10066, 20820, 11570, 1143, 12001,  5388, 12496, 24393,  8063, 14799,  3758,  3815,     0


# 630,  7866, 19861, 24214, 20820, 12001, 23747, 24393, 10066,  5209, 8063, 11248,  4331,  2448, 11025, 14799, 23464, 11570, 15125, 20435