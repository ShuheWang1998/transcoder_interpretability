from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils




model = HookedTransformer.from_pretrained('gpt2')


transcoder_template = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
for i in range(11):
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



# prompt = "This is a test string!"

prompt = "When John and Mary went to the shops, John gave the bag to"
prompt_answer = ("Mary", "John")

# print("length of prompt:", len(prompt.split(" ")))

# print("transcoders[8].cfg:", transcoders[8].cfg)

tokens_arr = model.to_tokens(prompt)


    
# Only replace the first 8 transcoders
for transcoder in transcoders[:8]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)
    # print("=========================")
    # print("transcoder.cfg.hook_point_layer:", transcoder.cfg.hook_point_layer)
    

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


logits = model(tokens_arr)

for transcoder in transcoders[:8]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp.hidden_acts.register_hook(save_grad(transcoder.cfg.hook_point_layer))



print("tokens_arr:", tokens_arr)

print("logits:", logits)
print("logits shape:", logits.shape)

print("model.blocks[7].mlp:", model.blocks[7].mlp)

print("model.blocks[7].mlp.hidden_acts:", model.blocks[7].mlp.hidden_acts)
print("model.blocks[7].mlp.hidden_acts shape:", model.blocks[7].mlp.hidden_acts.shape)
    
feature_activations = model.blocks[7].mlp.hidden_acts[0, -1] # batch 0, last token
print("Top 20 features activated on the last token:", torch.topk(feature_activations, k=20)) # what are the top features activated on the last token?






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

logits[:, -1, 1].backward()

# print("model.blocks[7].mlp.hidden_acts.grad: ", model.blocks[7].mlp.hidden_acts.grad)

# print("grads: ", grads)

print("grads[7]: ", grads[7])

print("grads[7].shape: ", grads[7].shape)

print("top 20 feature gradients on the last token: ", torch.topk(grads[7][0, -1], k=20))
