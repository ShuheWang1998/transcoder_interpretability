from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils




model = HookedTransformer.from_pretrained('gpt2')

# This function was stolen from one of Neel Nanda's exploratory notebooks
# Thanks, Neel!
import einops

def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming = False,
    max_length = 1024,
    column_name = "text",
    add_bos_token = True,
):
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it just outputs nothing. I'm not super sure why
    """
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)

    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples):
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[column_name],
    )
    #tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


from datasets import load_dataset
from huggingface_hub import HfApi

dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10_000)
tokenized_owt = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128, streaming=True)
tokenized_owt = tokenized_owt.shuffle(42)
tokenized_owt = tokenized_owt.take(12800*2)
owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])


owt_tokens_torch = torch.from_numpy(owt_tokens).cuda()


transcoder_template = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
frequencies = []
for i in range(11):
    transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())
    frequencies.append(torch.load(f"{transcoder_template.format(i)}_log_feature_sparsity.pt"))
    # print("===========")
    # print(type(torch.load(f"{transcoder_template.format(i)}_log_feature_sparsity.pt")))
    # print(torch.load(f"{transcoder_template.format(i)}_log_feature_sparsity.pt").size())


# Clean up memory
import gc

gc.collect()
torch.cuda.empty_cache()


# plt.hist(utils.to_numpy(frequencies[8]), bins=100)
# plt.xlabel("Log10 feature firing frequency")
# plt.ylabel("Number of features")
# plt.title("Transcoder 8 feature frequency")
# # plt.show()
# plt.savefig('./test.png')


live_features = np.arange(len(frequencies[8]))[utils.to_numpy(frequencies[8] > -4)]

feature_idx = live_features[0] # get zeroth live feature

# get feature activation scores
# scores = get_feature_scores(model, transcoders[8], owt_tokens_torch, feature_idx, batch_size=128)
# # what's happening in the above line of code?
# # owt_tokens_torch: our dataset of OpenWebText tokens
# # batch_size: how many inputs to process at once

# # display top activating examples
# display_activating_examples_dash(model, owt_tokens_torch, scores)


print(get_feature_scores(model, transcoders[8], model.tokenizer(
    "This was the defenseman's first championship win",
return_tensors='pt').input_ids, feature_idx))

print(get_feature_scores(model, transcoders[8], model.tokenizer(
    "This was the rookie's first goal of the season",
return_tensors='pt').input_ids, feature_idx))

print(get_feature_scores(model, transcoders[8], model.tokenizer(
    "After last night's game, the Cubs have finally gotten their first",
return_tensors='pt').input_ids, feature_idx))

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


logits, cache = model.run_with_cache("Hello, world!") # "model" is a HookedTransformer from TransformerLens
activations = cache[transcoders[8].cfg.hook_point] # transcoders[8].cfg.hook_point tells us where transcoders[8] gets its input from
feature_activations = transcoders[8](activations)[1]
feature_activations = feature_activations[0,-1] # batch 0, last token
print("Top 5 features activated on the last token:", torch.topk(feature_activations, k=5)) # what are the top features activated on the last token?
print("logits:", logits)
print("logits shape:", logits.shape)
print("cache:", cache)



# prompt = "After last night's game, the Cubs have finally gotten their first"

# scores = get_feature_scores(model, transcoders[8], model.tokenizer(prompt, return_tensors='pt').input_ids, live_features[0])
# print("Original feature activations:")
# print(f"\t{scores}")
# with TranscoderReplacementContext(model, transcoders[1:8]):
#     scores = get_feature_scores(model, transcoders[8], model.tokenizer(prompt, return_tensors='pt').input_ids, live_features[0])
# print("Replace all MLP layers after MLP0 until MLP8 with transcoders:")
# print(f"\t{scores}")

# with TranscoderReplacementContext(model, transcoders[:8]):
#     scores = get_feature_scores(model, transcoders[8], model.tokenizer(prompt, return_tensors='pt').input_ids, live_features[0])
# print("Replace all MLP layers up to MLP8 with transcoders:")
# print(f"\t{scores}")

prompt = "This is a test string!"

print("transcoders[8].cfg:", transcoders[8].cfg)

act_name = transcoders[8].cfg.hook_point
act_name_1 = transcoders[8].cfg.out_hook_point
layer = transcoders[8].cfg.hook_point_layer
# tokens_arr = model.tokenizer(prompt, return_tensors='pt').input_ids
tokens_arr = model.to_tokens(prompt)

# with TranscoderReplacementContext(model, transcoders[:8]):
#     # logits, cache = model.run_with_cache(tokens_arr, stop_at_layer=layer+1, names_filter=[
# 	# 			act_name, act_name_1
# 	# 		])


#     logits = model(tokens_arr)

#     print("logits:", logits)
#     print("logits shape:", logits.shape)

#     print("model.blocks[8].mlp:", model.blocks[8].mlp)

#     print("model.blocks[8].mlp.hidden_acts:", model.blocks[8].mlp.hidden_acts)
#     print("model.blocks[8].mlp.hidden_acts shape:", model.blocks[8].mlp.hidden_acts.shape)


    # mlp_acts = cache[act_name]

    # print("mlp_acts:", mlp_acts)
    # print("mlp_acts shape:", mlp_acts.shape)

    # print("logits:", logits)
    # print("logits shape:", logits.shape)

    # print("cache:", cache)

    # mlp_acts_flattened = mlp_acts.reshape(-1, transcoders[8].W_enc.shape[0])

    # print("mlp_acts_flattened:", mlp_acts_flattened)
    # print("mlp_acts_flattened shape:", mlp_acts_flattened.shape)

    # # print(transcoders[8](mlp_acts_flattened))
    # _, hidden_acts, _, _, _, _ = transcoders[8](mlp_acts_flattened)

    # print("hidden_acts:", hidden_acts)
    # print("hidden_acts shape:", hidden_acts.shape)

    # print("transcoders[8]:", transcoders[8])

    # new_acts = cache[act_name_1]

    # print("new_acts:", new_acts)
    # print("new_acts shape:", new_acts.shape)

    # print("transcoders[8].W_enc.shape:", transcoders[8].W_enc.shape)

    # print("transcoders[8].W_dec.shape:", transcoders[8].W_dec.shape)
    
    
    
for transcoder in transcoders[:8]:
    model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)
    print("=========================")
    print("transcoder.cfg.hook_point_layer:", transcoder.cfg.hook_point_layer)
    
logits = model(tokens_arr)

print("tokens_arr:", tokens_arr)

print("logits:", logits)
print("logits shape:", logits.shape)

print("model.blocks[7].mlp:", model.blocks[7].mlp)

print("model.blocks[7].mlp.hidden_acts:", model.blocks[7].mlp.hidden_acts)
print("model.blocks[7].mlp.hidden_acts shape:", model.blocks[7].mlp.hidden_acts.shape)
    
    
    
    
#     scores = get_feature_scores(model, transcoders[8], model.tokenizer(prompt, return_tensors='pt').input_ids, live_features[0])
# print("Replace all MLP layers up to MLP8 with transcoders:")
# print(f"\t{scores}")