import torch
from back_gradients import *
import json
from tqdm import tqdm








def read_jsonl(data_path):
    print(f"Reading {data_path}...")
    results = []
    file = open(data_path, 'r')
    for line in tqdm(file):
        results.append(json.loads(line))
    file.close()

    return results


def write_jsonl(data, data_path):
    print(f"Writing {data_path}...")
    with open(data_path, 'w') as f:
        for item in tqdm(data):
            f.write(json.dumps(item) + '\n')


def compute_kl_div(previous_vector, now_vector):
    previous_vector = torch.nn.functional.log_softmax(previous_vector, dim=-1)
    now_vector = torch.nn.functional.softmax(now_vector, dim=-1)

    kl_div = torch.nn.functional.kl_div(previous_vector, now_vector, log_target=False, reduction='batchmean')
    return kl_div


def change_ioi_format(data):
    results = []
    for idx_, item in tqdm(enumerate(data)):
        text = item['text']
        prompt = " ".join(text.strip().split(' ')[:-1])
        answer = text.strip().split(' ')[-1]
        results.append({
            'prompt': prompt,
            'correct_label': " " + answer,
            "model_predict": "",
            "id": idx_
        })
    return results


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











if __name__ == "__main__":
    # data = read_jsonl(data_path="/home/shuhewang/Easy-Transformer/ioi_prompts.jsonl")
    # data = change_ioi_format(data)
    # write_jsonl(data, "/home/shuhewang/transcoder_circuits/ioi_data_format.jsonl")

    a = torch.randn(10, 10)
    b = torch.randn(10, 10)

    kl_ = compute_kl_div(previous_vector=a, now_vector=b)
    print(kl_.shape)
    print(kl_)
