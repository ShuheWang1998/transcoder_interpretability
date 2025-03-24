from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import json



def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def load_model(model_name="gpt2"):
    model = HookedTransformer.from_pretrained(model_name)
    return model


def load_vocab(vocab_path):
    vocab = read_json(vocab_path)
    vocab = {v: k for k, v in vocab.items()}
    return vocab


def load_transcoders(transcoders_path, sub_transcoders_num):
    transcoder_template = transcoders_path
    transcoders = []
    for i in range(sub_transcoders_num):
        transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())

    return transcoders


def sub_transcoders_wrapper(model, transcoders, sub_transcoders_num):
    for transcoder in transcoders[:sub_transcoders_num]:
        model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)
    return model




if __name__ == "__main__":

    model_name = "gpt2"
    vocab_path = "/data/projects/punim2522/models/gpt2/vocab.json"
    transcoders_path = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    sub_transcoders_num = 12

    prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded Emma"
    prompt_answer = (" Mary", " Emma")




    model = load_model(model_name=model_name)
    vocab = load_vocab(vocab_path=vocab_path)
    transcoder = load_transcoders(transcoders_path=transcoders_path, sub_transcoders_num=sub_transcoders_num)


    model = sub_transcoders_wrapper(model=model, transcoders=transcoder, sub_transcoders_num=sub_transcoders_num)
    tokens_arr = model.to_tokens(prompt)


    print("tokens_arr:", tokens_arr)

    logits = model(tokens_arr)

    print("logits shape:", logits.shape)

    print("top 20 logits:", torch.topk(logits[0, -2], k=20))

    print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -2], k=20).indices])

    answer_token_indices = torch.tensor([[model.to_single_token(prompt_answer[0]), model.to_single_token(prompt_answer[1])]], device=model.cfg.device)


    print("answer_token_indices: ", answer_token_indices)