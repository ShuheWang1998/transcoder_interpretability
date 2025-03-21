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


def sub_transcoders_wrapper(model, transcoders, sub_transcoders_num, save_grads=False):
    for transcoder in transcoders[:sub_transcoders_num]:
        model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)

    if not save_grads:
        return model, None

    # Clean up memory
    import gc

    gc.collect()
    torch.cuda.empty_cache()


    grads = {}
    
    return model, grads


def save_gradient(model, transcoders, grads):
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook
    
    for transcoder in transcoders[:sub_transcoders_num]:
        model.blocks[transcoder.cfg.hook_point_layer].mlp.hidden_acts.register_hook(save_grad(transcoder.cfg.hook_point_layer))
    
    return model, grads


def get_logits_diff(logits, correct_token_index, incorrect_token_index):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]

    correct_logits = logits.gather(1, correct_token_index.unsqueeze(1))

    print("correct_logits: ", correct_logits)

    incorrect_logits = logits.gather(1, incorrect_token_index.unsqueeze(1))

    print("incorrect_logits: ", incorrect_logits)

    print("logits diff: ", correct_logits - incorrect_logits)

    # return (correct_logits - incorrect_logits).mean()
    return correct_logits - incorrect_logits


def normalize_gradient(model, grads):
    normalized_gradient = []

    for i in range(sub_transcoders_num):
        normalized_gradient.append(torch.norm(grads[i], p=1))
        print(f"normalized_gradient[{i}]: ", normalized_gradient[i] / torch.norm(model.blocks[i].mlp.hidden_acts, p=1))

    
    return normalized_gradient


def apply_gradient(model, grads, change_layers, change_layer_multipliers):
    for change_layer, change_layer_multiplier in zip(change_layers, change_layer_multipliers):
        model.blocks[change_layer].mlp.additional_gradients = grads[change_layer] * change_layer_multiplier
    
    return model


def update_model(model, logits, tokens_arr, answer_token_indices, grads, change_layers, change_layer_multipliers, repeat_times=1):
    logits[:, -1, answer_token_indices[0, 1]].backward()

    model = apply_gradient(model=model, grads=grads, change_layers=change_layers, change_layer_multipliers=change_layer_multipliers)

    logits = model(tokens_arr)

    for _ in range(1, repeat_times):
        logits[:, -1, answer_token_indices[0, 1]].backward()

        model = apply_gradient(model=model, grads=grads, change_layers=change_layers, change_layer_multipliers=change_layer_multipliers)

        logits = model(tokens_arr)
    
    return model, logits











if __name__ == "__main__":

    model_name = "gpt2"
    vocab_path = "/data/projects/punim2522/models/gpt2/vocab.json"
    transcoders_path = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    sub_transcoders_num = 12
    change_layers = [i for i in range(sub_transcoders_num)]
    change_layer_multipliers = [1 for i in range(sub_transcoders_num)]
    save_grads = True
    view_gradient_layer = 11
    repeat_times = 2

    prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded"
    prompt_answer = (" Mary", " Emma")




    model = load_model(model_name=model_name)
    vocab = load_vocab(vocab_path=vocab_path)
    transcoder = load_transcoders(transcoders_path=transcoders_path, sub_transcoders_num=sub_transcoders_num)


    model, grads = sub_transcoders_wrapper(model=model, transcoders=transcoder, sub_transcoders_num=sub_transcoders_num, save_grads=save_grads)
    tokens_arr = model.to_tokens(prompt)


    print("tokens_arr:", tokens_arr)

    logits = model(tokens_arr)

    print("logits shape:", logits.shape)

    print("top 20 logits:", torch.topk(logits[0, -1], k=20))

    print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])


    model, grads = save_gradient(model=model, transcoders=transcoder, grads=grads)

    answer_token_indices = torch.tensor([[model.to_single_token(prompt_answer[0]), model.to_single_token(prompt_answer[1])]], device=model.cfg.device)


    print("answer_token_indices: ", answer_token_indices)

    print("correct logits before applying gradient:", logits[:, -1, answer_token_indices[0, 0]])

    print("incorrect logits before applying gradient:", logits[:, -1, answer_token_indices[0, 1]])


    model, logits = update_model(model=model, logits=logits, tokens_arr=tokens_arr, answer_token_indices=answer_token_indices, grads=grads, change_layers=change_layers, change_layer_multipliers=change_layer_multipliers, repeat_times=repeat_times)

    normalized_gradient = normalize_gradient(model=model, grads=grads)


    print(f"grads[{view_gradient_layer}].shape: ", grads[view_gradient_layer].shape)

    print("normalized_gradient: ", normalized_gradient)

    print("logits shape after applying gradient:", logits.shape)

    print("top 20 logits after applying gradient:", torch.topk(logits[0, -1], k=20))

    print("top 20 words after applying gradient:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

    print("correct logits after applying gradient:", logits[:, -1, answer_token_indices[0, 0]])

    print("incorrect logits after applying gradient:", logits[:, -1, answer_token_indices[0, 1]])

    print("logits diff after applying gradient:", logits[:, -1, answer_token_indices[0, 0]] - logits[:, -1, answer_token_indices[0, 1]])

    # print("layer top 20 features on the last token after applying gradient:", torch.topk(model.blocks[change_layer].mlp.hidden_acts[0, -1], k=20))
