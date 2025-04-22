from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import json
import os



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


def compute_sigma(frequencies):
    # sigma_ = []
    # token_num = 9102780416
    # for layer_ in range(int(frequencies.shape[0])):
    #     print(f"layer_: {layer_}")
    #     torch.cuda.empty_cache()
    #     path = f"/data/projects/punim2522/models/gpt2-transcoders/sigma/{layer_}.pt"
    #     if os.path.exists(path):
    #         sub_sigma = torch.load(path)
    #     else:
    #         sub_sigma = []
    #         for feature_idx in range(int(frequencies.shape[1])):
    #             torch.cuda.empty_cache()
    #             tmp_vector = torch.cat((torch.ones(int(token_num * frequencies[layer_][feature_idx]), device=model.cfg.device), torch.zeros(token_num - int(token_num * frequencies[layer_][feature_idx]), device=model.cfg.device)))
    #             # print(tmp_vector)
    #             # print(torch.std(tmp_vector))
    #             sub_sigma.append(torch.std(tmp_vector).view(1, 1))
    #         torch.save(sub_sigma, path)
    #         # print(torch.cat(sub_sigma).view(1, -1).shape)
    #     sigma_.append(torch.cat(sub_sigma).view(1, -1))
    sigma_ = []
    token_num = 9102780416
    for layer_ in range(int(frequencies.shape[0])):
        # print(f"layer_: {layer_}")
        path = f"/data/projects/punim2522/models/gpt2-transcoders/sigma/{layer_}.pt"
        if os.path.exists(path):
            sub_sigma = torch.load(path)
        else:
            sub_sigma = []
            for feature_idx in range(int(frequencies.shape[1])):
                num_freq = int(token_num * frequencies[layer_][feature_idx])
                num_zero = token_num - num_freq
                current_sigma = num_freq * (1 - frequencies[layer_][feature_idx]) * (1 - frequencies[layer_][feature_idx]) + num_zero * frequencies[layer_][feature_idx] * frequencies[layer_][feature_idx]
                current_sigma = torch.sqrt(current_sigma / token_num)
                sub_sigma.append(current_sigma.view(1, 1))
            torch.save(sub_sigma, path)
        # print("top 20 sigma[{}]:".format(layer_), torch.topk(torch.cat(sub_sigma).view(1, -1), k=5))
        sigma_.append(torch.cat(sub_sigma).view(1, -1))

    # print(sigma_)

    return torch.cat(sigma_, dim=0)


def load_frequencies(frequences_path, sub_transcoders_num, include_dead_features=True):
    frequencies = []
    for i in range(sub_transcoders_num):
        frequencies.append(torch.load(f"{frequences_path.format(i)}_log_feature_sparsity.pt"))

    # print("frequencies[8][0]:", frequencies[8][0])

    frequencies = torch.stack(frequencies)
    # print("frequencies[0][:20]:", frequencies[0][:20])
    frequencies = torch.pow(10, frequencies)

    print("frequencies.shape:", frequencies.shape)
    print("frequencies[0][:20]:", frequencies[0][:20])

    # for i in range(sub_transcoders_num):
    #     print("frequencies[{}].shape:".format(i), frequencies[i].shape)
    #     print("sum(frequencies[{}])".format(i), torch.sum(frequencies[i]))
    #     print("top 20 frequencies[{}]:".format(i), torch.topk(frequencies[i], k=5))
    
    # print("frequencies.shape:", frequencies.shape)
    # print("frequencies[0][:20]:", frequencies[0][:20])

    sigma_ = compute_sigma(frequencies=frequencies)

    # print("sigma_.shape:", sigma_.shape)
    # print("sigma_[0][:20]:", sigma_[0][:20])

    if include_dead_features:
        sorted_frequencies, indices = torch.sort(frequencies, dim=1)
        print(sorted_frequencies[0][:2000])
        # print(sorted_frequencies[0][-1000:])

    return frequencies, sigma_


def sub_transcoders_wrapper(model, transcoders, sub_transcoders_num, save_grads=False):
    original_mlps = []
    for transcoder in transcoders[:sub_transcoders_num]:
        original_mlps.append(model.blocks[transcoder.cfg.hook_point_layer].mlp)
        model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)

    if not save_grads:
        return model, None, original_mlps

    # Clean up memory
    import gc

    gc.collect()
    torch.cuda.empty_cache()


    grads = {}
    
    return model, grads, original_mlps


def sub_transcoders_wrapper_specific_layer(model, transcoder, save_grads=False):
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


def apply_gradient_with_one_layer(model, grads, change_layer, change_layer_multipliers):
    model.blocks[change_layer].mlp.additional_gradients = grads[change_layer] * change_layer_multipliers[change_layer]
    model.blocks[change_layer].mlp.additional_gradients.requires_grad = False

    
    return model


def update_model(model, transcoders, logits, tokens_arr, answer_token_indices, grads, change_layers, change_layer_multipliers, repeat_times=1):
    logits[:, -1, answer_token_indices[0, 1]].backward()

    model = apply_gradient(model=model, grads=grads, change_layers=change_layers, change_layer_multipliers=change_layer_multipliers)

    logits = model(tokens_arr)

    for _ in range(1, repeat_times):
        grads = {}
        model, grads = save_gradient(model=model, transcoders=transcoders, grads=grads)
        model.zero_grad()

        logits[:, -1, answer_token_indices[0, 1]].backward()

        # print("grads[0]:", grads[0])

        model = apply_gradient(model=model, grads=grads, change_layers=change_layers, change_layer_multipliers=change_layer_multipliers)

        logits = model(tokens_arr)
    
    return model, logits


def top_k_features(model, layer, k=20):
    values, i = torch.topk(model.blocks[layer].mlp.hidden_acts[0].flatten(), k=k)
    indices = torch.column_stack(torch.unravel_index(i, model.blocks[layer].mlp.hidden_acts[0].shape))

    return values, indices


def top_m_gradients(grads, layer, m=20, is_abs=True):
    if is_abs:
        values, i = torch.topk(torch.abs(grads[layer][0].flatten()), k=m)
    else:
        values, i = torch.topk(grads[layer][0].flatten(), k=m)

    indices = torch.column_stack(torch.unravel_index(i, grads[layer][0].shape))

    return values, indices


def top_m_gradients_v2(grads, m=20, is_abs=True):
    if is_abs:
        values, i = torch.topk(torch.abs(grads.flatten()), k=m)
    else:
        values, i = torch.topk(grads.flatten(), k=m)

    indices = torch.column_stack(torch.unravel_index(i, grads.shape))

    return values, indices


def show_features_words(model, layer, indices, vocab, top_tokens_each_feature=5):
    words = model.W_E @ model.blocks[layer].mlp.transcoder.W_enc[:, indices]

    words = torch.transpose(words, 0, 1)

    # print("words shape: ", words.shape)

    most_pos = torch.topk(words, k=top_tokens_each_feature)
    
    top_idxs = most_pos.indices.cpu().tolist()

    # print("top_idxs: ", top_idxs)

    # print("len(top_idxs): ", len(top_idxs))

    top_tokens = [[vocab[top_idxs[idx][j]] for j in range(len(top_idxs[idx]))] for idx in range(len(top_idxs))]

    # print("top_tokens: ", top_tokens)

    for idx_ in range(len(top_tokens)):
        print(top_tokens[idx_])

    return top_tokens


def update_algorithm_node_v1(model, tokens_arr, answer_token_indices, sub_transcoders_num, top_k_features_each_layer, top_m_gradients_each_layer, transcoders, frequencies, sigma_, range_n):

    for layer_idx in range(sub_transcoders_num):
        logits = model(tokens_arr)

        if layer_idx == 0:
            print("logits shape:", logits.shape)
            print("top 20 logits:", torch.topk(logits[0, -1], k=20))
            print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])
            print("correct logits:", logits[:, -1, answer_token_indices[0]])
        grads = {}
        model, grads = save_gradient(model=model, transcoders=transcoders, grads=grads)
        model.zero_grad()

        
        # top_k_features_values, top_k_features_indices = top_k_features(model=model, layer=layer_idx, k=top_k_features_each_layer)

        # print(f"layer {layer_idx} top {top_k_features_each_layer} features:", top_k_features_values[:5])
        # print(f"layer {layer_idx} top {top_k_features_each_layer} features indices:", top_k_features_indices[:5])
    
        logits[:, -1, answer_token_indices[0]].backward()

        sub_grads = grads
        # sub_grads = torch.zeros_like(torch.randn(sub_transcoders_num, grads[0].shape[0], grads[0].shape[1], grads[0].shape[2]))
        # for i in range(top_k_features_each_layer):
        #     sub_grads[layer_idx][0][top_k_features_indices[i][0], top_k_features_indices[i][1]] = grads[layer_idx][0][top_k_features_indices[i][0], top_k_features_indices[i][1]]

        # print("sub_grads[0].shape:", sub_grads[0].shape)

        top_m_gradients_values, top_m_gradients_indices = top_m_gradients(grads=sub_grads, layer=layer_idx, m=top_m_gradients_each_layer, is_abs=True)

        # print(f"layer {layer_idx} top {top_m_gradients_each_layer} gradients:", top_m_gradients_values[:5])
        # print(f"layer {layer_idx} top {top_m_gradients_each_layer} gradients indices:", top_m_gradients_indices[:5])
        # print(grads[layer_idx][0][9, 5545])
        # print(grads[layer_idx][0][7, 9886])

        # applied_gradients = torch.zeros_like(sub_grads)
        # change_layer_multipliers = torch.zeros_like(sub_grads)
        applied_gradients = torch.zeros_like(torch.randn(sub_transcoders_num, grads[0].shape[0], grads[0].shape[1], grads[0].shape[2]))
        change_layer_multipliers = torch.zeros_like(torch.randn(sub_transcoders_num, grads[0].shape[0], grads[0].shape[1], grads[0].shape[2]))
        # tmp_alpha = []
        # tmp_grads = []
        for i in range(top_m_gradients_each_layer):
            applied_gradients[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] = grads[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]]

            if grads[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] > 0:
                alpha_ = frequencies[layer_idx][top_m_gradients_indices[i][1]] + range_n * sigma_[layer_idx][top_m_gradients_indices[i][1]]
            else:
                alpha_ = frequencies[layer_idx][top_m_gradients_indices[i][1]] + range_n * sigma_[layer_idx][top_m_gradients_indices[i][1]]

            # change_layer_multipliers[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] = min(alpha_, 3)
            change_layer_multipliers[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] = alpha_
            # tmp_alpha.append(alpha_)
            # tmp_grads.append(grads[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]])
            # if i < 4:
            #     print(frequencies[layer_idx][top_m_gradients_indices[i][1]], range_n, sigma_[layer_idx][top_m_gradients_indices[i][1]], alpha_, grads[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]], applied_gradients[layer_idx][0][top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]])
        
        # tmp_alpha = torch.tensor(tmp_alpha)
        # tmp_grads = torch.tensor(tmp_grads)
        # print("top 20 abs alpha:", torch.topk(torch.abs(tmp_alpha), k=5))
        # print("top 20 grads:", torch.topk(torch.abs(tmp_grads), k=5))

        # tmp_values, tmp_indices = top_m_gradients(grads=change_layer_multipliers, layer=layer_idx, m=5, is_abs=True)

        # print("top 5 abs change_layer_multipliers:", tmp_values)
        # print("top 5 change_layer_multipliers indices:", tmp_indices)

        # for i in range(5):
        #     print(applied_gradients[layer_idx][0][tmp_indices[i][0], tmp_indices[i][1]], change_layer_multipliers[layer_idx][0][tmp_indices[i][0], tmp_indices[i][1]], model.blocks[layer_idx].mlp.hidden_acts[0][tmp_indices[i][0], tmp_indices[i][1]], tmp_indices[i][0], tmp_indices[i][1], frequencies[layer_idx][tmp_indices[i][1]], sigma_[layer_idx][tmp_indices[i][1]])

        model = apply_gradient_with_one_layer(model=model, grads=applied_gradients.cuda(), change_layer=layer_idx, change_layer_multipliers=change_layer_multipliers.cuda())
        # model = apply_gradient(model=model, grads=applied_gradients.cuda(), change_layers=[layer_idx], change_layer_multipliers=change_layer_multipliers.cuda())

    return model


def update_algorithm_node_v2(model, tokens_arr, answer_token_indices, sub_transcoders_num, top_m_gradients_each_layer, frequencies, sigma_, range_n):

    for layer_idx in range(sub_transcoders_num):
        logits = model(tokens_arr)

        if layer_idx == 0:
            print("logits shape:", logits.shape)
            print("top 20 logits:", torch.topk(logits[0, -1], k=20))
            print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])
            print("correct logits:", logits[:, -1, answer_token_indices[0]])
        model.zero_grad()
    
        logits[:, -1, answer_token_indices[0]].backward()

        top_m_gradients_values, top_m_gradients_indices = top_m_gradients_v2(grads=model.blocks[layer_idx].mlp.transcoder.W_enc.grad, m=top_m_gradients_each_layer, is_abs=True)


        # print(f"layer {layer_idx} top {top_m_gradients_each_layer} gradients:", top_m_gradients_values[:5])
        # print(f"layer {layer_idx} top {top_m_gradients_each_layer} gradients indices:", top_m_gradients_indices[:5])


        applied_gradients = torch.zeros_like(model.blocks[layer_idx].mlp.transcoder.W_enc.grad)
        change_layer_multipliers = torch.zeros_like(model.blocks[layer_idx].mlp.transcoder.W_enc.grad)


        for i in range(top_m_gradients_each_layer):
            applied_gradients[top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] = model.blocks[layer_idx].mlp.transcoder.W_enc.grad[top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]]

            if model.blocks[layer_idx].mlp.transcoder.W_enc.grad[top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] > 0:
                alpha_ = frequencies[layer_idx][top_m_gradients_indices[i][1]] + range_n * sigma_[layer_idx][top_m_gradients_indices[i][1]]
            else:
                alpha_ = frequencies[layer_idx][top_m_gradients_indices[i][1]] + range_n * sigma_[layer_idx][top_m_gradients_indices[i][1]]

            change_layer_multipliers[top_m_gradients_indices[i][0], top_m_gradients_indices[i][1]] = alpha_

        model.blocks[layer_idx].mlp.transcoder.W_enc.data = model.blocks[layer_idx].mlp.transcoder.W_enc.data + change_layer_multipliers * applied_gradients

    return model


def update_algorithm_node_v3(model, tokens_arr, answer_token_indices, vocab, top_m_gradients_each_layer, frequencies, sigma_, range_n):

    # for layer_idx in range(sub_transcoders_num-1):
    for layer_idx in range(1):
        logits = model(tokens_arr)

        # if layer_idx == 0:
        print("layer_idx:", layer_idx)
        print("logits shape:", logits.shape)
        print("top 20 logits:", torch.topk(logits[0, -1], k=20))
        print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])
        print("correct logits:", logits[:, -1, answer_token_indices[0]])
        model.zero_grad()
    
        logits[:, -1, answer_token_indices[0]].backward()

        top_m_W_dec_gradients_values, top_m_W_dec_gradients_indices = top_m_gradients_v2(grads=model.blocks[layer_idx].mlp.transcoder.W_dec.grad, m=top_m_gradients_each_layer, is_abs=True)

        top_m_W_enc1_gradients_values, top_m_W_enc1_gradients_indices = top_m_gradients_v2(grads=model.blocks[layer_idx+1].mlp.transcoder.W_enc.grad, m=top_m_gradients_each_layer, is_abs=True)


        applied_W_dec_gradients = torch.zeros_like(model.blocks[layer_idx].mlp.transcoder.W_dec.grad)
        change_W_dec_multipliers = torch.zeros_like(model.blocks[layer_idx].mlp.transcoder.W_dec.grad)

        applied_W_enc1_gradients = torch.zeros_like(model.blocks[layer_idx+1].mlp.transcoder.W_enc.grad)
        change_W_enc1_multipliers = torch.zeros_like(model.blocks[layer_idx+1].mlp.transcoder.W_enc.grad)

        index_W_dec = 0
        index_W_enc1 = 0

        
        def compute_alpha(current_gradient, current_layer, current_index):
            if current_gradient > 0:
                alpha_ = frequencies[current_layer][current_index] + range_n * sigma_[current_layer][current_index]
            else:
                alpha_ = frequencies[current_layer][current_index] + range_n * sigma_[current_layer][current_index]

            return alpha_


        for i in range(top_m_gradients_each_layer):

            if top_m_W_dec_gradients_values[index_W_dec] > top_m_W_enc1_gradients_values[index_W_enc1]:
                applied_W_dec_gradients[top_m_W_dec_gradients_indices[index_W_dec][0], top_m_W_dec_gradients_indices[index_W_dec][1]] = model.blocks[layer_idx].mlp.transcoder.W_dec.grad[top_m_W_dec_gradients_indices[index_W_dec][0], top_m_W_dec_gradients_indices[index_W_dec][1]]

                alpha_ = compute_alpha(current_gradient=model.blocks[layer_idx].mlp.transcoder.W_dec.grad[top_m_W_dec_gradients_indices[index_W_dec][0], top_m_W_dec_gradients_indices[index_W_dec][1]], current_layer=layer_idx, current_index=top_m_W_dec_gradients_indices[index_W_dec][1])

                change_W_dec_multipliers[top_m_W_dec_gradients_indices[index_W_dec][0], top_m_W_dec_gradients_indices[index_W_dec][1]] = alpha_

                index_W_dec += 1
            else:
                applied_W_enc1_gradients[top_m_W_enc1_gradients_indices[index_W_enc1][0], top_m_W_enc1_gradients_indices[index_W_enc1][1]] = model.blocks[layer_idx+1].mlp.transcoder.W_enc.grad[top_m_W_enc1_gradients_indices[index_W_enc1][0], top_m_W_enc1_gradients_indices[index_W_enc1][1]]

                alpha_ = compute_alpha(current_gradient=model.blocks[layer_idx+1].mlp.transcoder.W_enc.grad[top_m_W_enc1_gradients_indices[index_W_enc1][0], top_m_W_enc1_gradients_indices[index_W_enc1][1]], current_layer=layer_idx+1, current_index=top_m_W_enc1_gradients_indices[index_W_enc1][1])

                change_W_enc1_multipliers[top_m_W_enc1_gradients_indices[index_W_enc1][0], top_m_W_enc1_gradients_indices[index_W_enc1][1]] = alpha_

                index_W_enc1 += 1


        model.blocks[layer_idx].mlp.transcoder.W_dec.data = model.blocks[layer_idx].mlp.transcoder.W_dec.data + change_W_dec_multipliers * applied_W_dec_gradients

        model.blocks[layer_idx+1].mlp.transcoder.W_enc.data = model.blocks[layer_idx+1].mlp.transcoder.W_enc.data + change_W_enc1_multipliers * applied_W_enc1_gradients

    return model

        
    
def interactive_with_user(model):
    while True:
        prompt = input("Enter a prompt: ")
        print("prompt:", prompt)
        prompt_answer = input("Enter the answer: ")
        print("prompt_answer:", prompt_answer)
        prompt_answer = [prompt_answer]

        tokens_arr = model.to_tokens(prompt)
        
        print("tokens_arr:", tokens_arr)

        answer_token_indices = torch.tensor([model.to_single_token(prompt_answer[0])], device=model.cfg.device)

        print("answer_token_indices:", answer_token_indices)

        logits = model(tokens_arr)

        print("logits shape:", logits.shape)

        print("top 20 logits:", torch.topk(logits[0, -1], k=20))

        print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

        print("correct logits:", logits[:, -1, answer_token_indices[0]])

        flag = input("Enter the flag: ")

        if flag == "0":
            continue


        top_m_gradients_each_layer = int(input("Enter the top m gradients each layer: "))
        range_n = int(input("Enter the range n: "))

        update_algorithm_node_v3(model=model, tokens_arr=tokens_arr, answer_token_indices=answer_token_indices, sub_transcoders_num=sub_transcoders_num, top_m_gradients_each_layer=top_m_gradients_each_layer, frequencies=frequencies, sigma_=sigma_, range_n=range_n)

        logits = model(tokens_arr)

        print("logits shape:", logits.shape)

        print("top 20 logits:", torch.topk(logits[0, -1], k=20))

        print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

        print("correct logits:", logits[:, -1, answer_token_indices[0]])





if __name__ == "__main__":

    model_name = "gpt2"
    vocab_path = "/data/projects/punim2522/models/gpt2/vocab.json"
    transcoders_path = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    sub_transcoders_num = 12
    change_layers = [i for i in range(sub_transcoders_num)]
    change_layer_multipliers = [1 for i in range(sub_transcoders_num)]
    save_grads = False
    view_gradient_layer = 0
    repeat_times = 1
    top_k_features_each_layer = 1000
    # top_m_gradients_each_layer = 24576
    top_m_gradients_each_layer = 2000
    range_n = 5
    # 2000 7 work for most of examples

    # prompt = "The first name of the person Donald Trump is"
    # prompt_answer = (" Donald", " Trump")
    # prompt_answer = (" Trump", " Donald")
    # prompt_answer = (" Trump", " dog")
    # prompt = "In the hotel laundry room, Emma burned Mary's shirt, so the manager scolded"
    # prompt_answer = (" Mary", " Emma")
    # prompt_answer = (" Emma", " Mary")
    # prompt_answer = (" Emma", " dog")
    prompt = "The number of r's in the word strawberry is:"
    prompt_answer = (" he", " 3")

    # 看一下top feature具体代表的词


    model = load_model(model_name=model_name)
    vocab = load_vocab(vocab_path=vocab_path)
    transcoder = load_transcoders(transcoders_path=transcoders_path, sub_transcoders_num=sub_transcoders_num)
    frequencies, sigma_ = load_frequencies(frequences_path=transcoders_path, sub_transcoders_num=sub_transcoders_num)

    # print("frequencies[0]:", frequencies[0].shape)
    # print("frequencies[0][:20]:", torch.pow(frequencies[0][:20], 10))

    model, grads = sub_transcoders_wrapper(model=model, transcoders=transcoder, sub_transcoders_num=sub_transcoders_num, save_grads=save_grads)

    interactive_with_user(model=model)
    # tokens_arr = model.to_tokens(prompt)


    # print("tokens_arr:", tokens_arr)

    # answer_token_indices = torch.tensor([model.to_single_token(prompt_answer[1])], device=model.cfg.device)

    # print("answer_token_indices:", answer_token_indices)

    # update_algorithm_node_v2(model=model, tokens_arr=tokens_arr, answer_token_indices=answer_token_indices, sub_transcoders_num=sub_transcoders_num, top_m_gradients_each_layer=top_m_gradients_each_layer, frequencies=frequencies, sigma_=sigma_, range_n=range_n)



    # logits = model(tokens_arr)

    # print("logits shape:", logits.shape)

    # print("top 20 logits:", torch.topk(logits[0, -1], k=20))

    # print("top 20 words:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

    # print("correct logits:", logits[:, -1, answer_token_indices[0]])


    # model, grads = save_gradient(model=model, transcoders=transcoder, grads=grads)

    # answer_token_indices = torch.tensor([[model.to_single_token(prompt_answer[0]), model.to_single_token(prompt_answer[1])]], device=model.cfg.device)


    # print("answer_token_indices: ", answer_token_indices)

    # print("correct logits before applying gradient:", logits[:, -1, answer_token_indices[0, 0]])

    # print("incorrect logits before applying gradient:", logits[:, -1, answer_token_indices[0, 1]])

    # values, indices = top_k_features(model=model, layer=view_gradient_layer, k=top_k_features_each_layer)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features:", values)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features indices:", indices)

    # show_features_words(model=model, layer=view_gradient_layer, indices=indices[:, -1].flatten(), vocab=vocab, top_tokens_each_feature=3)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the 0-th token before applying gradient:", torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, 0], k=top_k_features_each_layer))

    # show_features_words(model=model, layer=view_gradient_layer, indices=torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, 0], k=top_k_features_each_layer).indices, vocab=vocab, top_tokens_each_feature=3)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the 1-th token before applying gradient:", torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, 1], k=top_k_features_each_layer))

    # show_features_words(model=model, layer=view_gradient_layer, indices=torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, 1], k=top_k_features_each_layer).indices, vocab=vocab, top_tokens_each_feature=3)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the 2-th token before applying gradient:", torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, 2], k=top_k_features_each_layer))

    # show_features_words(model=model, layer=view_gradient_layer, indices=torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, 2], k=top_k_features_each_layer).indices, vocab=vocab, top_tokens_each_feature=3)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the -2-th token before applying gradient:", torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, -2], k=top_k_features_each_layer))

    # show_features_words(model=model, layer=view_gradient_layer, indices=torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, -2], k=top_k_features_each_layer).indices, vocab=vocab, top_tokens_each_feature=3)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the -1-th token before applying gradient:", torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, -1], k=top_k_features_each_layer))

    # show_features_words(model=model, layer=view_gradient_layer, indices=torch.topk(model.blocks[view_gradient_layer].mlp.hidden_acts[0, -1], k=top_k_features_each_layer).indices, vocab=vocab, top_tokens_each_feature=3)

    # values, indices = top_k_features(model=model, layer=view_gradient_layer, k=top_k_features_each_layer)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the last token before applying gradient:", values)

    # print(f"layer {view_gradient_layer} top {top_k_features_each_layer} features on the last token before applying gradient:", indices)

    # model, logits = update_model(model=model, transcoders=transcoder, logits=logits, tokens_arr=tokens_arr, answer_token_indices=answer_token_indices, grads=grads, change_layers=change_layers, change_layer_multipliers=change_layer_multipliers, repeat_times=repeat_times)

    # # normalized_gradient = normalize_gradient(model=model, grads=grads)

    # print(f"grads[{view_gradient_layer}].shape: ", grads[view_gradient_layer].shape)

    # # print("normalized_gradient: ", normalized_gradient)

    # print("logits shape after applying gradient:", logits.shape)

    # print("top 20 logits after applying gradient:", torch.topk(logits[0, -1], k=20))

    # print("top 20 words after applying gradient:", [vocab[int(i)] for i in torch.topk(logits[0, -1], k=20).indices])

    # print("correct logits after applying gradient:", logits[:, -1, answer_token_indices[0, 0]])

    # print("incorrect logits after applying gradient:", logits[:, -1, answer_token_indices[0, 1]])

    # print("logits diff after applying gradient:", logits[:, -1, answer_token_indices[0, 0]] - logits[:, -1, answer_token_indices[0, 1]])

    # # print("layer top 20 features on the last token after applying gradient:", torch.topk(model.blocks[change_layer].mlp.hidden_acts[0, -1], k=20))
