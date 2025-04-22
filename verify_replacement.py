from back_gradients import *
from utils import *
from tqdm import tqdm







@torch.no_grad()
def compute_last_layer_logits(model, data, batch_size=16):

    results = []
    start_idx = 0

    pbar = tqdm(len(data))

    while start_idx < len(data):
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = [item["prompt"] for item in data[start_idx:end_idx]]
        
        tokens_arr = model.to_tokens(batch_data)

        index_i = []
        for i in range(end_idx - start_idx):
            index_i.append(i)

        index_j = []
        for i in range(end_idx - start_idx):
            j = len(tokens_arr[i]) - 1
            while j >=0 and tokens_arr[i][j] == 50256:
                j -= 1
            index_j.append(j)

        logits = model(tokens_arr)

        results.append(logits[index_i, index_j, :])

        pbar.update(end_idx - start_idx)
        start_idx = end_idx

    pbar.close()

    results = torch.cat(results, dim=0)

    return results






if __name__ == "__main__":

    model_name = "gpt2"
    vocab_path = "/data/projects/punim2522/models/gpt2/vocab.json"
    transcoders_path = "/data/projects/punim2522/models/gpt2-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    sub_transcoders_num = 12
    save_grads = False

    model = load_model(model_name)

    # data = read_jsonl(data_path="/home/shuhewang/transcoder_circuits/ioi_data_format.jsonl")
    # data = read_jsonl(data_path="/home/shuhewang/transcoder_circuits/correct_list.jsonl")
    data = read_jsonl(data_path="/home/shuhewang/transcoder_circuits/incorrect_list.jsonl")

    previous_logits = compute_last_layer_logits(model, data, batch_size=16)

    transcoders = load_transcoders(transcoders_path, sub_transcoders_num)

    original_mlps = [model.blocks[t.cfg.hook_point_layer].mlp for t in transcoders]

    # for idx_ in range(1, sub_transcoders_num + 1):
    #     print(f"idx_: {idx_}")
    #     model, grads = sub_transcoders_wrapper(model, transcoders, idx_, save_grads)
    #     last_logits = compute_last_layer_logits(model, data, batch_size=4)
    #     kl_div = compute_kl_div(previous_vector=previous_logits, now_vector=last_logits)
    #     print(kl_div)

    #     torch.cuda.empty_cache()

    for idx_ in range(sub_transcoders_num):
        print(f"idx_: {idx_}")
        model, grads = sub_transcoders_wrapper_specific_layer(model, transcoders[idx_], save_grads)
        last_logits = compute_last_layer_logits(model, data, batch_size=4)
        kl_div = compute_kl_div(previous_vector=previous_logits, now_vector=last_logits)
        print(kl_div)

        model.blocks[transcoders[idx_].cfg.hook_point_layer].mlp = original_mlps[transcoders[idx_].cfg.hook_point_layer]

        torch.cuda.empty_cache()
        