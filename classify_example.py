from back_gradients import *



def read_jsonl(file_path):
    print("reading from", file_path)
    results = []
    with open(file_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def read_text(file_path):
    print("reading from", file_path)
    results = []
    with open(file_path, "r") as f:
        for line in f:
            results.append(line.strip())
    return results


def write_jsonl(file_path, data):
    print("writing to", file_path)
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def run_model(model, vocab, data, batch_size=16):
    correct_list = []
    incorrect_list = []

    from tqdm import tqdm

    pbar = tqdm(len(data))

    start_idx = 0
    while start_idx < len(data):
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = [" ".join(data[idx_]["text"].strip().split(" ")[:-1]) for idx_ in range(start_idx, end_idx)]

        correct_labels = [" " + data[idx_]["text"].strip().split(" ")[-1] for idx_ in range(start_idx, end_idx)]
        
        tokens_arr = model.to_tokens(batch_data)

        token_label = model.to_tokens(correct_labels)

        # print(token_label)

        index_i = []
        for i in range(end_idx - start_idx):
            index_i.append(i)

        index_j = []
        for i in range(end_idx - start_idx):
            j = len(tokens_arr[i]) - 1
            while j >=0 and tokens_arr[i][j] == 50256:
                j -= 1
            index_j.append(j)

        # print("index_j:", index_j)

        # print("index_i:", index_i)
                

        # print("batch_data:", batch_data)
        
        # print("tokens_arr:", tokens_arr)

        logits = model(tokens_arr)

        model_predicts = torch.argmax(torch.softmax(logits[index_i, index_j, :], dim=-1), dim=-1)
        model_tokens = [vocab[idx_] for idx_ in model_predicts.cpu().tolist()]
        # print(logits[index_i, index_j, :].shape)

        # for idx_ in range(end_idx - start_idx):
        #     model_predicts = [vocab[idx_] for idx_ in torch.argmax(torch.softmax(logits[idx_, :, :], dim=-1), dim=-1).cpu().tolist()]
        #     print("model_predicts:", model_predicts)
        #     print("index_j:", index_j[idx_])
        

        # model_predicts = [vocab[idx_] for idx_ in torch.argmax(torch.softmax(logits[index_i, index_j, :], dim=-1), dim=-1).cpu().tolist()]

        # print(torch.argmax(torch.softmax(logits[index_i, index_j, :], dim=-1), dim=-1).cpu().tolist())

        # print("model_predicts:", model_predicts)

        # print(torch.topk(torch.softmax(logits[:, -1, :], dim=-1), k=5, dim=-1).indices.cpu().tolist())

        # break

        for idx_ in range(end_idx - start_idx):
            if model_predicts[idx_] == token_label[idx_][1]:
                correct_list.append({"prompt": batch_data[idx_], "correct_label": correct_labels[idx_], "model_predict": model_tokens[idx_].replace("\u0120", " "), "id": start_idx + idx_})
            else:
                incorrect_list.append({"prompt": batch_data[idx_], "correct_label": correct_labels[idx_], "model_predict": model_tokens[idx_].replace("\u0120", " "), "id": start_idx + idx_})

        start_idx = end_idx

        pbar.update(batch_size)
    
    pbar.close()
    return correct_list, incorrect_list



if __name__ == "__main__":

    model_name = "gpt2"
    vocab_path = "/data/projects/punim2522/models/gpt2/vocab.json"

    # data = read_jsonl("/home/shuhewang/Easy-Transformer/ioi_prompts.jsonl")
    data = read_text("/home/shuhewang/transcoder_circuits/gpt_ioi_sentences.txt")

    data = [{"text": item} for item in data]

    model = load_model(model_name=model_name)
    vocab = load_vocab(vocab_path=vocab_path)

    correct_list, incorrect_list = run_model(model, vocab, data, batch_size=4)

    print("correct_list:", len(correct_list))
    print("incorrect_list:", len(incorrect_list))

    write_jsonl("/home/shuhewang/transcoder_circuits/correct_list_gpt.jsonl", correct_list)
    write_jsonl("/home/shuhewang/transcoder_circuits/incorrect_list_gpt.jsonl", incorrect_list)