import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelWithLMHead, GPT2Tokenizer, GPT2Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sparsify import flatten, count_parameters
import sys

LOCAL_FOLDER = 'local-models'


def assess(model_name, amount=0.1):
    model_path = f"../{LOCAL_FOLDER}/{model_name}-{amount}"
    tokenizer_path = f"../{LOCAL_FOLDER}/{model_name}-tokenizer"
    if model_name == 'gpt2' or model_name == 'gpt2-xl':
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    elif model_name == 'microsoft/deberta-v2-xxlarge' or model_name == 'microsoft/deberta-v2-xlarge':
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
    elif model_name == 't5-3b' or model_name == 't5-base':
        model = AutoModelWithLMHead.from_pretrained(model_path, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    modules = flatten(model)
    parameters = [(m, p) for m in modules for n, p in m.named_parameters() if n.endswith('weight')]

    s = 0
    th = 0.5
    cnt = 0
    for m, p in parameters:
        if p.requires_grad:
            n = torch.sum(torch.abs(p) > th)
            N = p.numel()
            print(f"Layer {cnt}:", m, ", (p>>0)n=", n.numpy(), ", (total)N=", N, ",fraction=", n.numpy()/N)
            cnt += 1
            s += n

    print("Total Number >> 0:", s.numpy())
    T = count_parameters(model)
    print("Model Total Parameters:", T)
    print("Fraction: ", s.numpy()/T)


if __name__ == '__main__':
    model_names = ['gpt2', 'microsoft/deberta-v2-xlarge', 't5-base']
    ori_stdout = sys.stdout
    for m in model_names:
         with open(f'../assessments/{m}-gt0.txt', 'w') as f:
            sys.stdout = f
            assess(m, 0.0)