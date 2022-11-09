import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelWithLMHead, GPT2Tokenizer, GPT2Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sparsify import flatten, count_parameters
import sys
import numpy as np
LOCAL_FOLDER = 'local-models'


def assess(model_name, amount=0.1):
    model_path = f"../{LOCAL_FOLDER}/{model_name}-{amount}"
    tokenizer_path = f"../{LOCAL_FOLDER}/{model_name}-tokenizer"
    mn = model_name
    if model_name == 'gpt2' or model_name == 'gpt2-xl':
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    elif model_name == 'microsoft/deberta-v2-xxlarge' or model_name == 'microsoft/deberta-v2-xlarge':
        mn = 'deberta'
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
    elif model_name == 't5-3b' or model_name == 't5-base':
        model = AutoModelWithLMHead.from_pretrained(model_path, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    modules = flatten(model)
    parameters = [(m, p) for m in modules for n, p in m.named_parameters() if n.endswith('weight')]
    ps = [p for p in model.parameters() if p.requires_grad]

    a = np.array([])
    for p in ps:
        fp = torch.flatten(p).detach().numpy()
        a = np.concatenate((a, fp))
    print(len(a))
    import matplotlib.pyplot as plt
    x, bins, p = plt.hist(a, log=True)
    plt.savefig(mn + '_hist')
    plt.close()


if __name__ == '__main__':
    model_names = ['microsoft/deberta-v2-xlarge', 't5-base']
    # assess('t5-base', 0.0)
    for m in model_names:
        assess(m, 0.0)
