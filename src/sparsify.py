import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelWithLMHead, GPT2Tokenizer, GPT2Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         # 1 input image channel, 6 output channels, 3x3 square conv kernel
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, int(x.nelement() / x.shape[0]))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    if len(flattened) == 0:
        res = [el]
    else:
        res = []
    for c in flattened:
        res += c
    return res


def model_prune(model, amount=0.1):
    modules = flatten(model)

    parameters_to_prune = [(m, n) for m in modules for n, p in m.named_parameters() if n.endswith('weight')]
    # print(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    for m, n in parameters_to_prune:
        try:
            prune.remove(m, n)
        except Exception as e:
            print(e)

    return model


def sparsify(model, name):
    for s in [.0, .1, .5, .9, .95, .99]:
        print(f"Pruning model: {name} with {s} reduction...")
        if s:
            model = model_prune(model, s)
        model.config.pad_token_id = model.config.eos_token_id
        model.save_pretrained(f'../local-models/{name}-{s}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # GPT2-XLarge - 1.5B params
    model_name = sys.argv[1]
    if model_name == 'gpt2' or model_name == 'gpt2-xl':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'microsoft/deberta-v2-xxlarge' or 'microsoft/deberta-v2-xlarge':
        model = AutoModel.from_pretrained(model_name)
    elif model_name == 't5-3b' or model_name == 't5-base':
        model = AutoModelWithLMHead.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f'../local-models/{model_name}-tokenizer')
    sparsify(model, model_name)

    print("Parameter Count:", count_parameters(model))


