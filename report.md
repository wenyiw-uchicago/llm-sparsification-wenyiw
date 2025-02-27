# Sparsification

## Selected Models

- Encoder-only: DeBERTa-v2 XXL (1.5B)
- Decoder-only: GPT2-XL (1.5B)
- Encoder-Decoder: T5-3B (3B)

## Sparsity Assessment
The following histograms illustrate the distribution of the parameters overall in the selected models.
The x-axis is the value of the parameters.
### GPT2 Parameter Distribution (Overall)

![gpt2 hist](src/graphs/gpt2_hist.png)

### Deberta-v2 Parameter Distribution (Overall)

![Deberta-v2 Hist](src/graphs/deberta_hist.png)

### T5 Parameter Distribution (Overall)

![T5 Hist](src/graphs/t5_hist.png)

As for per-layer distribution, I printed out the fraction of each layer under `assessments` folder, the report dump file is generated by the `assess.py` which layout detailed sparsity structure in each layer.

## Model Pruning
The script used to prune the models is in the `src/sparsify.py`. I also saved the tweaked tokenizers locally as some tasks 
from HF requires to do so. As recommended by the Pytorch Prune guide, I used global pruning by zeroing out the ones 
with the lowest L1-norm (`prune_method=L1Unstructured`). This method prunes the model all at once, by removing the lowest
_p_ percentage of connections across the whole model.

## Benchmarking the Sparsified Models
I selected 5 sets of benchmarks.
- GLUE benchmarks are used by both GPT2 and Deberta-v2.
- Casual Language Modeling (CLM) for GPT2 from HF's language-modeling where they used that for fine-tuning the models.
- Masked Language Modeling (MLM) for Deberta-v2, same as CLM
- SQuaD_v2 QA benchmarks for T5, however, only the number of samples shown in the evaluation results, we ignore the results.
- translation from HF for T5
The code for running the benchmarks are the `src/*.sh` files. The example commands can be found in the `run_benchmarks.ipynb`
### GPT2 Performance

![GPT2 perf](src/graphs/gpt2-perf.png)

I evaluated the accuracy for both benchmarks and for the CLM tasks, more pruning results in visible performance decrease.
As for the GLUE benchmark, the performance GPT2 does not change with pruning.
### Deberta-v2 Performance

![deberta-perf](src/graphs/deberta-perf.png)

Deberta's performance on GLUE, on the other hand, increased a bit after pruning. But it seems to fail the evaluation from MLM.

### T5 Performance

![t5-perf](src/graphs/t5-perf.png)

T5's translation performance drops abruptly as we increase pruning ratio.

## Runtime Comparison
Different from the benchmark graphs, runtime comparison is grouped by benchmarks instead of models.

For GLUE, it is obvious that in our example settings, which uses GPT2 and Deberta-v2-xlarge, the deberta has around eight
times the parameters count compared to gpt2.

For other benchmarks, the runtime simply reduces as the pruning ratio increases but if we look closely, the runtime decrease 
is trivial to the ratio of pruning done on the model, the PyTorch pruning will not reduce the model size, but it claims it will reduce
inference time which we can tell a bit from the graphs.
The exception occurs from the translation evaluation runtime for T5.

### GLUE Runtime (GPT2 and Deberta-v2)

![glue-rt](src/graphs/GLUE-runtime.png)

### CLM Runtime (GPT2)

![clm-rt](src/graphs/CLM-GPT2-runtime.png)

### MLM Runtime (Deberta-v2)

![mlm-rt](src/graphs/MLM-Deberta-runtime.png)

### Translation Runtime (T5)

![t-rt](src/graphs/Translation-T5-runtime.png)
### ~~Question Answering (T5)~~
Not available, the evaluation results file only contains sample count.

## Challenges of sparsifications on LLMs
One of the most obvious challenges of sparsifications on LLMs are the what, how, when, and how often to prune the model.
Certainly, we have developed many heuristic ways on what to prune based on the scores of each weight, we also attained empirical
observations or rule of thumb towards the answer of how often to prune to model, but these are not enough and most importantly,
cannot be explained seriously.

Moreover, current pruning during training strategies are related to 'regularization and ideas of dropout/reinforcing 
distributed representations'(From L5 slides). And training these sparse models from scratch requires collaborative effort 
to build system from hardware up to software stack, we are still at the very start of trying to make this happen, 
from hardware perspective, the memory organization can be a challenge as to make it easy to program but also keep its 
capability of fast accessing based on the dataflow.