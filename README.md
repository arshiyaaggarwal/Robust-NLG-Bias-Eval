# Towards Robust NLG Bias Evaluation with Syntactically-diverse Prompts

## Description

This repository has the code for the paper Towards Robust NLG Bias Evaluation with Syntactically-diverse Prompts (Findings of EMNLP 2022). We present a **robust** and **rich** mechanism for bias evaluation in NLG (we use GPT-2 to generate output texts in our paper). We use automatically generated syntactically-diverse prompts using paraphrasing using which we generate outputs and analyze the generated texts in an aggregated and syntactically-segregated manner.

## Create a conda environment with all dependencies

```conda create -n robust_nlg_bias python```
```conda activate robust_nlg_bias```
```conda install pip```
```pip install -r requirements.txt```

## Generate output texts using automatically generated syntactically-diverse prompts

* Create a file for the fixed handcrafted prompts as in `data/prompts/fixed-prompts.txt`
* Use this file to get paraphrases for 100 different syntactic structures using [AESOP](https://github.com/PlusLabNLP/AESOP) directly. You should expect to get outputs as in `data/prompts/paraphrased/`
* Use these paraphrased prompts to get GPT-2 generated text outputs using the huggingface `transformers` library as in `scripts/generate_texts.py`. You should get generated outputs as in `data/generated-outputs/` without the regard scores.

## Run regard classifier

* Clone the [regard classifier](https://github.com/ewsheng/nlg-bias) repository

* Follow the instructions to use `run_classifier.py` to get the regard scores for all generated  *prompts*. The expected outputs are as asown in `data/generated-outputs` with the regard scores and texts.

## Use our robust NLG bias analysis method in your own project

* Run `scripts/aggregated_analysis.py` and `scripts/segregated_analysis.py` (adapted from [regard repository](https://github.com/ewsheng/nlg-bias)) to get the distribution curves, mean, std deviation, KL divergence, regard gaps, and other statistical analysis of the regard  scores distributed across various seeds, demographic groups and syntactic structures.
* To use this repository, make sure that the generated outputs for each seed value are in separate text files, each new line contains the regard score separated by the PLM generated output, and the order is: all (101 syntactic structures for each fixed prompt) texts for man first, followed by woman, straight person, gay person, black person and white person; then for the next fixed prompt type and so on...
* Make sure you specify the seed values and the output directory (for all the regard score labeled text files) correctly.
* Run the following commands to use our repository:

```python scripts/aggregated_analysis.py --seeds="21,12,3,30,9,18,36,45,54,27" --dir_name=data/generated-outputs/```
```python scripts/segregated_analysis.py --seeds="21,12,3,30,9,18,36,45,54,27" --dir_name=data/generated-outputs/```

## Citation

To appear in **Findings of EMNLP 2022**. Please cite our work if you decide to use it for your project.