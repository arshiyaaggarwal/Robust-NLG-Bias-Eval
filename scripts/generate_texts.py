from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

#get large GPT2 tokenizer and GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
GPT2 = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
GPT2 = GPT2.cuda()

MAX_LEN = 40
SEEDS = [21, 12, 3, 30, 9, 18, 36, 45, 54, 27] ## get this using argparse
### combine all prompts
with open('data/prompts/fixed-prompts.txt') as file:
    lines = file.readlines()
    prompts = [line.rstrip() for line in lines]
print("The fixed prompts being used are, ", prompts)
with open('data/prompts/paraphrased/paranmt-prompt-50_sep_extract') as file:
    lines = file.readlines()
    paraphrases_paranmt = [line.rstrip() for line in lines]
with open('data/prompts/paraphrased/qqppos-prompt-50_sep_extract') as file:
    lines = file.readlines()
    paraphrases_qqppos = [line.rstrip() for line in lines]

i=0
final_prompts = []
for prompt in prompts:
  final_prompts.append(prompt)
  for j in range(i,i+50):
    final_prompts.append(paraphrases_paranmt[j])
    final_prompts.append(paraphrases_qqppos[j])
  i+=50

print("Total prompts being used for GPT-2 generation are: ", len(final_prompts))

### check the unique suntactic structure types
with open('data/prompts/paraphrased/promp_paranmt_50.source') as file:
    lines = file.readlines()
    paraphrases_paranmt = [line.rstrip() for line in lines]
with open('data/prompts/paraphrased/promp_qqppos_50.source') as file:
    lines = file.readlines()
    paraphrases_qqppos = [line.rstrip() for line in lines]

_qqppos = []
_paranmt = []
for j in range(50):
    _qqppos.extend([paraphrases_qqppos[i].split("<sep>")[2].strip() for i in range(j,len(paraphrases_qqppos),50)])
    _paranmt.extend([paraphrases_paranmt[i].split("<sep>")[2].strip() for i in range(j,len(paraphrases_paranmt),50)])

_qqppos.extend(_paranmt)

combined = np.array(_qqppos)
# print(combined)
print("The shape of the unique syntactic structures in all the prompts is: ", len(np.unique(combined)))

## generate outputs
for seed in SEEDS:
    torch.cuda.manual_seed(seed)
    prompt_outputs = []
    for input_sequence in final_prompts:
        input_ids = tokenizer.encode(input_sequence, return_tensors='pt')
        output = GPT2.generate(
            input_ids.cuda(),
            do_sample=True, 
            max_length = MAX_LEN, 
            top_k=40,
            temperature=0.7, 
            num_return_sequences = 1, 
            early_stopping = True
        )
        for i, out in enumerate(output):
            prompt_outputs.append(tokenizer.decode(out, skip_special_tokens=True))
    print("Done seed = "+str(seed)+". Length of generated outputs: "+str(len(prompt_outputs)))
    # print(prompt_outputs)
    with open('outputs_top_k_40_seed_'+str(seed)+'.txt', 'w') as f:
        for output in prompt_outputs:
            f.write("[Out]%s\n\n" % output)

"""### ***Get XYZ tsv files for all generated outputs for regard evaluation***"""
for seed in SEEDS:
    filename = "/content/outputs_seed_"+str(seed)+".txt"
    outputs = []
    with open(filename) as file:
        for line in file:
            if line.startswith("[Out]"):
                outputs.append(line)
            else:
                outputs[-1] = outputs[-1] + line
    for i in range(len(outputs)):
        outputs[i] = outputs[i].replace("\n", " ")
    for i in range(len(outputs)):
        outputs[i] = outputs[i].replace("[Out]", "")

    demographics = ["the man", "a man", "this man","the guy", "a guy", "this guy", "the woman", "a woman","this woman", "the straight person", "a straight person", "being straight", "straight person", "a straight guy", "straight people", "a straight man", "the gay person", " a gay guy", "gay people", "a gay person", "being gay", "gay person", "a gay man", "the black person", "black person", "the black guy","a black guy", "a black man", "black people", "the white person", "the white guy", "the white man", "a white person", "white people"]
    xyz_outputs = []
    for i in range(len(outputs)):
        for demo in demographics:
            if demo in outputs[i][:40]:
                xyz_outputs.append(outputs[i].replace(demo, "XYZ"))
    for i in range(len(xyz_outputs)):
        index = min(80,len(xyz_outputs[i]))
        if '.' in xyz_outputs[i][:index]:
            xyz_outputs[i] = xyz_outputs[i].replace('.', "")
    with open('out_seed_'+str(seed)+'.tsv', 'w') as f:
        for output in outputs:
            f.write("%s\n" % output)

    with open('out_seed_'+str(seed)+'.tsv.XYZ', 'w') as f:
        for output in xyz_outputs:
            f.write("%s\n" % output)