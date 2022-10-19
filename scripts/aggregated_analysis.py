import argparse
import numpy as np
import os
import seaborn as sns
from random import seed
from random import random

from collections import Counter, OrderedDict

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from analyze_util import calc_sample_scores, plot_scores, kl_divergence

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--seeds',
	                    required=True,
						type=str,
						default="21,12,3,30,9,18,36,45,54,27",
	                    help='List of seed values you wish to use for the analysis as a comma-separated string.')
	parser.add_argument('--dir_name',
	                    required=True,
	                    default='.',
	                    help='Path to directory where the regard score labeled output text files are stored.')
	params = parser.parse_args()

	print('params', params)

	SEEDS = [int(item) for item in params.seeds.split(',')]

	hists = [[],[],[],[],[],[],[]]
    categories = [[],[],[],[],[],[],[]]
    for seed in SEEDS:
        dir_name  = params.dir_name
		labeled_file = os.path.join(dir_name, 'out_seed_'+str(seed)+'_predictions.txt')
        sample_to_score = calc_sample_scores([labeled_file],
                                            first_period=True,
                                            score_type='bert')

        variants = ['all', 'man', 'woman', 'straight-person', 'gay-person', 'black-person', 'white-person']
        for ind, var in enumerate(variants):
            scores = {}
            for i in range(101):
                scores[str(i)] = []
            scores = OrderedDict(scores)
            # scores = OrderedDict({'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []})
            i=0
            for l, val in sample_to_score:
                key = str(i%101)
                ss_i = i%101
                if var == 'all':
                    categories[0].append(val)
                    scores[key].append(val)
                if var == 'man' and i%606 < 101:
                    categories[1].append(val)
                    scores[key].append(val)
                if var == 'woman' and i%606 >= 101 and i%606 < 202:
                    categories[2].append(val)
                    scores[key].append(val)
                if var == 'straight-person' and i%606 >= 202 and i%606 < 303:
                    categories[3].append(val)
                    scores[key].append(val)
                if var == 'gay-person' and i%606 >= 303 and i%606 < 404:
                    categories[4].append(val)
                    scores[key].append(val)
                if var == 'black-person' and i%606 >= 404 and i%606 < 505:
                    categories[5].append(val)
                    scores[key].append(val)
                if var == 'white-person' and i%606 >= 505:
                    categories[6].append(val)
                    scores[key].append(val)
                i+=1
            scores = list(scores.values())
            label_list = []
            for k in range(101):
                label_list.append(str(k))
            ratios = plot_scores(scores, label_list, seed, var, ratio=True)
            hists[ind].extend(ratios)
    means = []
    stddevs = []
    variants = ['all', 'man', 'woman', 'straight-person', 'gay-person', 'black-person', 'white-person']
    kl_div = np.zeros((7,7))
    for j, var in enumerate(variants):
        categories[j] = list(filter(lambda a: a != 2, categories[j]))
        means.append(np.mean(np.array(categories[j])))
        stddevs.append(np.std(np.array(categories[j])))
        b = Counter(categories[j])
        dist1 = [b[-1]/float(len(categories[j])), b[0]/float(len(categories[j])), b[1]/float(len(categories[j]))] if len(categories[j])>0 else [0,0,0]
        print(dist1)
        for a in range(len(variants)):
            dist2  = list(filter(lambda n: n != 2, categories[a]))
            b = Counter(dist2)
            dist2 = [b[-1]/float(len(dist2)), b[0]/float(len(dist2)), b[1]/float(len(dist2))] if len(dist2)>0 else [0,0,0]
            kl_div[j, a] = kl_divergence(np.array(dist1), np.array(dist2))
    regard_gaps = [means[1]-means[2], means[3]-means[4], means[6]-means[5]]

    x = ['all', 'man', 'woman', 'straight', 'gay', 'black', 'white']
    means =  np.array(means)
    np.set_printoptions(precision=4, suppress=True)
    # print(means)
    stddevs  = np.array(stddevs)
    # print(stddevs)
    plt.plot(x, means, color='blue')
    plt.fill_between(x, means+stddevs, means-stddevs, facecolor='blue', alpha=0.2)
    plt.xlabel('Demographic Groups')
    plt.ylabel('Mean and Std Dev of the regard categories')
    plt.title('Distribution statistics for various demographics')
    plt.savefig('mean_stddev_demo_.png')
    plt.clf()

    # print(kl_div)
    ax = sns.heatmap(kl_div, annot=True, fmt=".2f", cmap="YlGnBu")
    ax = sns.heatmap(kl_div, cmap="YlGnBu")
    plt.title("KL Divergence for the various demographics")
    plt.savefig("kl_div_demo_.png")
    plt.clf()

    print("Regard gaps overall: ", regard_gaps)

if __name__ == '__main__':
	main()
