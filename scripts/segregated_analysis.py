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

def get_best_worst_regard_gaps(seeds, dir_name, mean_ss_vals, stddev_ss_vals):

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

    x = ['all', 'man', 'woman', 'straight', 'gay', 'black', 'white']
    means =  np.array(means)

    print("Printing indices of syntactic structures better than the mean for all:")
    better_ss = []
    better_std = []
    regard_gaps_best = []
    x=0
    for i in range(101):
        truth_vals = mean_ss_vals[:, i]> means
        if np.sum(truth_vals)>6:
            print(mean_ss_vals[:, i])
            print(stddev_ss_vals[:, i])
            print(i)
            better_ss.append(mean_ss_vals[:, i])
            regard_gaps_best.append([mean_ss_vals[1, i]-mean_ss_vals[2,i], mean_ss_vals[3, i]-mean_ss_vals[4,i], mean_ss_vals[6, i]-mean_ss_vals[5,i]])
            better_std.append(stddev_ss_vals[:, i])
            x+=1
        if x>5:
            break

    print("Printing indices of syntactic structures worse than the mean for all:")
    worse_ss = []
    worse_std =[]
    regard_gaps_worst = []
    x=0
    for i in range(101):
        truth_vals = mean_ss_vals[:, i]< means
        if np.sum(truth_vals)>6:
            print(mean_ss_vals[:, i])
            print(stddev_ss_vals[:, i])
            print(i)
            worse_ss.append(mean_ss_vals[:, i])
            regard_gaps_worst.append([mean_ss_vals[1, i]-mean_ss_vals[2,i], mean_ss_vals[3, i]-mean_ss_vals[4,i], mean_ss_vals[6, i]-mean_ss_vals[5,i]])
            worse_std.append(stddev_ss_vals[:, i])
            x+=1
        if x>5:
            break
    print("Regard gaps for the best syntactic structures are: ", regard_gaps_best)
    print("Regard gaps for the worst syntactic structures are: ", regard_gaps_worst)

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

    mean_ss = dict()
    variants = ['all', 'man', 'woman', 'straight-person', 'gay-person', 'black-person', 'white-person']
    for var in variants:
        mean_ss[var] = dict()
        for i in range(101):
            mean_ss[var][i] = []
    for seed in SEEDS:
        dir_name  = params.dir_name
        labeled_file = os.path.join(dir_name, 'out_seed_'+str(seed)+'_predictions.txt')
        sample_to_score = calc_sample_scores([labeled_file],
                                            first_period=True,
                                            score_type='bert')
        for ind, var in enumerate(variants):
            scores = {}
            for i in range(101):
                scores[str(i)] = []
            scores = OrderedDict(scores)
            # scores = OrderedDict({'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []})
            i=0
            for l, val in sample_to_score:
                key = str(i%101)
                if var == 'all':
                    scores[key].append(val)
                if var == 'man' and i%606 < 101:
                    scores[key].append(val)
                if var == 'woman' and i%606 >= 101 and i%606 < 202:
                    scores[key].append(val)
                if var == 'straight-person' and i%606 >= 202 and i%606 < 303:
                    scores[key].append(val)
                if var == 'gay-person' and i%606 >= 303 and i%606 < 404:
                    scores[key].append(val)
                if var == 'black-person' and i%606 >= 404 and i%606 < 505:
                    scores[key].append(val)
                if var == 'white-person' and i%606 >= 505:
                    scores[key].append(val)
                i+=1
            for k in range(101):
                key = str(k)
                mean_ss[var][k].extend(scores[key])
            scores = list(scores.values())
            if seed ==27:
                label_list = []
                for k in range(101):
                    label_list.append(str(k))
                ratios = plot_scores(scores, label_list, seed, var, ratio=True, mode=3)
    mean_ss_vals = np.zeros((7,101))
    stddev_ss_vals = np.zeros((7,101))
    for j, var in enumerate(variants):
        for i in range(101):
            mean_ss[var][i] = list(filter(lambda a: a != 2, mean_ss[var][i]))
            mean_ss_vals[j][i] = np.mean(np.array(mean_ss[var][i]))
            stddev_ss_vals[j][i] = np.std(np.array(mean_ss[var][i]))

    ## you can print the mean_ss_all and stddev_ss_all to get statistics for all demographics and ss
    ss_argsorted = np.argsort(mean_ss_vals, axis = 1)[::-1]
    ss_top = ss_argsorted[:, :20]
    ss_bottom = ss_argsorted[:, -20:]
    ss_top_all = ss_top[0,:]
    for i in range(1,7, 1):
        ss_top_all = np.intersect1d(ss_top_all, ss_top[i, :])
    print("The intersection of the best syntactic structures (indices) for all demographics: ", ss_top_all)

    ss_bottom_all = ss_bottom[0,:]
    for i in range(1,7, 1):
        ss_bottom_all = np.intersect1d(ss_bottom_all, ss_top[i, :])
    print("The intersection of the worst syntactic structures (indices) for all demographics: ", ss_bottom_all)

    print("The union of the best syntactic structures (indices) for all demographics: ", ss_top)
    print("The union of the worst syntactic structures (indices) for all demographics: ", ss_bottom)

    get_best_worst_regard_gaps(SEEDS, dir_name, mean_ss_vals, stddev_ss_vals)


if __name__ == '__main__':
	main()
