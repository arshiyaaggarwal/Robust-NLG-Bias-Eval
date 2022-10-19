"""Script to analyze model's generated outputs. Adapted from the repository https://github.com/ewsheng/nlg-bias """

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


def calc_sample_scores(files, first_period=True, score_type='bert'):
	"""Calculate/format scores for samples."""
	scores = []
	lines = []

	for fi_idx, fi in enumerate(files):
		with open(fi, 'r') as f:
			for line in f:
				line = line.strip()
				sample = line.split('\t')[-1]
				if first_period:
					# Cut off the line when we see the first period.
					if '.' in sample:
						period_idx = sample.index('.')
					else:
						period_idx = len(sample)
					sample_end = min(period_idx + 1, len(sample))
				else:
					sample_end = len(sample)
				sample = sample[:sample_end]
				lines.append(sample)

	if score_type == 'bert':
		for fi in files:  # Analyze the classifier-labeled samples.
			with open(fi) as f:
				for line in f:
					line = line.strip()
					line_split = line.split('\t')
					score = int(line_split[0])
					scores.append(score)
	else:
		raise NotImplementedError('score_type = textblob, vader, bert')

	assert(len(scores) == len(lines))

	return list(zip(lines, scores))


def plot_scores(score_list, label_list, seed, demo, ratio=False, ax=None, row=None, col=None, mode=None):
	"""Plot sentiment"""
	# if seed==27:
	# 	plt.figure(demo)
	vals = []
	for score_idx in range(len(score_list)):
		scores = score_list[score_idx]
		score_counts = Counter()
		for s in scores:
			if s ==1:
				score_counts['+'] += 1
			elif s == -1:
				score_counts['-'] += 1
			elif s== 0:
				score_counts['0'] += 1
		if ratio:
			if len(scores):
				score_len = float(len(scores))
				score_counts['+'] /= score_len
				score_counts['-'] /= score_len
				score_counts['0'] /= score_len
		ordered_score_counts = [round(score_counts['-'], 3), round(score_counts['0'], 3),
		                            round(score_counts['+'], 3)]
		val = ordered_score_counts[2] + ordered_score_counts[1] - ordered_score_counts[0]
		vals.append(val)
	return vals

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_vals(scores):
	score_counts = Counter()
	vals = []
	for s in scores:
		if s ==1:
			score_counts['+'] += 1
		elif s == -1:
			score_counts['-'] += 1
		elif s== 0:
			score_counts['0'] += 1
	if len(scores):
		score_len = float(len(scores))
		score_counts['+'] /= score_len
		score_counts['-'] /= score_len
		score_counts['0'] /= score_len
		ordered_score_counts = [round(score_counts['-'], 3), round(score_counts['0'], 3),
		                            round(score_counts['+'], 3)]
		val = ordered_score_counts[2] + ordered_score_counts[1] - ordered_score_counts[0]
		vals.append(val)
	return vals