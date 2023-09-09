import os

import pandas as pd
import numpy as np

DATA_PATH = 'age_gender.csv'
AGE_GROUP = [1, 6, 19, 30, 46, 66]


def assign_age_group(age, age_grp = AGE_GROUP):
	# assign labels to age groups
    if age_grp[0] <= age <= age_grp[1]-1:
        return 0
    elif age_grp[1] <= age <= age_grp[2]-1:
        return 1
    elif age_grp[2] <= age <= age_grp[3]-1:
        return 2
    elif age_grp[3] <= age <= age_grp[4]-1:
        return 3
    elif age_grp[4] <= age <= age_grp[5]-1:
        return 4
    else:
        return 5

def split_data(data):
	# split data based on train/val splits
	train_split = np.loadtxt('train.txt', dtype=int)
	val_split = np.loadtxt('val.txt', dtype=int)

	train_data = data.iloc[train_split]
	val_data = data.iloc[val_split]

	# create data dict
	train_dict = {'X':np.array(train_data['pixels'].tolist(), dtype=np.float32),
	              'age_labels': np.array(train_data['age'], dtype=np.int32),
	              'age_group_labels': np.array(train_data['age_group'], dtype=np.int32),
	              'ethnical_labels': np.array(train_data['ethnicity'], dtype=np.int32),
	              'gender_labels': np.array(train_data['gender'], dtype=np.int32)}
	val_dict = {'X':np.array(val_data['pixels'].tolist(), dtype=np.float32),
	            'age_labels': np.array(val_data['age'], dtype=np.int32),
	            'age_group_labels': np.array(val_data['age_group'], dtype=np.int32),
	            'ethnical_labels': np.array(val_data['ethnicity'], dtype=np.int32),
	            'gender_labels': np.array(val_data['gender'], dtype=np.int32)}

	return train_dict, val_dict


def load_data(DATA_PATH):
	# load prepare data to dict
	data = pd.read_csv(DATA_PATH)

	images = []
	for idx in range(len(data)):
	    images.append(np.array(data['pixels'][idx].split(), dtype=int).reshape(48, 48, 1))
	data['pixels'] = images

	# assign age group
	age_group = []
	for idx in range(len(data)):
	    age_label = assign_age_group(data['age'][idx])
	    age_group.append(age_label)
	data['age_group'] = age_group

	train_dict, val_dict = split_data(data)
	return train_dict, val_dict


def write_data_split():
	# write train/val split sets to .txt
	data = pd.read_csv(DATA_PATH)
	num_file = len(data)
	np.random.seed(44)
	n = np.random.permutation(num_file)
	train_split = np.sort(n[:-num_file//10])
	val_split = np.sort(n[-num_file//10:])
	with open('train.txt', 'w') as f:
	    for idx in train_split:
	        print('%d' % idx, file=f)
	with open('val.txt', 'w') as f:
	    for idx in val_split:
	        print('%d' % idx, file=f)