import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from data_utils import load_data

DATA_PATH = 'age_gender.csv'
MODEL_PATH = 'checkpoint'
MODEL_NAME = 'ckpt-705.h5'

AGE_CLASSES = ['0-5', '6-18', '19-30', '31-45', '46-65', '66+']

def main():

	# load model and data
	model = tf.keras.models.load_model(os.path.join(MODEL_PATH, MODEL_NAME))
	_, val_dict = load_data(DATA_PATH)

	# prepare inputs
	val_input = np.concatenate([val_dict['X'], val_dict['X'], val_dict['X']], axis=3) # grayscle to RGB

	# evaluate model
	test_loss, test_acc = model.evaluate(x=val_input, y=val_dict['age_group_labels'], batch_size=32)
	print('test loss: %.3f, test acc: %.2f %%' % (test_loss, test_acc*100))


	# get random samples
	num_file = len(val_dict['age_group_labels'])
	random_list = np.random.permutation(num_file)

	n = 8 # number of samples to display
	samples = val_input[random_list[:n]]
	target = val_dict['age_labels'][random_list[:n]]

	# get predicitons from model
	outputs = model.predict(samples)
	results = np.argmax(outputs, axis=1)

	# display predicted samples
	fig = plt.figure(figsize=(12,8))
	for i in range(n):
	    fig.add_subplot(2, n//2, i+1)
	    plt.imshow(samples[i]/255)
	    string = 'Predicted age: %s \n real age: %s' % (AGE_CLASSES[results[i]], target[i])
	    plt.title(string)
	    plt.xticks([])
	    plt.yticks([])
	plt.show()


if __name__ == '__main__':
	main()