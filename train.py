import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

from data_utils import load_data


DATA_PATH = 'age_gender.csv'
AGE_GROUP = [1, 6, 19, 30, 46, 66]


def build_model():
	inputs = keras.Input(shape=(48, 48, 3))

	# data augmentation
	augmentation_layer = keras.Sequential([
	    layers.RandomFlip('horizontal'),
	    # layers.RandomBrightness(0.2),
	    layers.RandomTranslation(0.2, 0.2, fill_mode='constant', fill_value=0.0),
	    layers.RandomContrast(0.3),
	    layers.RandomRotation(0.1, fill_mode='constant', fill_value=0.0),
	    layers.Rescaling(scale=1/127.5, offset=-1),
	    layers.Resizing(128, 128, interpolation='bilinear')
	    ])

	# load EfficientNet model [cite: https://arxiv.org/abs/2104.00298]
	base_model = tf.keras.applications.EfficientNetV2S(
	    input_shape=(128,128,3),
	    # alpha=1.0,
	    include_top=False,
	    include_preprocessing=False,
	    weights="imagenet",
	    input_tensor=None,
	    pooling=None,
	)

	x = augmentation_layer(inputs)
	x = base_model(x)

	# new classifier layers
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(256, activation='relu') (x)
	x = layers.Dropout(0.3) (x)
	output = layers.Dense(6, activation='softmax', name='AGE_GROUP_out') (x)

	model = keras.Model(inputs=[inputs], outputs=[output])
	# print(model.summary())
	return model


def main():

	train_dict, val_dict = load_data(DATA_PATH)

	model = build_model()
	print(model.summary())

	lr_schedule = keras.optimizers.schedules.ExponentialDecay(
	    initial_learning_rate=0.001,
	    decay_steps=5000,
	    decay_rate=0.8)

	model.compile(loss=['sparse_categorical_crossentropy'],
		optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
		metrics=['accuracy'])

	# callback to save the bset model
	save_path = os.path.join('checkpoint', 'checkpoint.h5')
	ckpt_callback = ModelCheckpoint(save_path,
		monitor='val_accuracy',
		mode='max',
		save_best_only=True,verbose=1)

	train_input = np.concatenate([train_dict['X'], train_dict['X'], train_dict['X']], axis=3) # grayscle to RGB
	val_input = np.concatenate([val_dict['X'], val_dict['X'], val_dict['X']], axis=3) # grayscle to RGB

	history = model.fit(x=train_input, y=train_dict['age_group_labels'],
		batch_size=32, epochs=30, shuffle=True,
		validation_data=(val_input, val_dict['age_group_labels']),
		validation_freq=1, callbacks=ckpt_callback)


if __name__ == '__main__':
    main()