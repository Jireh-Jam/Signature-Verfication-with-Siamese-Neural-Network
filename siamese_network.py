from __future__ import absolute_import
from __future__ import print_function
import os
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import configuration
from model import SiameseNetwork
from miscellaneous import euclidean_distance, contrastive_loss, generate_batch, accuracy
import numpy as np

dataset_path = "/home/rafiqul/Bengali/"
all_sub_directory = next(os.walk(dataset_path))[1] #immediate child sub directory
all_sub_directory.sort()
genuine, fake = [], []
for each_sub_directory in all_sub_directory:
    images = os.listdir(dataset_path + each_sub_directory)
    images.sort()
    images = [dataset_path + each_sub_directory + '/' + x for x in images]
    fake.append(images[:30])
    genuine.append(images[30:])


genuine_training_data, genuine_validation_data, genuine_test_data = genuine[:80], genuine[80:90], genuine[90:]
fake_training_data, fake_validation_data, fake_test_data = fake[:80], fake[80:90], fake[90:]
del genuine, fake
image_height, image_width = configuration.configure_parameter["image_height"],configuration.configure_parameter["image_width"]

batch_size = configuration.configure_parameter["batch_size"]
num_train_samples = 57600
num_val_samples = num_test_samples = 11520

input_shape= image_height, image_width,1
get_network = SiameseNetwork(input_shape)
input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)
feature_a = get_network(input_1)
feature_b = get_network(input_2)
feature_distance=Lambda(euclidean_distance)([feature_a, feature_b])
model = Model([input_1, input_2], feature_distance)
adam=Adam(0.001, decay=2.5e-4)
model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])

callbacks = [
    EarlyStopping(patience=12, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('/home/rafiqul/SiameseNetwork/siamese-{epoch:05d}.h5', verbose=1, save_weights_only=True)
]

batch_size=configuration.configure_parameter["batch_size"]
epoch=configuration.configure_parameter["number_of_epoch"]

results = model.fit_generator(generate_batch(genuine_training_data, fake_training_data, batch_size),
                              steps_per_epoch = num_train_samples//batch_size,
                              epochs = epoch,
                              validation_data = generate_batch(genuine_validation_data, fake_validation_data, batch_size),
                              validation_steps = num_val_samples//batch_size,
                              callbacks = callbacks)

