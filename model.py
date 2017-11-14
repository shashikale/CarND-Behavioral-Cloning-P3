from data_processing import get_images, adjust_properties_per_transform, adjust_angle_per_camera
from keras.models import Model
from keras.layers import Cropping2D, Lambda, merge, BatchNormalization, Input
from keras.layers.core import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


def conv2d_bn(input_x, filters, rows, cols, border_mode='same', strides=(1, 1)):
    """ Combine Convolution2D and BatchNormalization
    """
    input_x = Convolution2D(filters, rows, cols,
                            subsample=strides,
                            activation='relu',
                            border_mode=border_mode)(input_x)
    input_x = BatchNormalization()(input_x)
    return input_x


def small_model():
    img_input = Input(shape=(160, 320, 3))
    x = Lambda(lambda x: x / 255.0 - 0.5)(img_input)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(x)
    x = conv2d_bn(x, 24, 5, 5, strides=(2, 2))
    x = conv2d_bn(x, 36, 5, 5, strides=(2, 2))
    x = conv2d_bn(x, 64, 3, 3)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(50)(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Dense(1)(x)
    return Model(img_input, x)


def nvidia_model():
    img_input = Input(shape=(160, 320, 3))
    x = Lambda(lambda x: x / 255.0 - 0.5)(img_input)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(x)
    x = conv2d_bn(x, 24, 5, 5, strides=(2, 2))
    x = conv2d_bn(x, 36, 5, 5, strides=(2, 2))
    x = conv2d_bn(x, 48, 5, 5, strides=(2, 2))
    x = conv2d_bn(x, 64, 3, 3)
    x = conv2d_bn(x, 64, 3, 3)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)
    x = Dense(1)(x)
    return Model(img_input, x)


def inception_model():
    img_input = Input(shape=(160, 320, 3))
    x = Lambda(lambda x: x / 255.0 - 0.5)(img_input)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(x)
    x = conv2d_bn(x, 8, 3, 3)
    x = conv2d_bn(x, 16, 3, 3)
    x = MaxPooling2D((3, 3), strides=(1, 1))(x)

    # Inception Module 1
    im1_1x1 = conv2d_bn(x, 8, 1, 1)

    im1_5x5 = conv2d_bn(x, 4, 1, 1)
    im1_5x5 = conv2d_bn(im1_5x5, 8, 5, 5)

    im1_3x3 = conv2d_bn(x, 4, 1, 1)
    im1_3x3 = conv2d_bn(im1_3x3, 8, 3, 3)
    im1_3x3 = conv2d_bn(im1_3x3, 8, 3, 3)

    im1_max_p = MaxPooling2D((3, 3), strides=(1,1))(x)
    im1_max_p = conv2d_bn(im1_max_p, 8, 1, 1)
    im1_max_p = ZeroPadding2D(padding=(1,1))(im1_max_p)
    
    x = merge([im1_1x1, im1_5x5, im1_3x3, im1_max_p],
              mode='concat')

    # Inception Module 2
    im2_1x1 = conv2d_bn(x, 8, 1, 1)

    im2_5x5 = conv2d_bn(x, 4, 1, 1)
    im2_5x5 = conv2d_bn(im2_5x5, 8, 5, 5)

    im2_3x3 = conv2d_bn(x, 4, 1, 1)
    im2_3x3 = conv2d_bn(im2_3x3, 8, 3, 3)
    im2_3x3 = conv2d_bn(im2_3x3, 8, 3, 3)

    im2_max_p = MaxPooling2D((3, 3), strides=(1,1))(x)
    im2_max_p = conv2d_bn(im2_max_p, 8, 1, 1)
    im2_max_p = ZeroPadding2D(padding=(1,1))(im2_max_p)
    
    x = merge([im2_1x1, im2_5x5, im2_3x3, im2_max_p],
              mode='concat')
    
    # Fully Connected
    x = AveragePooling2D((8, 8), strides=(8, 8))(x)
    x = Dropout(0.5)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1, name='predictions')(x)
    return Model(img_input, x)


def get_batch_properties(images, image_getter):
    """ Order the images and the stuff we are going to
        do to them in a dictionary for easy processing
        images: a list of integers
        image_getter: a function, receives an integer, 
            returns image data and what to do to the image
    """
    r = {}
    for i in images:
        image_path, features, transform, camera = image_getter(i)
        if image_path not in r:
            r[image_path] = [(transform, features, camera)]
        else:
            r[image_path].append((transform, features, camera))
    return r


def get_images_generator(images, image_getter, BATCH_SIZE, n_samples, name=None):
    """ The generator of data
        returns: Tuple(
            numpy array of BATCH_SIZE with images in it,
            numpy array of steering angles
            )
    """
    # Numpy arrays as large as expected
    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 160
    CHANNELS = 3
    batch_images = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), np.uint8)
    batch_features = np.zeros(BATCH_SIZE)
    begin_batch = 0
    while True:
        i = 0
        batch_dictionary = get_batch_properties(images[begin_batch:begin_batch+BATCH_SIZE],
                                                image_getter)
        for img_path in batch_dictionary.keys():
            abs_path = os.path.abspath(img_path)
            img = cv2.imread(abs_path)
            for transform, features, camera in batch_dictionary[img_path]:
                features_adjusted, camera_adjusted = adjust_properties_per_transform(features,
                                                                                     camera,
                                                                                     transform)
                payload = transform(img)
                steer = adjust_angle_per_camera(features_adjusted, camera_adjusted)
                batch_position = i % BATCH_SIZE
                batch_images[batch_position] = payload
                batch_features[batch_position] = steer[0]
                i += 1
        begin_batch = begin_batch + BATCH_SIZE if begin_batch + BATCH_SIZE < n_samples else 0
        yield batch_images, batch_features


def properties(total_images):
    """ We want our set of images to be divisible by the
        epochs
    """
    batch_size = 128
    remainder = total_images % batch_size
    images = total_images - remainder
    return batch_size, images # Prevent repeated samples



def main(get_model, EPOCHS, plot_loss):
    image_index_db, image_getter, _ = get_images('./data')
    x_train, x_test = train_test_split(image_index_db, test_size=0.3)
    x_train = shuffle(x_train)
    model = get_model()
    model.compile(optimizer='adam',
                  loss='mse')
    # MAGIC NUMBERS
    BATCH_SIZE, SAMPLES_PER_EPOCH = properties(len(x_train))
    VALIDATION_BATCH_SIZE, VALIDATION_SAMPLES_PER_EPOCH = properties(len(x_test))
    names = ['IMAGES IN TRAINING', 'IMAGES IN VALIDATION SET',
             'BATCH SIZE', 'SAMPLES PER EPOCH', 'VALIDATION BATCH SIZE',
             'VALIDATION SAMPLES PER EPOCH']
    magic_numbers = [len(x_train), len(x_test),
                     BATCH_SIZE, SAMPLES_PER_EPOCH, VALIDATION_BATCH_SIZE,
                     VALIDATION_SAMPLES_PER_EPOCH]
    for name, value in zip(names, magic_numbers):
        print("{0:<30s} {1}".format(name, value))
    training_generator = get_images_generator(x_train, image_getter, BATCH_SIZE, SAMPLES_PER_EPOCH, 'training')
    validation_generator = get_images_generator(x_test, image_getter, VALIDATION_BATCH_SIZE, VALIDATION_SAMPLES_PER_EPOCH, 'validation')
    history_object = model.fit_generator(training_generator,
                                  samples_per_epoch=SAMPLES_PER_EPOCH,
                                  verbose=1,
                                  validation_data=validation_generator,
                                  nb_val_samples=VALIDATION_SAMPLES_PER_EPOCH,
                                  nb_epoch=EPOCHS)
    model_name = './' + get_model.__name__
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    model.save(model_name + '/model.h5')
    with open(model_name + '/model.json', 'w') as json_file:
        json_string = model.to_json()
        json.dump(json_string, json_file)

    print(history_object.history.keys())
    if(plot_loss):
    	### plot the training and validation loss for each epoch
    	plt.plot(history_object.history['loss'])
    	plt.plot(history_object.history['val_loss'])
    	plt.title('model mean squared error loss')
    	plt.ylabel('mean squared error loss')
    	plt.xlabel('epoch')
    	plt.legend(['training set', 'validation set'], loc='upper right')
    	plt.show()
    import gc
    gc.collect()


def test_generator():
    image_index_db, image_getter, _ = get_images('./data')
    shuffled_images = image_index_db
    #shuffled_images = shuffle(image_index_db)
    BATCH_SIZE, SAMPLES_PER_EPOCH = properties(len(shuffled_images))
    names = ['IMAGES IN TRAINING', 'BATCH SIZE', 'SAMPLES PER EPOCH']
    magic_numbers = [len(shuffled_images), BATCH_SIZE, SAMPLES_PER_EPOCH]
    for name, value in zip(names, magic_numbers):
        print("{0:<30s} {1}".format(name, value))
    sauce_generator = get_images_generator(shuffled_images, image_getter, BATCH_SIZE, SAMPLES_PER_EPOCH)
    for i, p in enumerate(sauce_generator):
        rows = 5
        cols = 30 // rows
        fig, axes = plt.subplots(rows, cols)
        for j, (img, feature) in enumerate(zip(p[0][:30], p[1][:30])):
            ax = axes[j // cols, j % cols]
            ax.imshow(img)
            s = 'Steer %.5f' %feature
            label = '{steer}'.format(steer=s)
            ax.set_title(label)
            ax.axis('off')
        if i > 5:
            break
        plt.subplots_adjust(hspace=0.5)
        plt.show() 

    for name, value in zip(names, magic_numbers):
        print("{0:<30s} {1}".format(name, value))
    
if __name__ == '__main__':
    #test_generator()
    #main(inception_model, 2, False)
    #main(nvidia_model, 3, True)
    main(small_model, 5, True)

