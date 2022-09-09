import os
from datetime import datetime
from enum import Enum
from pathlib import PosixPath
import random
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from IPython.display import clear_output
# from loguru import logger
from tensorflow.python.data.ops.dataset_ops import (BatchDataset, Dataset,
                                                    PrefetchDataset)
from tensorflow.python.keras import Model, Sequential, layers, losses

from Augment import Augment
from DiceLoss import DiceLoss
from DiceMetric import DiceCoefficient
from ResNet import ResNet
from UNet import UNet
from EfficientNet import EfficientNet
from VGGNet import VGGNet

sm.set_framework('tf.keras')
os.environ['TF_CUDNN_DETERMINISTIC'] = 'false'
os.environ['TF_DETERMINISTIC_OPS'] = 'false'
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = 'true'

# tf.config.experimental.enable_op_determinism()


class InputMode(Enum):
    ARRAY = 1
    DIRECTORY = 2
    FILE = 3



class Manager:
    

    optimizer = 'adam'
    # loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    # metrics = ['accuracy']

    loss = DiceLoss()
    metrics = [DiceCoefficient()]

    batch_size = 32
    buffer_size = 1000

    width = 128
    height = 128

    validation_split = 0.2

    seed = 31415

    num_classes = 2
    class_names = []

    train_dataset = None
    validation_dataset = None
    test_dataset = None

    dataset_size = 0

    dropout = 0.2

    AUTOTUNE = tf.data.AUTOTUNE

    history = None
    training_id = 0
    base_path = '/'
    loads_model = False

    def __init__(
            self,
            dataset = None,
            train_dataset = None,
            validation_dataset = None,
            test_dataset = None,
            optimizer = None, 
            loss = None, 
            metrics = None, 
            input_mode = None, 
            images_path = None,
            masks_path = None,
            validation_split = None, 
            seed = None, 
            batch_size = None, 
            width = None, 
            height = None,
            num_classes = None, 
            dropout = None,
            training_id = None,
            loads_model = None,
            base_path = None
        ):

        self.seed = seed if seed else self.seed
        self.set_seed(self.seed)


        
        self.optimizer = optimizer if optimizer else self.optimizer
        self.loss = loss if loss else self.loss
        self.metrics = metrics if metrics else self.metrics

        self.validation_split = validation_split if validation_split else self.validation_split
        self.batch_size = batch_size if batch_size else self.batch_size

        self.num_classes = num_classes if num_classes else self.num_classes

        self.input_mode = input_mode if input_mode else InputMode.ARRAY

        self.width = width if width else self.width
        self.height = height if height else self.height

        self.dropout = dropout if dropout else self.dropout

        self.training_id = training_id if training_id else self.training_id

        self.loads_model = loads_model if loads_model else self.loads_model

        self.base_path = base_path if base_path else self.base_path

        os.system('rm -rf history')
        os.system('rm -rf logs')

        os.system('mkdir -p plots')
        os.system('mkdir -p logs')
        os.system('mkdir -p checkpoints')
        os.system('mkdir -p saved_models')
        os.system('mkdir -p history')

        if not self.loads_model:

            if self.input_mode == InputMode.ARRAY:
                self.train_dataset, self.validation_dataset, self.test_dataset = train_dataset, validation_dataset, test_dataset
                # self.train_dataset, self.validation_dataset, self.test_dataset = self.load_from_dataset(dataset)

            if self.input_mode == InputMode.DIRECTORY:
                image_count = len(list(images_path.glob('*.jpg')))
                self.dataset_size = image_count
                print(f'Found {image_count} images in {images_path}')

                self.images_path = images_path
                self.masks_path = masks_path

                self.train_dataset, self.validation_dataset, self.test_dataset =  self.load_from_directory()

            # self.model = UNet(
            #     num_classes=self.num_classes,
            #     width=self.width,
            #     height=self.height
            # )

            # self.model = ResNet(
            #     num_classes=self.num_classes,
            # )

            self.model = EfficientNet(
                num_classes=self.num_classes,
            )

            # self.model = VGGNet(
            #     num_classes=self.num_classes,
            # )

        else:
            self.load()

    def load_from_dataset(self, dataset):
        train_dataset = dataset['train']
        validation_dataset = dataset['validation']
        test_dataset = dataset['test']
        print(f"Found {train_dataset.cardinality()} images in train dataset")

        return train_dataset, validation_dataset, test_dataset


    def load_from_directory(self) -> Tuple[BatchDataset, BatchDataset, BatchDataset]:

        images_ds: Dataset = tf.keras.utils.image_dataset_from_directory(
            self.images_path,
            validation_split=0.9,
            subset="training",
            seed=self.seed,
            image_size=(self.height, self.width),
            # batch_size=self.batch_size,
            labels = None)
        masks_ds: Dataset = tf.keras.utils.image_dataset_from_directory(
            self.masks_path,
            validation_split=0.9,
            subset="training",
            seed=self.seed,
            image_size=(self.height, self.width),
            # batch_size=self.batch_size,
            color_mode='grayscale',
            labels = None)
        train_ds = tf.data.Dataset.zip((images_ds, masks_ds))
        del images_ds, masks_ds

        train_ds = train_ds.map(lambda images, masks: {'image': images, 'segmentation_mask': masks})

        print(f"dataset_size: {self.dataset_size}")

        self.dataset_size = train_ds.cardinality()*self.batch_size
        print(f"dataset_size: {self.dataset_size}")

        val_ds = train_ds.take(int(self.validation_split * (self.dataset_size/self.batch_size)))
        train_ds = train_ds.skip(int(self.validation_split * (self.dataset_size/self.batch_size)))

        test_ds = train_ds.take(int(self.validation_split * (self.dataset_size/self.batch_size)))
        train_ds = train_ds.skip(int(self.validation_split * (self.dataset_size/self.batch_size)))


        # test_ds: BatchDataset = tf.keras.utils.image_dataset_from_directory(
        #     self.data_path,
        #     validation_split=self.validation_split,
        #     subset="validation",
        #     seed=self.seed,
        #     image_size=(self.height, self.width),
        #     batch_size=self.batch_size)

        train_ds = train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        train_size = train_ds.cardinality()*self.batch_size
        train_percent = int((train_size/self.dataset_size)*100)

        val_size = val_ds.cardinality()*self.batch_size
        val_percent = int((val_size/self.dataset_size)*100)

        test_size = test_ds.cardinality()*self.batch_size
        test_percent = int((test_size/self.dataset_size)*100)

        print(f"Dataset: {self.dataset_size}, train: {train_size} ({train_percent}%), val: {val_size} ({val_percent}%), test: {test_size} ({test_percent}%)")

        return train_ds, val_ds, test_ds

    def retrieve_datasets(self):
        
        return self.train_dataset, self.validation_dataset, self.test_dataset

    def prepare(self, ds: PrefetchDataset, shuffle=False, augment=False):
        def load_image(datapoint):
            input_image = tf.image.resize(datapoint['image'], (self.height, self.width))
            input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.height, self.width))

            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask = tf.cast(input_mask, tf.float32) / 255.0
            # input_mask -= 1

            return input_image, input_mask

        ds = ds.map(load_image, num_parallel_calls=self.AUTOTUNE)
        
        # resize_and_rescale = Sequential([
        #     tf.keras.layers.Resizing(self.width, self.height),
        #     tf.keras.layers.Rescaling(1./255)
        # ])

        # data_augmentation = Sequential([
        #     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        #     tf.keras.layers.RandomRotation(0.2),
        # ])

        # # Resize and rescale all datasets.
        # ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
        #       num_parallel_calls=self.AUTOTUNE)

        if isinstance(self.model, ResNet):
        # if isinstance(self.model, Model):

            print(f"{type(self.model)} Model so preprocess inputs for {type(self.model)}")
            preprocess_input = sm.get_preprocessing(self.model.BACKBONE)
            ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=self.AUTOTUNE)

        elif isinstance(self.model, EfficientNet):
            print(f"{type(self.model)} Model so preprocess inputs for {type(self.model)}")
            preprocess_input = tf.keras.applications.efficientnet.preprocess_input
            ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=self.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(self.buffer_size)

        # # Batch all datasets.
        # ds = ds.batch(self.batch_size)

        # Use data augmentation only on the training set.
        # if augment:
        #     ds = ds.map(Augment(self.seed))
            # ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
            #           num_parallel_calls=self.AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=self.AUTOTUNE)

    def preprocess(self, train_only = False):
        self.train_dataset = self.prepare(self.train_dataset, shuffle=True, augment=True)
        if not train_only:
            self.validation_dataset = self.prepare(self.validation_dataset)
            self.test_dataset = self.prepare(self.test_dataset)

    def compile(self, optimizer = None, loss = None, metrics = None):
        self.model.compile(
            optimizer=self.optimizer if optimizer is None else optimizer,
            loss=self.loss if loss is None else loss, 
            metrics=self.metrics  if metrics is None else metrics)
        self.model.summary()  
        

    def fit(self, x, y = None, validation_data = None, epochs=10, validation_steps=None, steps_per_epoch=None):
        self.history = self.model.fit(
                                      x=x,
                                      y=y,
                                      validation_data=validation_data,
                                      validation_steps=validation_steps,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      batch_size=self.batch_size
                                    )

    def fit_with_photo(self, x, y = None, validation_data = None, epochs=10, validation_steps=None, steps_per_epoch=None):
        for images, masks in self.train_dataset.take(1):
            sample_image, sample_mask = images[1], masks[1]
        def show_predictions(dataset=None, num=1):
            def create_mask(pred_mask):
                pred_mask = tf.math.argmax(pred_mask, axis=-1)
                pred_mask = pred_mask[..., tf.newaxis]
                return pred_mask[0]

            if dataset:
                for image, mask in dataset.take(num):
                    pred_mask = self.model.predict(image)
                    self.display([image[0], mask[0], create_mask(pred_mask)])

            else:
                self.display([sample_image, sample_mask,create_mask(self.model.predict(sample_image[tf.newaxis, ...]))])

        class DisplayCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                clear_output(wait=True)
                show_predictions(dataset=None, num=1)
                print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

        self.history = self.model.fit(
                                      x=x,
                                      y=y,
                                      validation_data=validation_data,
                                      validation_steps=validation_steps,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      batch_size=self.batch_size,
                                      callbacks=[DisplayCallback()]
                                    )
        
    def fit_and_save(self,
                     x,
                     y = None, 
                     validation_data = None, 
                     epochs=10, 
                     initial_epoch = 0, 
                     id = None, 
                     csv_id = 0, 
                     photo=True, 
                     save_best = True, 
                     monitor_mode = 'loss', # Other option metric
                     early_stopping = True, 
                     early_stopping_patience = 5,
                     reduce_lr = True,
                     reduce_lr_factor = 0.2,
                     reduce_lr_patience = 3):
        id = id if id is not None else self.training_id
        to_display = []
        for images, masks in self.train_dataset.take(1):
            # n_images = images.shape[0]
            to_display.append((images[0:10, :, :, :], masks[0:10, :, :, :]))

        def show_predictions(dataset=None, num=1):
            def create_mask(pred_mask):
                # pred_mask = tf.math.argmax(pred_mask, axis=-1)
                print(f"Predict: {pred_mask.shape}")

                # pred_mask = pred_mask[..., tf.newaxis]
                return pred_mask[0]

            if dataset:
                for image, mask in dataset.take(num):
                    pred_mask = self.model.predict(image)
                    self.display([image[0], mask[0], create_mask(pred_mask)])

            else:
                random_image = random.randint(0, 9)
                sample_image = to_display[0][0][random_image]
                sample_mask = to_display[0][1][random_image]
                print(f"Image: {sample_image.shape}")
                print(f"Mask: {sample_mask.shape}")
                self.display([sample_image, sample_mask,create_mask(self.model.predict(sample_image[tf.newaxis, ...]))])

        class DisplayCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                clear_output(wait=True)
                show_predictions(dataset=None, num=1)
                print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = "checkpoints/training_" + str(id) + "/cp-{epoch:04d}.ckpt"
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            # monitor='val_dice_coefficient',
            monitor='val_loss' if monitor_mode is 'loss' else 'val_dice_coefficient',
            mode='min' if monitor_mode is 'loss' else 'max',
            verbose = 1, 
            save_best_only = save_best,
            save_weights_only = True)
        early_stop = tf.keras.callbacks.EarlyStopping(
            # monitor = 'val_dice_coefficient',
            monitor = 'val_loss' if monitor_mode is 'loss' else 'val_dice_coefficient',
            mode='min' if monitor_mode is 'loss' else 'max',
            patience = early_stopping_patience,
            verbose = 1
        )
        csv_logger = tf.keras.callbacks.CSVLogger(
            filename = "logs/" + "logs_" + str(csv_id) + ".csv",
            append = True
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss' if monitor_mode is 'loss' else 'val_dice_coefficient',
            mode ='min' if monitor_mode is 'loss' else 'max',
            verbose = 1,
            factor = reduce_lr_factor,
            patience = reduce_lr_patience,
        )
        # Save the weights using the `checkpoint_path` format
        self.model.save_weights(checkpoint_path.format(epoch=0))

        callbacks = [cp_callback, csv_logger]
        if early_stopping:
            callbacks.append(early_stop)
        if reduce_lr:
            callbacks.append(reduce_lr_cb)
        if photo:
            callbacks.append(DisplayCallback())

        # Train the model with the new callback
        self.history = self.model.fit(
                                      x=x,
                                      y=y,
                                      validation_data=validation_data,
                                      epochs=epochs,
                                      initial_epoch = initial_epoch,
                                      batch_size=self.batch_size,
                                      callbacks=callbacks
                                     )

    def load_from_checkpoint(self,  id = None):
        id = id if id is not None else self.training_id
        checkpoint_path = "checkpoints/training_" + str(id) + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            # Load the previously saved weights
            self.model.load_weights(latest)
        
    def evaluate(self, x, y = None):
        test_loss, test_acc = self.model.evaluate(x, y, verbose=1)
        print(f'Test Dice Coefficient: {test_acc}')

    def save_results(self, epochs=10):
        # create directory called plots using command line in python
        os.system('mkdir -p plots')

        history = self.history
        acc = history.history['dice_coefficient']
        val_acc = history.history['val_dice_coefficient']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Dice Coefficient')
        plt.plot(epochs_range, val_acc, label='Validation Dice Coefficient')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Dice Coefficient')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        # save the plot as png file
        filename = str(datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.png'
        plt.savefig(f'plots/{filename}')
        print(f"Plot saved as {filename}")
        # plt.show()

    def show_all_results(self):
        # create directory called plots using command line in python
        os.system('mkdir -p plots')

        if self.history is not None:
            history_path = 'history'
            if os.path.exists(history_path):

                values_dict = {}
                for key in self.history.history.keys():
                    with open(history_path + '/' + key + '.txt', 'r') as f:
                        values = [float(i) for i in f.read().splitlines()]
                        values_dict[key] = values

                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.plot(range(len(values_dict['dice_coefficient'])), values_dict['dice_coefficient'], label='Training Dice Coefficient')
                plt.plot(range(len(values_dict['val_dice_coefficient'])), values_dict['val_dice_coefficient'], label='Validation Dice Coefficient')
                plt.legend(loc='lower right')
                plt.title('Training and Validation Dice Coefficient')

                plt.subplot(1, 2, 2)
                plt.plot(range(len(values_dict['loss'])), values_dict['loss'], label='Training Loss')
                plt.plot(range(len(values_dict['val_loss'])), values_dict['val_loss'], label='Validation Loss')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Loss')
                # save the plot as png file
                filename = str(datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.png'
                plt.savefig(f'plots/{filename}')
                print(f"Plot saved as {filename}")

    def predict_by_array(self, img_array):
        # img = tf.keras.utils.load_img(
        #     image_path, target_size=(self.height, self.width)
        # )
        # img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        predictions = tf.math.argmax(predictions, axis=-1)
        predictions = predictions[..., tf.newaxis]
        return predictions[0]
        # score = tf.nn.softmax(predictions[0])

        # print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(self.class_names[np.argmax(score)], 100 * np.max(score)))

    def save(self, id = None, save_format = 'tf'):
        os.system('mkdir -p saved_models')
        os.system('mkdir -p history')
        id = id if id is not None else self.training_id

        if save_format == 'h5':
            self.model.save(f'saved_models/model_{id}.h5')
        else:
            self.model.save(f'saved_models/model_{id}')
        print(f"Model saved as model_{id}")
        if self.history is not None:
            for key, value in self.history.history.items():
                with open('history/' + key +'.txt', 'a') as f:
                    for line in self.history.history[key]:
                        f.write(f"{line}\n")

        

    def load(self, id = None, load_format = 'tf'):
        id = id if id is not None else self.training_id

        saved_model_path = self.base_path + '/' + f'saved_models/model_{id}'
        if load_format == 'h5':
            saved_model_path = self.base_path + '/' + f'saved_models/model_{id}.h5'

        self.model = tf.keras.models.load_model(saved_model_path, 
                                                custom_objects={'DiceCoefficient':DiceCoefficient, 
                                                                'DiceLoss': DiceLoss, 
                                                                # 'SimpleSegmentation':SimpleSegmentation, 
                                                                'Augment':Augment}, 
                                                compile=False)
        print(f"Model loaded from model_{id}")

    def display(self, display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        # plt.savefig('display/sample.png')
        plt.show()

    def dice_coefficient(self, y_true, y_pred):
        # Dice Coefficient
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        return (2 * intersection + 1.0) / (union + 1.0)
        # return tf.keras.backend.mean( (2. * intersection + 1.) / (union + 1.))

    def loss_function(self, y_true, y_pred):
        # Dice Coefficient Loss
        return 1 - tf.keras.backend.mean(self.metric_function(y_true, y_pred))
        # return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)

    # Reproducability
    def set_seed(self, seed=31415):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # tf.keras.utils.set_random_seed(1)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
