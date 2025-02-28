# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='imgs.pkl labels.pkl')
    parser.add_argument('--output', default='alexnet.h5')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets')
    parser.add_argument('--verbose', '-v', default=False)
    parser.add_argument('--history', default='alexnet.json')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--GPU')
    parser.add_argument('--fraction', type=int, default=60)
    parser.add_argument('--size', type=int, default=500000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--predict', default="Q")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)
    os.environ['TF_CPP_MIN_LOG_LEVEL']="2" if eval(args.verbose) else "3"
    tf.random.set_seed(args.seed)
    tf.debugging.set_log_device_placement(eval(args.verbose)) # display an allocated computing device name as False

    [imgs, labels] = utils.load_data(args.src, args.input)
    imgs = imgs[0:args.size]; labels = labels[0:args.size]
    
    img_train, img_test, label_train, label_test = train_test_split(imgs, labels, random_state=args.seed, test_size=0.4, shuffle=True)
    img_test, img_val, label_test, label_val = train_test_split(img_test, label_test, random_state=args.seed, test_size=0.5, shuffle=True)
    
    train_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    
    if args.predict == 'SoC':
        label_train = label_train/100.0
        label_test  = label_test/100.0
        label_val   = label_val/100.0

    if eval(args.verbose):
        print(img_train.shape, label_train.shape, img_test.shape, label_test.shape, img_val.shape, label_val.shape)
        print(np.max(labels), np.max(img_train), np.min(labels), np.min(img_train))
        
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction/100
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.InputLayer(input_shape=img_train[0].shape),
#         tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', filters=32),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', filters=64),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(1, activation='linear')
#     ])
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=img_train[0].shape,
        pooling="avg")
    flat = tf.keras.layers.Flatten()(resnet.output)
    rslt = tf.keras.layers.Dense(1, activation='linear')(flat)
    model = tf.keras.Model(inputs=resnet.input, outputs=rslt)
    
    opt = tf.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, 50))
    model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mape'])
    model.build(img_train[0].shape)
    model.summary()
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=True),
             tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.dest, args.output), monitor='val_loss', save_best_only=True, verbose=True),]
    
    history = model.fit(train_datagen.flow(img_train, label_train, batch_size=args.batch_size), 
                        validation_data=valid_datagen.flow(img_val, label_val, batch_size=args.batch_size), 
                        steps_per_epoch=int(len(img_train)/args.batch_size),
                        epochs=args.epochs, 
                        callbacks=callbacks)
    
    #model.evaluate(img_test, label_test)
    
    utils.save_data(args.dest, args.history, history.history)
    print(f'{args.history} is successfully saved.')

if __name__ == "__main__":
    main()
