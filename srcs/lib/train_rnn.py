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
    parser.add_argument('--output', default='vgg16.h5')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets')
    parser.add_argument('--verbose', '-v', default=False)
    parser.add_argument('--history', default='vgg16.json')
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

    file_list = args.input.split()
    V_train = pd.read_pickle(os.path.join(args.src, file_list[0]))
    I_train = pd.read_pickle(os.path.join(args.src, file_list[1]))
    T_train = pd.read_pickle(os.path.join(args.src, file_list[2]))
    y_train = pd.read_pickle(os.path.join(args.src, file_list[3]))
    V_valid = pd.read_pickle(os.path.join(args.src, file_list[4]))
    I_valid = pd.read_pickle(os.path.join(args.src, file_list[5]))
    T_valid = pd.read_pickle(os.path.join(args.src, file_list[6]))
    y_valid = pd.read_pickle(os.path.join(args.src, file_list[7]))
    
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction / 100
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    data_train = np.concatenate((V_train, I_train, T_train), axis=1)
    data_valid = np.concatenate((V_valid, I_valid, T_valid), axis=1)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=data_train[0].shape),

        tf.keras.layers.SimpleRNN(2048),

        tf.keras.layers.Dense(1, activation='linear')
    ])

    opt = tf.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, 50))
    model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mape'])
    model.build(data_train[0].shape)
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=True),
             tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.dest, args.output), monitor='val_loss', save_best_only=True, verbose=True)]

    history = model.fit(data_train, y_train, 
                        validation_data=(data_valid, y_valid), 
                        batch_size=args.batch_size,
                        epochs=args.epochs, 
                        callbacks=callbacks)

    utils.save_data(args.dest, args.history, history.history)
    print(f'{args.history} is successfully saved.')

if __name__ == "__main__":
    main()
