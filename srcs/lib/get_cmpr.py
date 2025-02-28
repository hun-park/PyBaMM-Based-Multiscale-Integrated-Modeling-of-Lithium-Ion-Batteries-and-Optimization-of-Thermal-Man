# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import get_imgs

    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='imgs.pkl labels.pkl pick_data.pkl pick_label.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--algorithm', default='Recurrent Plot')
    parser.add_argument('--iscumm', default='True')
    parser.add_argument('--verbose', '-v', default=False)
    parser.add_argument('--assets', default='/working/srcs/assets')
    parser.add_argument('--models', default='vgg16.h5 alexnet.h5')
    parser.add_argument('--historys', default='vgg16.json alexnet.json')
    parser.add_argument('--examples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--fraction', type=int, default=60)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)
    os.environ['TF_CPP_MIN_LOG_LEVEL']="2" if eval(args.verbose) else "3"
    tf.random.set_seed(args.seed)
    tf.debugging.set_log_device_placement(eval(args.verbose))
    
    [imgs, labels, pick_data, pick_label] = utils.load_data(args.src, args.input)
    historys = utils.load_data(args.src, args.historys)
    
    for model in args.models.split():
        locals()[str(model)] = tf.keras.models.load_model(os.path.join(args.src, model))
        pick_imgs = get_imgs.encode_data(pick_data, args.algorithm, args.iscumm, "False")/255.0
        locals()[str(model)+"predict"] = locals()[str(model)].predict(pick_imgs)
    
    if len(args.historys.split()) - 1:
        for idx in range(len(historys)):
            plt.plot(historys[idx]['loss'], 'y', label='train loss')
            plt.plot(historys[idx]['val_loss'], 'r', label='val loss')
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_loss.jpg'))
            print(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_loss.jpg'))
            plt.clf()

            plt.plot(historys[idx]['mape'], 'y', label='train mape')
            plt.plot(historys[idx]['val_mape'], 'r', label='val mape')
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_mape.jpg'))
            print(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_mape.jpg'))
            plt.clf()

            plt.plot(locals()[str(args.models.split()[idx])+"predict"][0:args.examples], 'yo', label='predict')
            plt.plot(pick_label[0:args.examples], 'ro', label='truth')
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_predict.jpg'))
            print(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_predict.jpg'))
            plt.clf()
    else:
        plt.plot(historys['loss'], 'y', label='train loss')
        plt.plot(historys['val_loss'], 'r', label='val loss')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models)[0]}_loss.jpg'))
        print(os.path.join(args.assets, f'{os.path.splitext(args.models)[0]}_loss.jpg'))
        plt.clf()

        plt.plot(historys['mape'], 'y', label='train mape')
        plt.plot(historys['val_mape'], 'r', label='val mape')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models)[0]}_mape.jpg'))
        print(os.path.join(args.assets, f'{os.path.splitext(args.models)[0]}_mape.jpg'))
        plt.clf()

        plt.plot(locals()[str(args.models)+"predict"][0:args.examples], 'yo', label='predict')
        plt.plot(pick_label[0:args.examples], 'ro', label='truth')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models)[0]}_predict.jpg'))
        print(os.path.join(args.assets, f'{os.path.splitext(args.models)[0]}_predict.jpg'))
        plt.clf()

if __name__ == "__main__":
    main()
