# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_data(data, algorithm, iscumm, verbose, chunk):
    imgs = []
    
    data = np.array(data*255, np.uint8)
    ttrain = tqdm.tqdm(range(len(data.transpose()[0].transpose())-(chunk-1))) if eval(verbose) else range(len(data.transpose()[0].transpose())-(chunk-1))
    for t in ttrain:
        current = 3 if eval(iscumm) else 1
        img = np.array([np.array(utils.encode(data.transpose()[0].transpose(), t, algorithm), np.uint8),
                        np.array(utils.encode(data.transpose()[2].transpose(), t, algorithm), np.uint8),
                        np.array(utils.encode(data.transpose()[current].transpose(), t, algorithm), np.uint8)])
        img = np.swapaxes(img, 0, -1)
        imgs.append(img)
    imgs = np.array(imgs)
    
    return imgs
    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='data.pkl')
    parser.add_argument('--output', default='imgs.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets/')
    parser.add_argument('--cycle', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--algorithm', default='Recurrent Plot')
    parser.add_argument('--iscumm', default='True')
    parser.add_argument('--verbose', '-v', default="False")

    args = parser.parse_args()

    raw_data = utils.load_data(args.src, args.input)
    imgs = encode_data(raw_data, args.algorithm, args.iscumm, args.verbose, args.chunk)
    if eval(args.verbose):
        print(imgs.shape)
        plt.imsave(os.path.join(args.assets, f'{args.output.split(".")[0]}_{args.cycle}.jpg'), imgs[args.cycle])
        print(os.path.join(args.assets, f'{args.output.split(".")[0]}_{args.cycle}.jpg'))
    utils.save_data(args.dest, args.output, imgs)

if __name__ == "__main__":
    main()
