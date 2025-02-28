# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def classify(labels, sign, verbose):
    classes = np.array([np.round(label, sign) for label in labels])
    if verbose : print(f'labels:{labels} ==> classes:{classes}')
    return classes
    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='data.pkl')
    parser.add_argument('--output', default='imgs.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--sign', type=int, default=2)
    parser.add_argument('--verbose', '-v', default="False")

    args = parser.parse_args()

    labels = utils.load_data(args.src, args.input)
    classes = classify(labels, args.sign, args.verbose)
    if eval(args.verbose):
        print(f'labels : {np.array(labels).shape} ==> classes : {classes.shape}')
    utils.save_data(args.dest, args.output, classes)

if __name__ == "__main__":
    main()
