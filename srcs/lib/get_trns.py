# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd


def chunk_data(data, chunk, pick, predict, dest, verbose):
    x_data, y_data = [], []
    pick_data, pick_label = [], []
    cycles = tqdm.tqdm(data.keys()) if eval(verbose) else data.keys()

    for cycle in cycles:
        # Extract relevant columns
        cycle_data = data[cycle][['Idt', 'T', 'I', 'V']]
        
        # Shift columns and drop NaN rows
        shifted_data = pd.concat([cycle_data.shift(shift) for shift in range(1, chunk + 1)], axis=1).dropna()

        # Extract reduced_Q corresponding to shifted_data
        reduced_Q = data[cycle][predict].loc[shifted_data.index]

        # Create the chunked data
        chunked_data = np.flip(shifted_data.values, axis=1)

        # Extend x_data and y_data
        if cycle == pick:
            pick_data.extend(chunked_data)
            pick_label.extend(reduced_Q)

        x_data.extend(chunked_data)
        y_data.extend(reduced_Q)

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (-1, chunk, 4))
    pick_data, pick_label = np.reshape(np.array(pick_data), (-1, chunk, 4)), np.array(pick_label)
    return x_data, y_data, pick_data, pick_label


def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='scaled.pkl')
    parser.add_argument('--output', default='data.pkl labels.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets/')
    parser.add_argument('--predict', default='Q')
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--pick', type=int, default=11)
    parser.add_argument('--verbose', '-v', default="False")

    args = parser.parse_args()

    raw_data = utils.load_data(args.src, args.input)
    chunked_data, label, picked_data, picked_label = chunk_data(raw_data, args.chunk, args.pick, args.predict, args.assets, args.verbose)
    if eval(args.verbose):
        print('V\t', 'I\t', 'T\t', 'Idt')
        print(chunked_data[0])
        print(chunked_data[0].shape)
    utils.save_data(args.dest, args.output, [chunked_data, label, picked_data, picked_label])

if __name__ == "__main__":
    main()
