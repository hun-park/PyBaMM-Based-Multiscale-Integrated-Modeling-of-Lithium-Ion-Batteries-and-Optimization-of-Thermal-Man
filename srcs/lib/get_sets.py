# %% [code]
import argparse
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(src, input_file):
    return pd.read_pickle(os.path.join(src, input_file))

def merge_data(data, cell, interval, chunk): 
    cycles = tqdm.tqdm(data[cell].keys())
    mems = {}

    for cycle in cycles:
        cycle_data = data[cell][cycle]
        if hasattr(data[cell][cycle], "__iter__"):
            online = np.column_stack((cycle_data['V'].values.flatten(),
                                     cycle_data['I'].values.flatten(),
                                     cycle_data['T'].values.flatten(),
                                     cycle_data['Q'].values.flatten()))
            t_values = cycle_data['t'].values.flatten()
            end = t_values[-1]
            timestamps = np.arange(0, end, interval)

            mem = np.empty((len(timestamps), 6))
            mem[:, 0] = timestamps
            mem[:, 1:5] = np.nan

            idx = np.searchsorted(t_values, timestamps)
            valid_idx = idx < len(t_values)

            mem[valid_idx, 1:5] = online[idx[valid_idx], :]

            I = mem[:, 2]
            mem[:, 5] = (np.cumsum(I) * interval) / 1000  # mA

            mems[cycle] = pd.DataFrame(mem, columns = ['t','V','I', 'T', 'Q', 'Idt'])
    return mems

def rearrange_data(data, pick, chunk):
    x_data, y_data = [], []

    cycles = tqdm.tqdm(data.keys().drop(pick))
    for cycle in cycles:
        if hasattr(data[cell][cycle], "__iter__"):
            shifted_V, shifted_I, shifted_T, shifted_Idt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            for shift in range(1, chunk+1):
                shifted_V[f'V(t-{shift})']     = mems[cycle]['V'].shift(shift)
                shifted_I[f'I(t-{shift})']     = mems[cycle]['I'].shift(shift)
                shifted_T[f'T(t-{shift})']     = mems[cycle]['T'].shift(shift)
                shifted_Idt[f'Idt(t-{shift})'] = mems[cycle]['Idt'].shift(shift)
            shifted_V, shifted_I   = shifted_V.dropna(), shifted_I.dropna()
            shifted_T, shifted_Idt = shifted_T.dropna(), shifted_Idt.dropna()
            reduced_Q = mems[cycle]['Q'].loc[shifted_V.index]

            for idx in reduced_Q.index:
                chunked = np.column_stack((shifted_Idt.loc[idx].values, shifted_T.loc[idx].values, 
                                           shifted_I.loc[idx].values, shifted_V.loc[idx].values))
                x_data.append(np.flip(chunked))

            y_data.extend(reduced_Q.values)
    x_data, y_data = np.array(x_data), np.array(y_data)

    picked_x, picked_y = [], []

    picked_V, picked_I, picked_T, picked_Idt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for shift in range(1, chunk+1):
        picked_V[f'V(t-{shift})'] = picked['V'].shift(shift)
        picked_I[f'I(t-{shift})'] = picked['I'].shift(shift)
        picked_T[f'T(t-{shift})'] = picked['T'].shift(shift)
        picked_Idt[f'Idt(t-{shift})'] = picked['Idt'].shift(shift)
    picked_V, picked_I   = picked_V.dropna(), picked_I.dropna()
    picked_T, picked_Idt = picked_T.dropna(), picked_Idt.dropna()
    red_Q = picked['Q'].loc[picked_V.index]

    for idx in red_Q.index:
        chunked = np.column_stack((picked_Idt.loc[idx].values, picked_T.loc[idx].values, 
                                   picked_I.loc[idx].values, picked_V.loc[idx].values))
        picked_x.append(np.flip(chunked))

    picked_y.extend(red_Q.values)
    picked_x, picked_y = np.array(picked_x), np.array(picked_y)
    picked = {'x':picked_x, 'y':picked_y}
    
    return x_data, y_data, picked

def save_data(dest, ch_output, ch_data, disch_output, disch_data):
    with open(os.path.join(dest, ch_output), 'wb') as fp:
        pickle.dump(ch_data, fp)
        
    with open(os.path.join(dest, disch_output), 'wb') as fp:
        pickle.dump(disch_data, fp)
    
def main():
    parser = argparse.ArgumentParser(description='Rearrange the ch/disch from the "Source/Input" and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='ch_data.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--cell', default='b1c18')
    parser.add_argument('--data', default='data.pkl')
    parser.add_argument('--label', default='label.pkl')
    parser.add_argument('--pick', type=int, default=11)
    parser.add_argument('--interval', type=int, default=0.01)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--verbose', '-v', default=False)

    args = parser.parse_args()

    input_data = load_data(args.src, args.input)
    merge_data = merge_data(input_data, args.cell, args.interval)
    data, pick = rearrange_data(merge_data, args.pick, args.chunk)
    
    save_data(args.dest, args.ch_output, ch_data, args.disch_output, disch_data)

if __name__ == "__main__":
    main()
