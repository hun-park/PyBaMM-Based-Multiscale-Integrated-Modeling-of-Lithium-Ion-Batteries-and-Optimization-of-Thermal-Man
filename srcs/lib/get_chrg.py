# %% [code]
import argparse
import os
import utils
import pickle
import numpy as np
import pandas as pd


def split_test(data, dest, cell, cycle):
    cell = data[cell]['cycles'][cycle]
    t_disch = np.where(np.asanyarray(np.isnan(cell['t'][cell['I']>=-0.2])))[0][0]
    
    utils.save_plot([cell['t'],cell['t'],cell['t'],cell['t'],cell['t']],
                    ["time (min)","time (min)","time (min)","time (min)","time (min)"],
                    [cell['V'],cell['I'],cell['T'],cell['Qc'],cell['Qd']],
                    ["Voltage (V)","Current (A)","Temperature (ºC)","Ch. Cap. (Ah)","Disch. Cap. (Ah)"],
                    dest,
                    "united"
                   )
    
    utils.save_plot([cell['t'][:t_disch],cell['t'][:t_disch],cell['t'][:t_disch],cell['t'][:t_disch],cell['t'][:t_disch]],
                    ["time (min)","time (min)","time (min)","time (min)","time (min)"],
                    [cell['V'][:t_disch],cell['I'][:t_disch],cell['T'][:t_disch],cell['Qc'][:t_disch],cell['Qd'][:t_disch]],
                    ["Voltage (V)","Current (A)","Temperature (ºC)","Ch. Cap. (Ah)","Disch. Cap. (Ah)"],
                    dest,
                    "splited"
                   )

def split_data(data): 
    ch_dict, disch_dict = {}, {}

    for cell_id in data.keys():
        cell_data = data[cell_id]
        ch_cell_dict, disch_cell_dict = {}, {}

        cycles = cell_data['cycles'].keys()

        for cycle_num in cycles:
            cycle_data = cell_data['cycles'][cycle_num]
            ch_cycle_dict, disch_cycle_dict = {}, {}

            t_disch = np.where(np.asanyarray(np.isnan(cycle_data['t'][cycle_data['I']>=-0.2])))[0][0]
            ch_cycle_dict['V'], disch_cycle_dict['V'] = cycle_data['V'][:t_disch], cycle_data['V'][t_disch:]
            ch_cycle_dict['I'], disch_cycle_dict['I'] = cycle_data['I'][:t_disch], cycle_data['I'][t_disch:]
            ch_cycle_dict['T'], disch_cycle_dict['T'] = cycle_data['T'][:t_disch], cycle_data['T'][t_disch:]
            ch_cycle_dict['Q'], disch_cycle_dict['Q'] = cycle_data['Qc'][:t_disch], cycle_data['Qd'][t_disch:]
            ch_cycle_dict['t'], disch_cycle_dict['t'] = cycle_data['t'][:t_disch], cycle_data['t'][t_disch:]

            ch_cell_dict[cycle_num], disch_cell_dict[cycle_num] = ch_cycle_dict, disch_cycle_dict

        ch_dict[cell_id], disch_dict[cell_id] = ch_cell_dict, disch_cell_dict

    ch_frame, disch_frame = pd.DataFrame(ch_dict), pd.DataFrame(disch_dict)
    return ch_frame, disch_frame
    
def main():
    parser = argparse.ArgumentParser(description='Split the ch/disch from the "Source/Input" and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='norm_data.pkl')
    parser.add_argument('--output', default='ch_data.pkl disch_data.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets/')
    parser.add_argument('--cell', default='b1c18')
    parser.add_argument('--cycle', type=int, default=100)
    parser.add_argument('--verbose', '-v', default=False)

    args = parser.parse_args()

    clean_dict = utils.load_data(args.src, args.input)
    if eval(args.verbose):
        split_test(clean_dict, args.assets, args.cell, args.cycle)
    ch_data, disch_data = split_data(clean_dict)
    utils.save_data(args.dest, args.output, [ch_data, disch_data])

if __name__ == "__main__":
    main()
