# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd


def scale_data(standard, data, algorithm, cycle, dest, verbose, pass_through):
    voltage = np.concatenate([value['V'].to_numpy() for key, value in standard.items() if 'V' in value.columns])
    current = np.concatenate([value['I'].to_numpy() for key, value in standard.items() if 'I' in value.columns])
    temperature = np.concatenate([value['T'].to_numpy() for key, value in standard.items() if 'T' in value.columns])
    cumulated = np.concatenate([value['Idt'].to_numpy() for key, value in standard.items() if 'Idt' in value.columns])
    capacity = np.concatenate([value['Q'].to_numpy() for key, value in standard.items() if 'Q' in value.columns])
    
    if eval(verbose):
        utils.save_hist([capacity, current, cumulated, voltage, temperature],
                        ["Ch. Cap.", "Current", "Counted Coulomb", "Voltage", "Temperature"],
                        dest,f"{algorithm}_raw_hist")
    
    sc_standard = standard if eval(pass_through) else utils.scale(standard, data, algorithm)
    
    sc_voltage = np.concatenate([value['V'].to_numpy() for key, value in sc_standard.items() if 'V' in value.columns])
    sc_current = np.concatenate([value['I'].to_numpy() for key, value in sc_standard.items() if 'I' in value.columns])
    sc_temperature = np.concatenate([value['T'].to_numpy() for key, value in sc_standard.items() if 'T' in value.columns])
    sc_cumulated = np.concatenate([value['Idt'].to_numpy() for key, value in sc_standard.items() if 'Idt' in value.columns])
    sc_capacity = np.concatenate([value['Q'].to_numpy() for key, value in sc_standard.items() if 'Q' in value.columns])
    
    if eval(verbose):
        utils.save_hist([sc_capacity, sc_current, sc_cumulated, sc_voltage, sc_temperature],
                        ["Ch. Cap.", "Current", "Counted Coulomb", "Voltage", "Temperature"],
                        dest,f"{algorithm}_scaled_hist")
        
        utils.save_plot([sc_standard[cycle]['t'], sc_standard[cycle]['t'], sc_standard[cycle]['t']],
                        ["time (min)", "time (min)", "time (min)"],
                        [sc_standard[cycle]['V'], sc_standard[cycle]['I'], sc_standard[cycle]['T']],
                        ["Voltage (V)", "Current (A)", "Temperature (ºC)"],
                        dest,f"{algorithm}_scaled_plot")
    return sc_standard
    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='b1c18_mems.pkl b1c19_mems.pkl b1c38_mems.pkl b1c11_mems.pkl')
    parser.add_argument('--output', default='b1c18_scaled.pkl b1c19_scaled.pkl b1c38_scaled.pkl b1c11_scaled.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets/')
    parser.add_argument('--cycle', type=int, default=100)
    parser.add_argument('--algorithm', default='MM')
    parser.add_argument('--verbose', '-v', default="False")
    parser.add_argument('--pass_through', default="False")

    args = parser.parse_args()

    raw_data = utils.load_data(args.src, args.input)
    
    if type(raw_data) is list:
        scaled_data = []
        for idx in range(len(raw_data)):
            scaled_data.append(scale_data(raw_data[0], raw_data[idx], args.algorithm, args.cycle, args.assets, args.verbose, args.pass_through))
    else:
        scaled_data = scale_data(raw_data, raw_data, args.algorithm, args.cycle, args.assets, args.verbose, args.pass_through)
    
    utils.save_data(args.dest, args.output, scaled_data)

if __name__ == "__main__":
    main()
