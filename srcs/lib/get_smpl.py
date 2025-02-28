# %% [code]
import argparse
import os
import tqdm
import utils
import pickle
import numpy as np
import pandas as pd


def sample_data(data, cell, interval, verbose):
    """
    Samples data from the given data dictionary and returns a dictionary of DataFrames.
    
    Parameters:
    - data: dict, the data loaded from the source
    - cell: str, the cell identifier
    - interval: float, the sampling interval
    - verbose: bool, flag to show progress bar
    
    Returns:
    - mems: dict, a dictionary containing sampled data for each cycle
    """
    mems = {}
    # Determine the appropriate data access method based on the cell
    if cell != 'NASA':
        data_cycles = data[cell]
    else:
        data_cycles = data

    cycles = tqdm.tqdm(data_cycles.keys()) if eval(verbose) else data_cycles.keys()
    
    for cycle in cycles:
        cycle_data = data_cycles[cycle]
        if hasattr(cycle_data, "__iter__"):
            try:
                # Extract and flatten data arrays
                V = cycle_data['V'].values.flatten()
                I = cycle_data['I'].values.flatten()
                T = cycle_data['T'].values.flatten()
                Q = cycle_data['Q'].values.flatten()
                SoC = 100 * (Q / max(Q))
                t_values = cycle_data['t'].values.flatten()
                
                # Determine sampling timestamps
                end = t_values[-1]
                timestamps = np.arange(0, end, interval)
                
                # Initialize memory array
                mem = np.empty((len(timestamps), 7))
                mem[:, 0] = timestamps
                mem[:, 1:] = np.nan  # Initialize data columns with NaN
                
                # Find indices where t_values matches timestamps
                idx = np.searchsorted(t_values, timestamps)
                valid_idx = idx < len(t_values)
                
                # Stack online data for valid indices
                online = np.column_stack((V, I, T, Q, SoC))
                mem[valid_idx, 1:6] = online[idx[valid_idx], :]
                
                # Calculate cumulative current (Idt)
                I_sampled = mem[:, 2]
                mem[:, 6] = np.cumsum(I_sampled * interval)
                
                # Create DataFrame and add to mems dictionary
                mems[cycle] = pd.DataFrame(mem, columns=['t', 'V', 'I', 'T', 'Q', 'SoC', 'Idt'])
            except Exception as e:
                print(f"Error processing cycle {cycle}: {e}")
                continue  # Skip to the next cycle if there's an error
    return mems
    
def main():
    parser = argparse.ArgumentParser(description='Sample data from the ch_data with intervals and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='ch_data.pkl')
    parser.add_argument('--output', default='b1c18_mems.pkl b1c18_cycle.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets/')
    parser.add_argument('--cell', default='b1c18')
    parser.add_argument('--cycle', type=int, default=100)
    parser.add_argument('--interval', type=float, default=0.01)
    parser.add_argument('--verbose', '-v', default=False)

    args = parser.parse_args()

    ch_data = utils.load_data(args.src, args.input)
    mems = sample_data(ch_data, args.cell, args.interval, args.verbose)
    if eval(args.verbose):
        cycle = args.cycle
        utils.save_plot([mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t'],mems[cycle]['t']],
                    ["time (min)","time (min)","time (min)","time (min)","time (min)","time (min)","time (min)","time (min)","time (min)","time (min)","time (min)","time (min)"],
                    [mems[cycle]['Q'],mems[cycle]['Idt'],mems[cycle]['I'],mems[cycle]['Q'],mems[cycle]['SoC'],mems[cycle]['I'],mems[cycle]['V'],mems[cycle]['I'],mems[cycle]['T'],mems[cycle]['V'],mems[cycle]['Idt'],mems[cycle]['T']],
                    ["Ch. Cap. (Ah)","Counted Coulomb (Ah)","Current (A)","Ch. Cap. (Ah)","SoC (%)","Current (A)","Voltage (V)","Current (A)","Temperature (ºC)","Voltage (V)","Counted Coulomb (Ah)","Temperature (ºC)"],
                    args.assets,
                    f"{args.cell}_cumulated"
                   )
    utils.save_data(args.dest, args.output, mems)

if __name__ == "__main__":
    main()
