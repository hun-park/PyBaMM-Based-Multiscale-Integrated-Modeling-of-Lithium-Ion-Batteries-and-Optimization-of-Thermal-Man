# %% [code]
import argparse
import os
import h5py
import utils
import pickle
import pandas as pd

def load_data(src, input_file):
    package = h5py.File(os.path.join(src, input_file))
    batch_dict = {}

    for cell_idx in range(package['batch']['summary'].size):
        cell_dict = {}
        
        # Extract summary data
        summary_group = package[package['batch']['summary'][cell_idx, 0]]
        summary_dict = {key: summary_group[key][0] for key in summary_group}
        
        # Extract cycle data
        cycle_dict = {}
        cycles_group = range(package[package['batch']['cycles'][cell_idx, 0]]['I'].size)
        
        for cycle in cycles_group:
            cycle_data = package[package['batch']['cycles'][0, 0]]
            
            # Convert cycle_key to an integer
            cycle_dict[cycle] = {key: package[package[package['batch']['cycles'][cell_idx, 0]][key][cycle, 0]][()][0] for key in cycle_data}
        
        # Create a DataFrame for summary data
        cell_dict['summary'] = pd.DataFrame(summary_dict)
        
        # Add cycle data
        cell_dict['cycles'] = cycle_dict
        
        # Add to batch_dict
        batch_dict[f"b1c{cell_idx}"] = cell_dict

    return batch_dict

def main():
    parser = argparse.ArgumentParser(description='Get the dataset from the "Source/Input" and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data/mit-stanford')
    parser.add_argument('--input', '-i', default='2017-05-12_batchdata_updated_struct_errorcorrect.mat')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--output', '-o', default='2017-05-12_batchdata_updated_struct_errorcorrect.pkl')
    parser.add_argument('--verbose', '-v', default=False)

    args = parser.parse_args()

    if args.verbose:
        for dirname, _, filenames in os.walk(args.src):
            for filename in filenames:
                print(os.path.join(dirname, filename))

    batch_dict = load_data(args.src, args.input)
    utils.save_data(args.dest, args.output, batch_dict)

if __name__ == "__main__":
    main()
