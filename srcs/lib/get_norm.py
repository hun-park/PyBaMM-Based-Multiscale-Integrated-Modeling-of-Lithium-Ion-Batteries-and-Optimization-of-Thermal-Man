# %% [code]
import argparse
import os
import utils
import pickle
import pandas as pd

def get_IQR(df, col, constance=1.5):
    describe = df.describe()[col]
    lower  = (1+constance)*describe['25%'] - constance*describe['75%']
    higher = (1+constance)*describe['75%'] - constance*describe['25%']
    return lower, higher

def get_gradient(df, idx, delta=1):
    return abs(df[idx] - df[idx-delta])/delta if idx else None
    
def remove_data(bat_dict): 
    processed_dict = {}
    
    cell_ids = list(bat_dict.keys())
    for cell in cell_ids:
        SOH = bat_dict[cell]['summary']['QDischarge'].iat[-1]/bat_dict[cell]['summary']['QDischarge'].iat[1]
        
        if SOH <= 0.85:
            cell_dict = {}
            summary_dict = {}
            
            for summary_key in bat_dict[cell]['summary'].keys():
                if summary_key in ['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmax', 'Tmin']:
                    bat_dict[cell]['summary'].insert(0,"gradient",[get_gradient(bat_dict[cell]['summary'][summary_key], idx) for idx in range(len(bat_dict[cell]['summary'][summary_key]))])
                    
                    lower, higher = get_IQR(bat_dict[cell]['summary'], 'gradient', 3)
                    bat_dict[cell]['summary']['gradient'] = bat_dict[cell]['summary']['gradient'].apply(lambda x : x if (x > lower) and (x < higher) else None)
                    bat_dict[cell]['summary'][summary_key].loc[bat_dict[cell]['summary']['gradient'].isnull()] = None
                    bat_dict[cell]['summary'][summary_key].interpolate(inplace=True)
                    bat_dict[cell]['summary'][summary_key] = (bat_dict[cell]['summary'][summary_key] - bat_dict[cell]['summary'][summary_key].min())/(bat_dict[cell]['summary'][summary_key].max() - bat_dict[cell]['summary'][summary_key].min())
                    bat_dict[cell]['summary'].drop(columns=['gradient'], inplace=True)
                    
                summary_dict[summary_key] = bat_dict[cell]['summary'][summary_key]
            
            cycle_dict = {
                cycle: {
                    cycle_key: pd.DataFrame(bat_dict[cell]['cycles'][cycle][cycle_key])
                    for cycle_key in bat_dict[cell]['cycles'][cycle].keys()
                    }
                for cycle in range(1, len(bat_dict[cell]['cycles']))
                }            
            
            cell_dict['summary'] = summary_dict
            cell_dict['cycles'] = cycle_dict
            processed_dict[f"{cell}"] = cell_dict
    return processed_dict
    
def main():
    parser = argparse.ArgumentParser(description='Remove the outlier from the "Source/Input" and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='2017-05-12_batchdata_updated_struct_errorcorrect.pkl')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--output', '-o', default='norm_data.pkl')
    parser.add_argument('--verbose', '-v', default=False)

    args = parser.parse_args()

    if args.verbose:
        for dirname, _, filenames in os.walk(args.src):
            for filename in filenames:
                print(os.path.join(dirname, filename))

    batch_dict = utils.load_data(args.src, args.input)
    clean_dict = remove_data(batch_dict)
    utils.save_data(args.dest, args.output, clean_dict)

if __name__ == "__main__":
    main()
