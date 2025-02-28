# %%
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import utils

def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTM model on data and save the results.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='V_test.pkl I_test.pkl T_test.pkl y_test.pkl')
    parser.add_argument('--iscumm', default='True')
    parser.add_argument('--verbose', '-v', default="False")
    parser.add_argument('--test', default="False")
    parser.add_argument('--cell', default="b1c18")
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--assets', default='/working/srcs/assets')
    parser.add_argument('--models', default='big-cnn-model.h5 small-cnn-model.h5')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--GPU', type=int, default=-1)
    parser.add_argument('--predict', default="SoC")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)
    os.environ['TF_CPP_MIN_LOG_LEVEL']="2" if eval(args.verbose) else "3"
    tf.random.set_seed(args.seed)
    tf.debugging.set_log_device_placement(eval(args.verbose))
    
    [V_test, I_test, T_test, labels] = utils.load_data(args.src, args.input)
    data = np.concatenate((V_test, I_test, T_test), axis=1)
    
    for model in args.models.split():
        imgs = data
        if os.path.splitext(model)[1] == ".h5":
            locals()[str(model)+"result"] = utils.inference(os.path.join(args.src, model), imgs, args.verbose);
        else:
            locals()[str(model)+"result"] = utils.q_inference(os.path.join(args.src, model), imgs, args.chunk, args.verbose);
        if args.predict == 'SoC':
            locals()[str(model)+"result"] = locals()[str(model)+"result"]
    
    for idx in range(len(args.models.split())):
        plt.plot(labels, 'r--', label='truth')
        plt.plot(locals()[str(args.models.split()[idx])+"result"], 'y-', label='predict')
        labels_lower = [label*0.975 for label in labels]; labels_upper = [label*1.025 for label in labels];
        plt.fill_between(range(len(labels)), np.array(labels_lower).reshape(-1), np.array(labels_upper).reshape(-1), color='red', alpha=0.3)
        
        plt.legend(loc='upper left')
        plt.title(f'{os.path.splitext(args.models.split()[idx])[0]}_{args.cell}')
        if eval(args.test):
            plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_{args.cell}_result.jpg'))
            if eval(args.verbose): print(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_{args.cell}_result.jpg'))
        else:
            plt.savefig(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_result.jpg'))
            if eval(args.verbose): print(os.path.join(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_result.jpg'))
        plt.clf()
    utils.save_data(args.assets, f'{os.path.splitext(args.models.split()[idx])[0]}_{args.cell}_result.pkl', locals()[str(model)+"result"])

if __name__ == "__main__":
    main()
