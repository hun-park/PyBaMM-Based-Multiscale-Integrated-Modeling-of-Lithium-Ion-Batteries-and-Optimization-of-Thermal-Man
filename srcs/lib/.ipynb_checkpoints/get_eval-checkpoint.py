# +
import argparse
import os
import utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import get_imgs

    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='pick_data.pkl pick_labels.pkl')
    parser.add_argument('--algorithm', default='Stack')
    parser.add_argument('--iscumm', default='True')
    parser.add_argument('--verbose', '-v', default="False")
    parser.add_argument('--model', default='big-cnn-model.h5')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--fraction', type=int, default=60)
    parser.add_argument('--size', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)
    os.environ['TF_CPP_MIN_LOG_LEVEL']="2" if eval(args.verbose) else "3"
    tf.random.set_seed(args.seed)
    tf.debugging.set_log_device_placement(eval(args.verbose)) # display an allocated computing device name as False

    [raws, labels] = utils.load_data(args.src, args.input)
    
    imgs = np.array(get_imgs.encode_data(raws, args.algorithm, args.iscumm, args.verbose)/255.0, np.float32)
    imgs = imgs[0:args.size]; labels = labels[0:args.size]
    
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction/100
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    model = tf.keras.models.load_model(os.path.join(args.src, args.model))

    loss, mse, mape = model.evaluate(imgs, labels, batch_size=16, verbose=2)

    print(f'mse : {mse}, mape : {mape}')

if __name__ == "__main__":
    main()
