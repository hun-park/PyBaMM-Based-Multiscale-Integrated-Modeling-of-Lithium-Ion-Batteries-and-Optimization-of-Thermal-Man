# %% [code]
import argparse
import os
import pickle
import tensorflow as tf

    
def main():
    parser = argparse.ArgumentParser(description='Scale data from the mems and save it as a .pkl file in the Destination directory.')
    parser.add_argument('--src', '-s', default='/working/data')
    parser.add_argument('--input', '-i', default='small-cnn-model.h5')
    parser.add_argument('--output', default='light-cnn-model.tflite')
    parser.add_argument('--dest', '-d', default='/working/data')
    parser.add_argument('--assets', default='/working/srcs/assets')
    parser.add_argument('--verbose', '-v', default=False)
    parser.add_argument('--opt', default='SIZE')

    args = parser.parse_args()
    model = tf.keras.models.load_model(os.path.join(args.src, args.input))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    opt = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] if args.opt == "SIZE" else [tf.lite.Optimize.DEFAULT]
    converter.optimizations = opt
    
    q_model = converter.convert()
    open(os.path.join(args.dest, args.output), "wb").write(q_model)

if __name__ == "__main__":
    main()
