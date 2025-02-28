# %% [code]
import os
import cv2
import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import tensorflow as tf
import time
import pyts.image


def load_data(src, input_file):
    if len(input_file.split()) - 1:
        return_list = []
        for file in input_file.split():
            return_list.append(pd.read_pickle(os.path.join(src, file)))
        return return_list
    else:
        return pd.read_pickle(os.path.join(src, input_file))

def save_data(dest, output_file, data):
    if len(output_file.split()) - 1:
        for idx in range(len(output_file.split())):
            with open(os.path.join(dest, output_file.split()[idx]), 'wb') as fp:
                pickle.dump(data[idx], fp, protocol=4)
    else:
        with open(os.path.join(dest, output_file), 'wb') as fp:
            pickle.dump(data, fp, protocol=4)

def save_plot(x_axises, x_labels, y_axises, y_labels, dest, files, colors=['red', 'green', 'blue'], iter=1):
    plt.subplots(constrained_layout=True)
    sample_len = min(3, len(y_axises))
    for idx in range(sample_len):
        locals()['fig'+str(idx)] = plt.subplot(sample_len, 1, idx+1)
        locals()['fig'+str(idx)].plot(x_axises.pop(0), y_axises.pop(0), color=colors[idx%3])
        locals()['fig'+str(idx)].set_xlabel(x_labels.pop(0))
        locals()['fig'+str(idx)].set_ylabel(y_labels.pop(0))
    plt.savefig(os.path.join(dest, f'{files}_{iter}.jpg'))
    print(os.path.join(dest, f'{files}_{iter}.jpg'))
    if len(y_axises):
        iter += 1
        return save_plot(x_axises, x_labels, y_axises, y_labels, dest, files, ['red', 'green', 'blue'], iter)

def save_hist(y_axises, y_labels, dest, files, iter=1):
    plt.subplots(constrained_layout=True)
    sample_len = min(3, len(y_axises))
    for idx in range(sample_len):
        plt.subplot(sample_len, 1, idx+1)
        plt.hist(y_axises.pop(0))
        plt.title(y_labels.pop(0))
    plt.savefig(os.path.join(dest, f'{files}_{iter}.jpg'))
    print(os.path.join(dest, f'{files}_{iter}.jpg'))
    if len(y_axises):
        iter += 1
        return save_hist(y_axises, y_labels, dest, files, iter)

def scale(standard, raw, algorithm):
    voltage = np.concatenate([value['V'].to_numpy() for key, value in standard.items() if 'V' in value.columns])
    current = np.concatenate([value['I'].to_numpy() for key, value in standard.items() if 'I' in value.columns])
    temperature = np.concatenate([value['T'].to_numpy() for key, value in standard.items() if 'T' in value.columns])
    cumulated = np.concatenate([value['Idt'].to_numpy() for key, value in standard.items() if 'Idt' in value.columns])
    capacity = np.concatenate([value['Q'].to_numpy() for key, value in standard.items() if 'Q' in value.columns])    
    
    min_voltage = np.min(voltage); max_voltage = np.max(voltage);
    min_current = np.min(current); max_current = np.max(current);
    min_temperature = np.min(temperature); max_temperature = np.max(temperature);
    min_cumulated = np.min(cumulated); max_cumulated = np.max(cumulated);
    min_capacity = np.min(capacity); max_capacity = np.max(capacity);  
    
    for cycle in raw.keys():            
        if algorithm in ['Log-bounded Min Max', 'LMM']:
            raw[cycle]['V'] = np.clip(
                np.log((1 + raw[cycle]['V']) / (1 + min_voltage)) * (1 / np.log((1 + max_voltage) / (1 + min_voltage))),
                0, 1)
            raw[cycle]['I'] = np.clip(
                np.log((1 + raw[cycle]['I']) / (1 + min_current)) * (1 / np.log((1 + max_current) / (1 + min_current))),
                0, 1)
            raw[cycle]['T'] = np.clip(
                np.log((1 + raw[cycle]['T']) / (1 + min_temperature)) * (1 / np.log((1 + max_temperature) / (1 + min_temperature))),
                0, 1)
            raw[cycle]['Idt'] = np.clip(
                np.log((1 + raw[cycle]['Idt']) / (1 + min_cumulated)) * (1 / np.log((1 + max_cumulated) / (1 + min_cumulated))), 
                0, 1)
            raw[cycle]['Q'] = np.clip(
                np.log((1 + raw[cycle]['Q']) / (1 + min_capacity)) * (1 / np.log((1 + max_capacity) / (1 + min_capacity))), 
                0, 1)
        
        elif algorithm in ['Min Max', 'MM']:
            raw[cycle]['V'] = (raw[cycle]['V'] - min_voltage) / (max_voltage - min_voltage)
            raw[cycle]['I'] = (raw[cycle]['I'] - min_current) / (max_current - min_current)
            raw[cycle]['T'] = (raw[cycle]['T'] - min_temperature) / (max_temperature - min_temperature)
            raw[cycle]['Idt'] = (raw[cycle]['Idt'] - min_cumulated) / (max_cumulated - min_cumulated)
            raw[cycle]['Q'] = (raw[cycle]['Q'] - min_capacity) / (max_capacity - min_capacity)
        
        else:
            pass
    return raw

def encode(series, time, algorithm='Recurrent Plot'):
    if algorithm in ['Recurrent Plot', 'RP']:
        series = series[time]
        length = series.size
        repeat = np.repeat(series[None, :], length, axis=0)
        recurr = np.floor(np.abs(repeat - repeat.T))
        recurr[recurr>255] = 255
        return recurr
    
    elif algorithm in ['GASF']:
        gasf = pyts.image.GramianAngularField(image_size=32, method="summation")
        series_s = gasf.fit_transform(series[time].reshape(1, -1))
        
        return series_s[0]
    
    elif algorithm in ['GADF']:
        gadf = pyts.image.GramianAngularField(image_size=32, method="difference")
        series_d = gadf.fit_transform(series[time].reshape(1, -1))
        
        return series_d[0]
    
    elif algorithm in ['GAF']:
        gasf = pyts.image.GramianAngularField(image_size=32, method="summation")
        series_s = gasf.fit_transform(series[time].reshape(1, -1))
        
        gadf = pyts.image.GramianAngularField(image_size=32, method="difference")
        series_d = gadf.fit_transform(series[time].reshape(1, -1))
        
        gaf = series_s + series_d
        
        return gaf[0]
    
    elif algorithm in ['Stack']:
        return np.array(np.vstack(series[time:time+32])*255, np.uint8)

def plot2img(*plots):
    img = cv2.merge(plots)
    return img

def inference(model, data, verbose):
    model = tf.keras.models.load_model(model)
    vb = 2 if eval(verbose) else 1
    predictions = model.predict(data, verbose=vb)
    
    return predictions
        
def q_inference(q_model, data, chunk, verbose):
    data = np.reshape(data, (-1, 1, chunk, chunk, 3))
    model = tf.lite.Interpreter(q_model)
    model.allocate_tensors()
    
    input_index = model.get_input_details()[0]['index']
    output_index = model.get_output_details()[0]['index']

    predictions = []
    
    sum_time = 0
    for datum in tqdm.tqdm(data):
        model.set_tensor(input_index, datum)
        
        start_time = time.time()
        model.invoke()
        elapsed = time.time() - start_time
        sum_time += elapsed
        
        output = model.get_tensor(output_index)
        predictions.append(output)
        
        if eval(verbose): print(f"---{elapsed}s seconds---\ttotal: ---{sum_time}s seconds---")
    
    return np.array(predictions).flatten()
