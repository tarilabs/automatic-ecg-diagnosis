# %% Import packages
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
import h5py
from nyoka import KerasToPmml

parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
parser.add_argument('--tracings', default="./data/ecg_tracings.hdf5",  # or date_order.hdf5
                    help='HDF5 containing ecg tracings.')
parser.add_argument('--model', default="./dnn_predicts/model.hdf5",  # or model_date_order.hdf5
                    help='file containing training model.')
parser.add_argument('--output_file', default="./dnn_output.csv",  # or predictions_date_order.csv
                    help='output csv file.')
parser.add_argument('-bs', type=int, default=32,
                    help='Batch size.')

args, unk = parser.parse_known_args()
if unk:
    warnings.warn("Unknown arguments:" + str(unk) + ".")


# %% Import

# Import model
model = load_model(args.model, compile=False)
model.compile(loss='binary_crossentropy', optimizer=Adam())

pmmlObj=KerasToPmml(model)
pmmlObj.export(open('model.pmml','w'),0)

print("Model saved")