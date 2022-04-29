#%% DATABASE: CINC2017
import scipy.io
import pandas as pd
import pathlib


#%% Create training files 
training_files = list(pathlib.Path('Physionet2017_data/training').glob('*.mat'))
file_class = pd.read_csv('Physionet2017_data/training/REFERENCE.csv',names=['ID','Class'])

training_df = pd.DataFrame(data=None, columns=['Label','Signal'])

for file in training_files:
    
    ref = file.name[:-4]
    signal_class = file_class.loc[file_class['ID'] == ref].iloc[0,1]
    signal = scipy.io.loadmat(file)['val'][0]
    
    training_df = training_df.append(pd.Series({'Label': signal_class, 'Signal': signal}, name=ref))

#%% Create validation files
validation_files = list(pathlib.Path('Physionet2017_data/validation').glob('*.mat'))
file_class = pd.read_csv('Physionet2017_data/validation/REFERENCE-v3.csv',names=['ID','Class'])

validation_df = pd.DataFrame(data=None, columns=['Label','Signal'])

for file in validation_files:
    
    ref = file.name[:-4]
    signal_class = file_class.loc[file_class['ID'] == ref].iloc[0,1]
    signal = scipy.io.loadmat(file)['val'][0]
    
    validation_df = validation_df.append(pd.Series({'Label': signal_class, 'Signal': signal}, name=ref))

#%% Save to pckl files
training_df.to_pickle('raw_training_df.pckl')
validation_df.to_pickle('raw_validation_df.pckl')
