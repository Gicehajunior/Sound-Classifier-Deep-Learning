import os
import sys
import time
import glob
import pandas
import numpy
import librosa
from pathlib import Path
from librosa import display
from IPython.display import HTML, Audio, display_html
from IPython.display import display as display_ipython
from sklearn.preprocessing import LabelEncoder

#from tensorflow & keras API
import tensorflow

print(tensorflow.__version__)


#date & time
from datetime import datetime

trains_dataset_path = 'input/silero-audio-classifier/train'
validation_dataset_path = 'input/silero-audio-classifier/val'
train_metadata_path = "input/silero-audio-classifier/train.csv"  
test_metadata_path = "input/silero-audio-classifier/sample_submission.csv"

output_dir = 'input/output/'
pandas.set_option('display.max_colwidth', 500)

def read_metadata(train_metadata_path, test_metadata_path):
  pandas_dataframe_train = pandas.read_csv(train_metadata_path)
  pandas_dataframe_test = pandas.read_csv(test_metadata_path)
  
  return pandas_dataframe_train.iloc[:1000], pandas_dataframe_test.iloc[:1000]


#### extract features for each audio using librosa library ####
def extract_features(file_name):
      try:

        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = numpy.mean(mfccs.T,axis=0)

      except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 

      return mfccsscaled

#reading our dataset from the directory and applying extraction.
def load_dataset_extract_features(trains_dataset_path,validation_dataset_path, pandas_dataframe_train, pandas_dataframe_test):
    train_features = []
    test_features = []
    print('________________________________________________________________________________________________________')
    train_iter_time = datetime.now()
    print('Ready to itterate the train dataframe and append, Start time:', train_iter_time)
    for index, row in pandas_dataframe_train.iterrows():
        class_labels = row['target']
        #print(class_labels)

        train_filenames = os.path.join(os.path.abspath(trains_dataset_path),str(row["wav_path"]))
        #print(train_filenames)
        
        train_wav_dataset = extract_features(train_filenames)
        
        train_features.append([train_wav_dataset, class_labels])
    print('Processing the training dataframe and appending took a duration of, End time:', datetime.now() - train_iter_time)
    
    print('________________________________________________________________________________________________________')
    
    
    test_itr_time = datetime.now()
    print('Ready to itterate the test dataframe and append, Start time:', test_itr_time)
    
    for index, row in pandas_dataframe_test.iterrows():
        class_labels = row['target']
        #print(class_labels)
        
        validation_filenames = os.path.join(os.path.abspath(validation_dataset_path),str(row["wav_path"]))
        #print(validation_filenames)
        
        test_wav_dataset = extract_features(validation_filenames)
        
        test_features.append([test_wav_dataset, class_labels])
    print('Processing the testing dataframe and appending took a duration of, End time:', datetime.now() - test_itr_time)
       
    #print(train_features)
    #print(test_features)
    print('________________________________________________________________________________________________________')
    train_features_dataframe = pandas.DataFrame(train_features, columns=['features', 'class_labels'])  
    test_features_dataframe = pandas.DataFrame(test_features, columns=['features', 'class_labels'])
    return train_features_dataframe, test_features_dataframe

#conversion of the data into numpy arrays in form of lists
#converting to tensors
def further_processing(training_dataset, training_labels, testing_dataset, testing_labels):
    training_dataset_lists = numpy.array(training_dataset.tolist())
    training_labels_list = numpy.array(training_labels.tolist())
    #print(training_dataset_lists)
    
    testing_dataset_list = numpy.array(testing_dataset.tolist())
    testing_labels_list = numpy.array(testing_labels.tolist())
    

    # Encode the classification labels
    lable_encoder = LabelEncoder()
    categorized_labels = tensorflow.keras.utils.to_categorical(lable_encoder.fit_transform(training_labels_list))
    
    return training_dataset_lists, training_labels_list, testing_dataset_list, testing_labels_list, categorized_labels

def create_model(categorized_labels):
    
    num_rows = 40
    num_columns = 174
    num_channels = 1

    labels_shape = categorized_labels.shape[1]
    filter_size = 2

        # Construct model 
    model = tensorflow.keras.models.Sequential()

    model.add(tensorflow.keras.layers.Dense(256, input_shape=(40,)))
    model.add(tensorflow.keras.layers.Activation(activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.5))

    model.add(tensorflow.keras.layers.Dense(256))
    model.add(tensorflow.keras.layers.Activation(activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.5))

    model.add(tensorflow.keras.layers.Dense(labels_shape))
    model.add(tensorflow.keras.layers.Activation(activation='softmax'))
    
    return model

def train_model(created_model, training_dataset_list, training_labels_list, testing_dataset_list, testing_labels_list, output_dir):
    
    print(created_model.summary())
    created_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    epochs = 100
    batch_size = 32
    
    model_filepath = output_dir +'/saved_models/weights.best.basic_mlp.hdf5'
    
    #to cater for re-training another set of data again.
    ModelCheckPointer = tensorflow.keras.callbacks.ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)
    
    
    model = created_model.fit(training_dataset_list, training_labels_list, batch_size=batch_size, epochs=epochs, validation_data=(testing_dataset_list, testing_labels_list), callbacks=[ModelCheckPointer], verbose=1 )
    
    return model
    
####function calls####
#displays the dataframe of the metadata file
pandas_dataframe_train, pandas_dataframe_test = read_metadata(train_metadata_path, test_metadata_path)
#print(pandas_dataframe_train)
#print(pandas_dataframe_test)

#loading the dataset from the folders & extracting the features from the dataset, one by one
train_features_dataframe, test_features_dataframe = load_dataset_extract_features(trains_dataset_path, validation_dataset_path, pandas_dataframe_train, pandas_dataframe_test)
print('Finished feature extraction from', len(pandas_dataframe_train), 'files')
print('Finished feature extraction from', len(pandas_dataframe_test), 'files')

#extracting from the dataframes returned
training_dataset, training_labels, testing_dataset, testing_labels = train_features_dataframe.features, train_features_dataframe.class_labels, test_features_dataframe.features, test_features_dataframe.class_labels

#conversion of the training and validation datasets to lists.
training_dataset_list, training_labels_list, testing_dataset_list, testing_labels_list, categorized_labels = further_processing(training_dataset, training_labels, testing_dataset, testing_labels)

#create the model
create_model = create_model(categorized_labels)
print("Number of weights after calling the model:", len(create_model.weights))
#display the archtecture of our model
#create_model.summary()

if create_model:
    print('Model build successfully.')
else:
    print('Model refused to build successfully.')
print(create_model)
#### Training our model ####
training_start_time = datetime.now()
print(training_start_time)

model = train_model(create_model, training_dataset_list, training_labels_list, testing_dataset_list, testing_labels_list, output_dir)

print(model.history)

Duration = datetime.now() - training_start_time
print("The training of the model took a duration of:", Duration)

model.save(output_dir)