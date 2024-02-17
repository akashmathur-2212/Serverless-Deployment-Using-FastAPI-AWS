import pandas as pd
import numpy as np
import pickle
from io import StringIO

def load_pickle(filename):
    with open(filename, 'rb') as file: # read file
        contents = pickle.load(file) # load contents of file
    return contents

with open("../artifacts/model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)


def feature_engineering_encoding(data):
    """
    
    Put your Feature Engineering and Encoding Code
    For this example, I am not doing any feature engineering
    
    """
    
# function to create a new column 'Bmi'
def process_label(row):
    if row['Predicted Label'] == 1:
        return 'Sepsis status is Positive'
    elif row['Predicted Label'] == 0:
        return 'Sepsis status is Negative'
    

def return_columns():
    # create new columns
    new_columns =  ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    return new_columns

# function to create a new column 'Bmi'
def process_label(row):
    if row['Predicted Label'] == 1:
        return 'Diabetic'
    elif row['Predicted Label'] == 0:
        return 'Non Diabetic'
    

def make_prediction(data, model):
    new_columns = return_columns() 
    dict_new_old_cols = dict(zip(data.columns, new_columns)) # create a dict of original columns and new columns
    data = data.rename(columns=dict_new_old_cols)
    # feature_engineering_encoding(data) # create new features
    # make prediction
    label = model.predict(data) # make a prediction
    probs = model.predict_proba(data) # predit sepsis status for inputs
    return label, probs.max()


def process_json_csv(contents, file_type, valid_formats):

    # Read the file contents as a byte string
    contents = contents.decode()  # Decode the byte string to a regular string
    new_columns = return_columns() # return new_columns
    # Process the uploaded file
    if file_type == valid_formats[0]:
        data = pd.read_csv(StringIO(contents)) # read csv files
    elif file_type == valid_formats[1]:
        data = pd.read_json(contents) # read json file
    
    dict_new_old_cols = dict(zip(data.columns, new_columns)) # get dict of new and old cols
    data = data.rename(columns=dict_new_old_cols) # rename colums to appropriate columns
    return data

        
def output_batch(data1, labels):
    data_labels = pd.DataFrame(labels, columns=['Predicted Label']) # convert label into a dataframe
    data_labels['Predicted Label'] = data_labels.apply(process_label, axis=1) # change label to understanding strings
    results_list = [] # create an empty lits
    x = data1.to_dict('index') # convert  datafram into dictionary
    y = data_labels.to_dict('index') # convert  datafram into dictionary
    for i in range(len(y)):
        results_list.append({i:{'inputs': x[i], 'output':y[i]}}) # append input and labels

    final_dict = {'results': results_list}
    return final_dict