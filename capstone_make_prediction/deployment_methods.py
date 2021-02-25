import pandas as pd
import pickle

from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from pycaret.classification import *

def check_numerical_values(df_to_check, log_file):
#def check_numerical_values(df_to_check):
    _ = log_file.write('\nBEGINNING CHECK ON NUMERICAL VALUES')
    found_error = False
    
    # check for non-numerical columns and cast to int64
    # check for null values
    df_data_types = df_to_check.dtypes
    for col in list(df_to_check.columns):
        if df_data_types[col] == 'object':
            _ = log_file.write('\nERROR  Column {} contains values that are not numerical.'.format(col))
        elif pd.isnull(df_to_check[col]).sum() > 0:
            _ = log_file.write('\nERROR  Column {} contains NULL values.'.format(col))
        else:
            df_to_check[col] = df_to_check[col].astype('int64')
    
    # check for negative values
    list_negative_cols = []
    df_min_values = df_to_check.describe().transpose()['min'].sort_values()
    for icol in range(len(df_min_values)):
        if df_min_values.iloc[icol] < 0:
            if df_min_values.index[icol] != 'VD_hydrology':
                list_negative_cols.append(df_min_values.index[icol])
                found_error = True
        else:
            break
    
    if len(list_negative_cols) > 0:
        _ = log_file.write('\nERROR  Negative values not permitted in these columns: {}.'.format(list_negative_cols))
    
    _ = log_file.write('\nCOMPLETED CHECK ON NUMERICAL VALUES')
    
    return found_error


def check_user_column_names(df_to_check, log_file):
    _ = log_file.write('\nBEGINNING CHECK OF COLUMN NAMES')
    found_error = False

    req_cols = ['elevation', 'aspect', 'slope', 'HD_hydrology', 'VD_hydrology', 'HD_roadways', 'HD_fire_points',
                'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
                'wilderness_index', 'soil_index']
    
    user_cols = list(df_to_check.columns)
        
    for check_col in user_cols:
        if req_cols.count(check_col) == 0 and check_col.find('.') < 0:
            _ = log_file.write('\nERROR  Column {}: present in csv file but is not needed by model.'.format(check_col))
        
        if check_col.find('.') > 0:
            if check_col.split('.')[0] in req_cols:
                _ = log_file.write('\nERROR  Required column {}: present more than once in csv file. Please enter it only once.'.format(check_col.split('.')[0]))
                found_error = True
    
    for check_col in req_cols:
        check_count = user_cols.count(check_col)

        if check_count == 0:
            _ = log_file.write('\nERROR  Required column {}: missing from csv file.'.format(check_col))
            found_error = True
        else:
            _ = log_file.write('\nRequired column {}: present in csv file.'.format(check_col))
    
    _ = log_file.write('\nCOMPLETED CHECK OF COLUMN NAMES')
    
    return found_error


def check_values_wilderness_soil_indices(df_to_check, log_file):
    _ = log_file.write('\nBEGINNING CHECK wilderness_index AND soil_index VALUES')
    found_error = False
    
    wilderness_range = {'min':0 ,
                        'max':3,
                        'name':'wilderness_index'}
    soil_range = {'min':1 ,
                  'max':40,
                  'name':'soil_index'}
    
    error_wilderness = check_category_data_values(df_to_check, wilderness_range, log_file)
    error_soil       = check_category_data_values(df_to_check, soil_range, log_file)
    
    _ = log_file.write('\nCOMPLETED CHECK wilderness_index AND soil_index VALUES')
    return (error_wilderness or error_soil)


def check_category_data_values(df_to_check, valid_range, log_file):
    found_error = False
    
    if valid_range['name'] in df_to_check.columns  and  df_to_check[valid_range['name']].dtype != 'object':
        user_max = df_to_check[valid_range['name']].max()
        user_min = df_to_check[valid_range['name']].min()
        
        if (user_max < valid_range['min'] or user_max > valid_range['max'] or
            user_min < valid_range['min'] or user_min > valid_range['max']):
            _ = log_file.write('\nERROR  Check on column {}: there are values outside valid range of {} to {}; found min and max: {}  {}' \
                               .format(valid_range['name'], valid_range['min'], valid_range['max'], user_min, user_max))
            found_error = True
        else:
            _ = log_file.write('\nCheck on column {}: all values are valid.'.format(valid_range['name']))
        
        return found_error

        
def check_input_data_csv_file(filename, log_file):
    _ = log_file.write('\n\nBEGINNING CHECK OF FILE {}'.format(filename))
    
    df_to_check = pd.read_csv(filename)
    
    error_numerical_values = check_numerical_values(df_to_check, log_file)
    error_col_names = check_user_column_names(df_to_check, log_file)
    error_category_indices = check_values_wilderness_soil_indices(df_to_check, log_file)
    
    _ = log_file.write('\nCOMPLETED CHECK OF FILE {}'.format(filename))
    return (error_col_names or error_category_indices or error_numerical_values)


def execute_all_unit_tests():
    log_file = open('testing_log.txt', 'w')
    
    for test_file in ['model_data/data_check_column_names.csv' ,
                      'model_data/data_check_category_indices.csv' ,
                      'model_data/data_check_numerical_values.csv']:
        found_error = check_input_data_csv_file(test_file, log_file)
    
    log_file.close()


def make_prediction(input_data_filename, log_filename):
    log_file = open(log_filename, 'w')
    soil_index_filename = 'model_data/wrangled_soil_type.pkl'
    tuned_model_filename = 'model_data/tuned_decision_tree_model.pkl'
    
    # check input data
    found_data_errors = check_input_data_csv_file(input_data_filename, log_file)
    if found_data_errors:
        _ = log_file.write('\n\nFOUND ERRORS IN DATA. PREDICTION WILL NOT BE MADE.')
        log_file.close()
        return False
    else:
        _ = log_file.write('\n\nNO ERRORS FOUND IN DATA.')
    
    # read input data into dataframe
    _ = log_file.write('\n\nREADING INPUT CSV FILE INTO DATAFRAME.')
    df_user_input = pd.read_csv(input_data_filename)
    for col in list(df_user_input.columns):
        df_user_input[col] = df_user_input[col].astype('int64')
    
    # read file with information on soil index
    _ = log_file.write('\n\nPREPROCESSING DATA.')
    df_soil = pd.read_pickle(soil_index_filename)
    df_soil = df_soil.rename(columns={'study_code':'soil_index'})
    df_soil['climatic_zone'] = df_soil['climatic_zone'].astype('int64')
    df_soil['geologic_zone'] = df_soil['geologic_zone'].astype('int64')
    df_soil['both_zones']    = df_soil['both_zones'].astype('int64')
    
    # add column for clustering of soil_index categories
    soil_cols_to_use = ['soil_index', 'geologic_zone']
    df_input_data = pd.merge(df_user_input, df_soil[soil_cols_to_use], how='left',
                             left_on='soil_index', right_on='soil_index')
    
    # read tuned decision tree model
    _ = log_file.write('\n\nREADING MODEL.')
    tuned_dt_model = pickle.load(open(tuned_model_filename, 'rb'))
    
    # make prediction
    _ = log_file.write('\n\nMAKING PREDICTION.')
    prediction = predict_model(tuned_dt_model, data = df_input_data)
    
    # add prediction column to user dataframe
    _ = log_file.write('\n\nADDING PREDICTION RESULT TO DATAFRAME.')
    df_user_input['predicted_cover_index'] = prediction['Label']
    
    # add column for cover name to user datframe
    cover_names = ['spruce_fir', 'lodgepole_pine', 'ponderosa_pine', 'cottonwood_willow', 'aspen', 'douglas_fir', 'krummholz']
    get_target_name_from_integer = lambda i : cover_names[i-1]
    df_user_input['predicted_cover_name'] = df_user_input['predicted_cover_index'].apply(get_target_name_from_integer)
    
    # write csv file with input data and prediction
    name_split = input_data_filename.split('.')
    results_file_name = name_split[0] + '_with_prediction.' + name_split[1]
    _ = log_file.write('\n\nWRITING CSV FILE {}.'.format(results_file_name))
    df_user_input.to_csv(results_file_name, index=False, header=True)
    
    log_file.close()
    return True
    
    

