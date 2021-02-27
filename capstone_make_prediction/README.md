# This folder contains everything needed to make a model prediction.
   
### These packages are used:
- `pandas`
- `pickle`
- `pycaret`
- `imblearn` (automatically installed with `pycaret`)
   
### Making a prediction with script `make_prediction`
- 1 command line argument--name of csv file containing input data
- writes new csv file to same directory as input file--contains data and prediction
- logging file `prediction_log.txt` written to current directory   
   
### Example prediction in this folder
- run script at command line `python make_prediction user_data/user_input_data.csv`
- input data `user_data/user_input_data.csv`
- result `user_data/user_input_data_with_prediction.csv` and `prediction_log.txt`   
   
### Testing with script `execute_tests`
- 3 `csv` files in `model_data` contain various types of bad data
- each file is checked for errors using the same methods used by `make_prediction`
- logging file `testing_log.txt` contains the results (errors found; data that is correct)