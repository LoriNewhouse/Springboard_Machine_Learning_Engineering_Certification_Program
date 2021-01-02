# Project Description
This is a classification project in environmental science.
The dataset is for the Roosevelt National Forest, Colorado, USA. It is available from the University of California--Irvine Machine Learning Repository:   
http://archive.ics.uci.edu/ml/datasets/Covertype

- number observations: 581,012
- multi-class target (`cover_name`): 7 values
- 10 numerical features
- non-ordinal, categorical feature: 4 categories
- non-ordinal, high cardinality categorical feature: 40 categories

There is significant class imbalance with the 5 smallest classes present at < 7% each, totaling 15%.
![target hist](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/target_histogram.PNG)

Here is a histogram of the high cardinality categorical feature, `soil_index`. Five encoding techniques were investigated, including 3 domain informed clustering approaches.   
![soil hist](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/soil_index_histogram.PNG)

# Modeling Results
Initial modeling investigation was done using`sklearn` with follow-up investigations using `pycaret`. The modeling work is still in progress. Here are 2 plots of model performance:
- x-axis: encoding technique
- y-axis: metric value
- colored dots: model
### overall accuracy
![model_metric](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/soil_encoding_accuracy.PNG)

### metrics for 2 target classes (fraction present in training set)
![class_metrics](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/soil_encoding_by_class.PNG)

# Notebooks
Here is a summary of the notebooks:
- `data_wrangling_and_EDA_update_soil.ipynb`   
visualizations, pre-processing, pickle results
   
   
- `modeling_first_pass.ipynb`   
logistic regression, SVM, naive bayes
   
   
- `modeling_trees.ipynb`   
decision tree, random forest, gradient boosting, XGBoost, LightGBM, CatBoost
   
   
- `modeling_feature_soil.ipynb`   
5 encoding techniques for `soil_index`   
5 models run for each using `pycaret`
   
   
- `pycaret_results_summary.ipynb`   
plots of model prediction metrics for soil encoding investigation
   
   
- `imblearn_investigation.ipynb`   
get familiar with `imblearn` to address class imbalance   
preliminary work to incorporating `imblearn` in `pycaret` processing pipeline

