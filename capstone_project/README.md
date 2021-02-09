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

# Model Development
Initial modeling work was done using`sklearn` followed by detailed work using `pycaret`. Here are 3 plots of model performance:
- x-axis: encoding technique; minority classes over-sampling technique (factor=3 or 1.5)
- y-axis: metric value
- colored dots: model
### overall accuracy
![model_accuracy](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/model_accuracy.PNG)

### metrics for 2 target classes (fraction present in training set)
![model_by_class](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/model_by_class.PNG)

### impact of over-sampling 5 smallest classes
![imblearn_results](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/imblearn_results.PNG)

# Model Tuning
Decision Tree and CatBoost models were tuned using 3-fold cross-validation with 15 iterations. Both used minority class (5 smallest) oversampling at factor=3. Optimization of 2 different metrics was investigated:
- f1 macro, average of all 7 classes, each class weight=1
- f1 custom, 5 smallest classes weight=3, 2 largest weight=1
- f1 macro 2, same as f1 macro but used a different search grid because 1st grid gave poor performing model
![tune_by_class](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/tune_by_class.PNG)

# Model Finalization
Decision Tree model was finalized by fitting with all the data and the tuned hyper-parameters.
![finalize_results](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/finalize_results.PNG)
![finalized_by_class](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/capstone_project/images/finalized_by_class.PNG)

# Notebooks
Here is a summary of the notebooks:
- `data_wrangling_and_EDA_update_soil.ipynb`   
visualizations, pre-processing, pickle results
   
   
- `modeling_first_pass.ipynb`   
logistic regression, SVM, naive bayes
   
   
- `modeling_trees.ipynb`   
decision tree, random forest, gradient boosting, XGBoost, LightGBM, CatBoost
   
   
- `modeling_pycaret_soil.ipynb`   
5 encoding techniques for `soil_index`   
5 models run for each using `pycaret`
   
   
- `imblearn_investigation.ipynb`   
get familiar with `imblearn` to address class imbalance   
preliminary work to incorporating `imblearn` in `pycaret` processing pipeline
   
   
- `modeling_pycaret_imblearn-----.ipynb`   
oversample with factor = 3 and 1.5   
3 models: decision tree, XGBoost, CatBoost
   
   
- `pycaret_results_summary_plots.ipynb`   
plots of performance metrics and class metrics for all models     
   
   
- `tuning_cat_boost_f1_macro.ipynb`   
tune CatBoost model with metric f1 macro     
similar notebook for tuning with metric f1 custom are not included here   
similar notebooks for tuning decision tree are not included here   
   
   
- `tuning_results_summary_plots.ipynb`   
plots of performance metrics and class metrics for tuned decision tree and CatBoost models   
plots for performance of finalized decision tree model fit with all data     
