# Learning Exercises
The `ipynb` notebooks in this directory are homework exercises associated with a lecture, video, or reading.

### data_preparation
- `website_api.ipynb` (OMDB, Digital Public Library of America, seeclickfix; `json`,  `urllib`)
- `rss_feeds.ipynb` (NYTimes, pandas dataframe, write csv; `feedparser`)

### data_processing
- `text_data_EDA.ipynb` (feature extraction; `sklearn`, `wordcloud`)
- `finding_outlier.ipynb` (z-score, interquartile range; `np`)
- `transformations.ipynb` (log, square root, box plot, distribution plot; `np`, `seaborn`)
- `uber_data_analysis.ipynb` (datetime, histogram, bar plot, heat map; `pandas`, `seaborn`)
- `income_prediction.ipynb` (EDA, cleaning, dummies, missing data, outliers, polynomial features, PCA, logistic regression; `sklearn`)
- `working_with_strings.ipynb` (json; `pandas`)
- `time_series.ipynb` (dates, times, datetime, resample, shift, rolling window; `pandas`)

### data_transformation
- `scale_normalize.ipynb` (standardize, min-max, robust; `sklearn`)
- `missing_data.ipynb` (constant fill, forward fill, back fill, drop, interpolate; `pandas`)
- `replace_values.ipynb` (`pandas`)
- `duplicate_data.ipynb` (`pandas`)
- `train_test_split.ipynb` (linear regression; `sklearn`, `pandas`)
- `k_fold_cross_validation.ipynb` (digits dataset; logistic regression, random forest, SVC; `sklearn`)
- `imbalanced_over_sampling.ipynb` (credit card dataset; `imblearn`, `sklearn`)
- `imbalanced_under_sampling.ipynb` (credit card dataset; `imblearn`, `sklearn`)

### regression
- `boston_housing_price.ipynb` (linear, ridge, lasso, regularization; `sklearn`)
- `fertility.ipynb` (linear, ridge, lasso, regularization, cross validation; `sklearn`)
- `knn_classification.ipynb` (iris and digits datasets; k nearest neighbors; `sklearn`)
- `noisy_sine_curve.ipynb` (polynomial, ridge, lasso, regularization; `sklearn`)

### classification
- `logistic_regression.ipynb` (titanic dataset; pre-processing, modeling; `sklearn`)
- `logistic_regression_tuning.ipynb` (diabetes dataset; pre-processing, modeling; `sklearn`)
- `naive_bayes.ipynb` (spam data; `sklearn`)
- `naive_bayes_tuning.ipynb` (titanic dataset; `sklearn`)
- `knn.ipynb` (car mpg data; `sklearn`)
- `knn_tuning.ipynb` (titanic dataset; pre-processing, modeling; `sklearn`)
- `svm.ipynb` (titanic dataset; `sklearn`)
- `svm_tuning.ipynb` (iris dataset; `sklearn`)

- `binary_classification_metrics.ipynb` (`sklearn`)
- `multi_class_metrics.ipynb` (classification report, confusion matrix, precision, recall, f-score; `sklearn`)
- `multi_class_roc.ipynb` (iris dataset; `sklearn`)
- `metrics_visualizations.ipynb` (confusion matrix, ROC curve, precision-recall curve; `sklearn`)

- `digits_dataset_modeling.ipynb` (logistic regression, svm, cross validation; `sklearn`)
- `diabetes_dataset_metrics.ipynb` (logistic regression, accuracy, confusion matrix, recall, precision, classification threshold, ROC, AUC; `sklearn`)
- `iris_dataset_EDA_and_modeling.ipynb` (`sklearn`)
- `iris_dataset_bayesian.ipynb` (`sklearn`)

### trees_and_boosting
- `iris_dataset_random_forest.ipynb` (`sklearn`)
- `optimization.ipynb` (`xgboost`, `lightgbm`, `catboost`, `skopt`)
- `titanic_dataset_visualization.ipynb` (classification, regression; `graphviz`, `dtreeviz`)

### deep_learning
- `auto_mpg_regression.ipynb` (normalization layer, linear and non-linear transformation layers; `tensorflow`)
- `fashion_image_classification.ipynb` (flatten and dense layers; `tensorflow`)

### anomaly_detection
- `general_techniques.ipynb` (box plot, histogram, clustering; `scipy`, `pyod`)
- `credit_card_component_analysis.ipynb` (PCA, random projection, dictionary learning, ICA; `sklearn`)
- `sales_feature_selection.ipynb` (low variance filter, high correlation filter, random forest feature selection, recursive feature elimination, forward feature selection; `sklearn`)

### recommendation_systems
- `song_recommender.ipynb` (popularity based, collaborative filtering; `sklearn`)
- `movie_recommender.ipynb` (collaborative filtering, Azure Databricks; `pyspark.ml`, `pyspark.sql`)

### time_series_analysis
- `general_analysis_techniques.ipynb` (resample, autocorrelation, de-trend, seasonal decomposition, auto regression, residuals, ARIMA; `statsmodels`, `sklearn`)
- `autoregressive_packages.ipynb` (`statsmodels`, `pmdarima`, `scipy`)
- `LSTM_neural_network.ipynb` (stock market data; `keras`, `tensorflow`)
