# Mini Projects
The `ipynb` notebooks in this directory are final projects associated with a technical skill unit. They are more complicated, real-world application driven tasks than the homework style work in the `learning_exercises` directory. Some were done on a cloud provider (Paperspace) or Databricks. Each took 1-3 days to complete. Some are reading large `csv` files which are not present in the directory. These `csv`'s are larger than the GitHub 50MB limit (even after zipping.)

#### data_preparation: `quandl_website_api.ipynb`
- financial market data
- create dictionary; calculate max, min, avg, changes
- `requests`, `json`   
![mp_data_prep](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_data_prep.PNG)

#### data_processing: `pandas_data_wrangling.ipynb`
- 3 csv files with movies data
- select, query, sort, aggregate, merge, plot   
![mp_data_process_pandas](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_data_process_pandas.PNG)

#### data_processing: `json_data_wrangling.ipynb`
- normalize, selection, counts, unique
- `json`, `pandas`   
![mp_data_process_json](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_data_process_json.PNG)

#### data_processing: `SQL_at_scale_with_Spark.ipynb`
- select, from, where, case, join, group by, having, order by, subquery
- used Azure Databricks (create database and 3 tables)   
![mp_data_process_SQL](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_data_process_SQL.PNG)

#### data_transformation: `ETL_and_analyze_web_server_logs_with_Spark.ipynb`
- ETL: regular expressions, missing values, find NULLs, user defined funtion, timestamp, save as csv and json
- analyze: statistics, group by, sort, plot
- used Azure Databricks, 3 million records, PySpark SQL, `pandas`, `matplotlib`   
![mp_data_transform](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_data_transform.PNG)

#### regression: `boston_housing_price.ipynb`
- EDA, linear, residuals, mean squared error, F-statistic, AIC
- `statsmodels`, `scipy`, `sklearn`, `matplotlib`, `seaborn`   
![mp_regression](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_regression.PNG)

#### classification: `gender_from_height_weight.ipynb`  
- logistic regression, write grid search algorithm, predict class, predict probability
- `scipy`, `sklearn`, `matplotlib`, `seaborn`   
![mp_classification](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_classification.PNG)

#### clustering: `customer_segmentation.ipynb`
- K-means, silhouette method, visualize via PCA
- `sklearn`, `matplotlib`, `seaborn`   
![mp_cluster](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_cluster.PNG)

#### trees_and_boosting: `personal_loan_worthiness.ipynb`  
- continuous numerical, discreet numerical, categorical features
- decision tree, random forest, classification report, scores, grid search, feature importance
- `sklearn`, `matplotlib`, `seaborn`, `graphviz`, `dtreeviz`   
![mp_trees](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_trees.PNG)

#### anomaly_detection: `retail_store_sales.ipynb`  
- univariate statistical and isolation forest; multivariate clustering, isolation forest, auto-encoder
- `pyod`, `tensorflow`, `sklearn`, `matplotlib`, `seaborn`   
![mp_anomaly](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_anomaly.PNG)

#### recommendation_systems: `movie_recommender.ipynb`  
- global, content based, collaborative filtering, hybrid
- `keras`, `tensorflow`, `scipy`, `sklearn`, `matplotlib`, `seaborn`
- used Paperspace   
![mp_recommend](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_recommend.PNG)

#### time_series_analysis: `stock_market_forecasting.ipynb`  
- ARIMA, rolling windows, differencing, auto correlation, LSTM
- `keras`, `tensorflow`, `statsmodels`, `sklearn`, `matplotlib`   
![mp_time_series](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_time_series.PNG)

#### spark_ml: `data_exploration.ipynb`
- used Databricks
- US census data
- SQL table, Spark DataFrame, Pandas DataFrame, SQL queries, Spark queries, visualization
- `pyspark.sql`, `matplotlib`
###### income category
![mp_SparkML_data_investigate](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_SparkML_data_investigate.PNG)

#### spark_ml: `income_category.ipynb`
- used Databricks
- US census data; binary classification
- pre-process categorical features and target, performance metrics, coefficients, feature importance, cross validation
- `OneHotEncoder`, `StringIndexer`, `VectorAssembler`, `Pipeline`, `LogisticRegression`, `GBTClassifier`, `RandomForestClassifier`, `BinaryClassificationEvaluator`, `BinaryClassificationMetrics`, `MulticlassMetrics`, `ParamGridBuilder`, `CrossValidator`
###### GBTClassifier feature importance
![mp_SparkML_GBT_feature_importance](https://github.com/LoriNewhouse/Springboard_Machine_Learning_Engineering_bootcamp/blob/main/mini_projects/images/mp_SparkML_GBT_feature_importance.PNG)
