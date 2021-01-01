# Mini Projects
The `ipynb` notebooks in this directory are final projects associated with a technical skill unit. They are more complicated, real-world application driven tasks than the homework style work in the `learning_exercises` directory. Some were done on a cloud provider (Paperspace) or Azure Databricks. Each took 1-3 days to complete.

#### data_preparation: `quandl_website_api.ipynb`
- financial market data
- create dictionary; calculate max, min, avg, changes
- `requests`, `json`

#### data_processing:
`pandas_data_wrangling.ipynb`
- 3 csv files with movies data
- select, query, sort, aggregate, merge, plot

`json_data_wrangling.ipynb`
- normalize, selection, counts, unique
- `json`, `pandas`

`SQL_at_scale_with_Spark.ipynb`
- select, from, where, case, join, group by, having, order by, subquery
- used Azure Databricks (create database and 3 tables)

#### data_transformation: `ETL_and_analyze_web_server_logs_with_Spark.ipynb`
- ETL: regular expressions, missing values, find NULLs, user defined funtion, timestamp, save as csv and json
- analyze: statistics, group by, sort, plot
- used Azure Databricks, 3 million records, PySpark SQL, `pandas`, `matplotlib`

#### regression: `boston_housing_price.ipynb`
- EDA, linear, residuals, mean squared error, F-statistic, AIC
- `statsmodels`, `scipy`, `sklearn`, `matplotlib`, `seaborn`

#### classification: `gender_from_height_weight.ipynb`  
- logistic regression, write grid search algorithm, predict class, predict probability
- `scipy`, `sklearn`, `matplotlib`, `seaborn`

#### trees_and_boosting: `personal_loan_worthiness.ipynb`  
- continuous numerical, discreet numerical, categorical features
- decision tree, random forest, classification report, scores, grid search, feature importance
- `sklearn`, `matplotlib`, `seaborn`, `graphviz`, `dtreeviz`

#### anomaly_detection: `retail_store_sales.ipynb`  
- univariate statistical and isolation forest; multivariate clustering, isolation forest, auto-encoder
- `pyod`, `tensorflow`, `sklearn`, `matplotlib`, `seaborn`

#### recommendation_systems: `movie_recommender.ipynb`  
- global, content based, collaborative filtering, hybrid
- `keras`, `tensorflow`, `scipy`, `sklearn`, `matplotlib`, `seaborn`
- used Paperspace

#### time_series_analysis: `stock_market_forecasting.ipynb`  
- ARIMA, rolling windows, differencing, auto correlation, LSTM
- `keras`, `tensorflow`, `statsmodels`, `sklearn`, `matplotlib`

