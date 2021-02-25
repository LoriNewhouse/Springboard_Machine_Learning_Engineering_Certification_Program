# This folder contains the code needed to prepare the data and fit a decision tree model using the tuned hyper-parameters.
   
### These packages are used:
- `pandas`
- `numpy`
- `pickle`
- `matplotlib`
- `seaborn`
- `pycaret`
- `imblearn` (automatically installed with `pycaret`)   
### Follow these steps:
1. download data from http://archive.ics.uci.edu/ml/datasets/Covertype (not included here due to size)
2. put download here: `data/covtype.data.gz`
3. run notebook `data_wrangling_and_EDA.ipynb` (will create `data/wrangled_data.pkl` used in next step)
4. run notebook `fit_tuned_decision_tree.ipynb` (will create `finalized_dt_model.pkl`)