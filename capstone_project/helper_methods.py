
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import classification_report

def plot_confusion_matrices(model_name, y_train, y_train_pred, y_test, y_test_pred, target_names):

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test  = confusion_matrix(y_test,  y_test_pred)

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    cm_display = ConfusionMatrixDisplay(cm_train, display_labels=target_names)
    cm_display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax1);
    ax1.set_title('train data');

    cm_display = ConfusionMatrixDisplay(cm_test, display_labels=target_names)
    cm_display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax2);
    ax2.set_title('test data');

    fig.suptitle(model_name, size=30, weight=100)

    plt.show()


def make_dataframe_class_report(model_name, data_name, target_names, y_true, y_pred):

    class_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    dummy_line = {'model':model_name , 'data':data_name , 'class':'green' ,
                  'precision':0.1 , 'recall':0.2 , 'f1-score':0.4 , 'support':8}
    df_report = pd.DataFrame(dummy_line, index=[0])
    #df_report

    for name in target_names:
        one_line = {'model':model_name , 'data':data_name, 'class':name}
        one_line.update(class_dict[name])
        #one_line
        df_report = df_report.append(one_line, ignore_index=True)

    #df_report
    df_report.drop(index=[0], inplace=True)
    return(df_report)


def make_dataframes_from_classification_report(model_name, data_name, target_names, y_true, y_pred):
    class_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    # metrics for each class
    dummy_line = {'model':model_name , 'data':'dummy' , 'class':'green' ,
                  'precision':0.1 , 'recall':0.2 , 'f1-score':0.4 , 'support':8}
    df_report_by_class = pd.DataFrame(dummy_line, index=[0])
    #df_report_by_class

    for name in target_names:
        one_line = {'model':model_name , 'data':data_name, 'class':name}
        one_line.update(class_dict[name])
        #one_line
        df_report_by_class = df_report_by_class.append(one_line, ignore_index=True)
    #df_report_by_class
    
    
    # metrics for model
    dummy_line = {'model':model_name , 'data':'dummy' , 'accuracy':0.8 ,
                  'macro_precision':0.1 , 'macro_recall':0.2 , 'macro_f1-score':0.3 ,
                  'weight_precision':0.5 , 'weight_recall':0.6 , 'weight_f1-score':0.7}
    df_report_summary = pd.DataFrame(dummy_line, index=[0])

    one_line = {'model':model_name , 'data':data_name , 'accuracy':class_dict['accuracy'] ,
                'macro_precision': class_dict['macro avg']['precision'] ,
                'macro_recall':    class_dict['macro avg']['recall'] ,
                'macro_f1-score':  class_dict['macro avg']['f1-score'] ,
                'weight_precision':class_dict['weighted avg']['precision'] ,
                'weight_recall':   class_dict['weighted avg']['recall'] ,
                'weight_f1-score': class_dict['weighted avg']['f1-score']}
    df_report_summary = df_report_summary.append(one_line, ignore_index=True)
    #df_report_summary
    
    df_report_by_class.drop(index=[0], inplace=True)
    df_report_summary.drop(index=[0], inplace=True)
    return(df_report_by_class, df_report_summary)


def plot_classification_report(model_name, df_class_report):
    for_plot = ['class', 'precision', 'recall', 'f1-score']
    df_to_plot_train = df_class_report[(df_class_report['model']==model_name) & (df_class_report['data']=='train')][for_plot]
    df_to_plot_test  = df_class_report[(df_class_report['model']==model_name) & (df_class_report['data']=='test')][for_plot]

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    df_to_plot_train.set_index('class').plot.bar(ax=ax1)
    ax1.set_title('train data');
    ax1.set_xlabel('');
    ax1.set_ybound(lower=0, upper=1)

    df_to_plot_test.set_index('class').plot.bar(ax=ax2)
    ax2.set_title('test data');
    ax2.set_xlabel('');
    ax2.set_ybound(lower=0, upper=1)
    
    fig.suptitle(model_name + ': metrics by class', size=30, weight=100)

    plt.show


def plot_model_summary_metrics(df_class_report):
    for_plot = ['model', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1-score',
                'weight_precision', 'weight_recall', 'weight_f1-score']
    df_to_plot_train = df_class_report[(df_class_report['data']=='train')][for_plot]
    df_to_plot_test  = df_class_report[(df_class_report['data']=='test')][for_plot]

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    df_to_plot_train.set_index('model').plot.bar(ax=ax1)
    ax1.set_title('train data');
    ax1.set_xlabel('');
    ax1.set_ybound(lower=0, upper=1)

    df_to_plot_test.set_index('model').plot.bar(ax=ax2)
    ax2.set_title('test data');
    ax2.set_xlabel('');
    ax2.set_ybound(lower=0, upper=1)
    
    fig.suptitle('summary metrics by model', size=30, weight=100)
    
    plt.show


def plot_small_classes_metric(df_class_report, the_metric):   
    df_to_plot_train = make_dataframe_small_classes(df_class_report, 'train', the_metric)
    df_to_plot_test  = make_dataframe_small_classes(df_class_report, 'test',  the_metric)

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    df_to_plot_train.plot.bar(ax=ax1)
    ax1.set_title('train data');
    ax1.set_xlabel('');
    ax1.set_ybound(lower=0, upper=1)

    df_to_plot_test.plot.bar(ax=ax2)
    ax2.set_title('test data');
    ax2.set_xlabel('');
    ax2.set_ybound(lower=0, upper=1)
    
    fig.suptitle(the_metric, size=30, weight=100)
    
    plt.show


def make_dataframe_small_classes(df_input, the_data, the_metric):
    small_classes = ['ponderosa_pine', 'krummholz', 'douglas_fir', 'aspen','cottonwood_willow']
    for_plot = ['model', the_metric]
    df_grand = pd.DataFrame()

    for the_class in small_classes:
        df_to_plot = df_input[(df_input['data']==the_data) & (df_input['class']==the_class)] \
                           [for_plot]
        #df_to_plot_train.head(10)

        df_temp = df_to_plot.transpose()
        #list(df_temp.loc['model'].values)

        df_temp.columns=list(df_temp.loc['model'].values)
        #df_temp

        df_temp = df_temp.drop(index=['model'])
        #df_temp

        df_temp.index = [the_class]
        df_grand = df_grand.append(df_temp)
    
    return(df_grand)
