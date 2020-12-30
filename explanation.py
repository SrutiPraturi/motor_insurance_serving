import numpy as np 
import pandas as pd 
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os 




def explain_prediction(data , model , explain_data,mode):
    
    
    #data = pd.read_csv(train_data)
    feature_names = list(data.columns)
    target = ['Payment']
    categorical_features = []
    
    
    for col in data.columns:
        if data[col].dtypes == 'object' or data[col].dtypes =='bool':
            categorical_features.append(col)
            
    explainer = lime.lime_tabular.LimeTabularExplainer(data.to_numpy(), feature_names=feature_names, class_names=target, categorical_features=categorical_features,  verbose=True, mode=mode)
    
    exp = explainer.explain_instance(explain_data.loc[0], model.predict, num_features=10)
    
    exp.as_pyplot_figure();
    
    plt.savefig('explanation.png')
    
    if os.path.exists('explanation.png'):
        return 1
    else :
        return 0
    
    return 
