import pandas as pd 
import numpy as np
import pickle
import sys

def load_data(data_path):
    data = pd.read_csv(data_path)
    
    return data


def model_ready_data(data):
    file = open('assets/RFR_claims_insured.pkl','rb')
    model = pickle.load(file)
    file.close()
    
    data['PLS'] = model.predict(data[['Claims','Insured']])
    data.drop(['Claims','Insured'] , axis = 1 , inplace = True)
    
    new_columns =['Kilometres_1','Kilometres_2','Kilometres_3','Kilometres_4','Kilometres_5',
                  'Zone_1','Zone_2','Zone_3','Zone_4','Zone_5','Zone_6','Zone_7',
                  'Bonus_1','Bonus_2','Bonus_3','Bonus_4','Bonus_5','Bonus_6','Bonus_7',
                  'Make_1','Make_2','Make_3','Make_4','Make_5','Make_6','Make_7','Make_8','Make_9']
    for i in new_columns:
        data[i] = 0
    pivot = ['Kilometres','Zone','Bonus','Make']
    
    for index , row in data.iterrows():
        for col in pivot:
            value = row[col]
            pivot_col = col+'_'+str(int(value))
            data.at[index,pivot_col] = 1
    data = data.drop(columns = pivot) 
    return data

def save_data(data):
    
    data.to_csv('datasets/insurance_processed_data.csv',index = False)
        
    

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == 'training':
        data = load_data('datasets/insurance_treated_data.csv')
        data = model_ready_data(data)
        save_data(data)
    elif mode == 'serving':
        data = model_ready_data()
    
    
    


        
