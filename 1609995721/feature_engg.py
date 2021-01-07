import pandas as pd 
import numpy as np
import pickle
import sys

class FeatureEngg:

    def load_data(self,data_path):
        data = pd.read_csv(data_path)
    
        return data


    def model_ready_data(self,data):
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

    def save_data(self,data):
    
        data.to_csv('assets/insurance_processed_data.csv',index = False)
        
    

if __name__ == "__main__":
    
    featurengg = FeatureEngg()
    mode = sys.argv[1]
    if mode == 'training':
        data = featurengg.load_data('datasets/insurance_treated_data.csv')
        data = featurengg.model_ready_data(data)
        featurengg.save_data(data)
    elif mode == 'serving':
        data = featurengg.model_ready_data()
    
    
    


        
