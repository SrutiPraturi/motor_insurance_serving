import pandas as pd 
import pickle
import yaml
import numpy as np 
from flask import Flask, jsonify, request
from explanation import Explanation
import base64
from feature_engg import model_ready_data

app = Flask(__name__)

lime_data = pd.read_csv('assets/insurance_processed_data.csv').drop(columns = ['Payment'])

@app.route('/payment', methods = ['POST']) 
def get_data():
    
    explain_data = request.data.decode()
    
    explain_data = explain_data.split(',')
    print(request.data)
    predict_data = []
    for val in explain_data:
        if val.isdigit() == True:
            predict_data.append(int(val))
        elif val.replace('.', '', 1).replace('"','',1).isdigit() == True:
            predict_data.append(float(val))
        else:
            predict_data.append(str(val))
            
    file = open('assets/claims_model.pkl','rb')
    model = pickle.load(file)
    file.close()
    
    data = pd.DataFrame(columns = ['Kilometres','Zone','Bonus','Make','Insured','Claims'])
    data.loc[0,:] = predict_data
    model_data = model_ready_data(data)
    
    print(model_data)
        
    return_data = {}
    
    prediction = model.predict(np.array(model_data.iloc[0,:]).reshape(1,-1))
    
    return_data['prediction'] = str(np.expm1(prediction))
    
    result = explanation.explain_prediction(lime_data,model,model_data,'regression')
    
    
    
    if result == 1:
        
        with open('explanation.png', mode='rb') as file:
            img = file.read()

    return_data['img'] = base64.b64encode(img).decode()
    
    return_data = jsonify(return_data)
        
    
    return return_data

if __name__ == "__main__":
    explanation = Explanation()
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
    #app.run(port=8080)
        
        
    
    
    
    
    
    
    
    
    
    
    
