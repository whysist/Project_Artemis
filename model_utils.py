import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import os
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler

model=None
with open('model.pkl','rb') as f:
    model=pickle.load(f)

load_dotenv(".env.txt")
API_KEY=os.getenv("API_KEY")


def get_stock_data(symbol:str):
    ts=TimeSeries(key=API_KEY,output_format="pandas")
    data,_=ts.get_daily(symbol=symbol,outputsize="compact")
    data.reset_index(inplace=True)
    df=pd.DataFrame(data)
    return df
def preprocess(df):
    
    rename_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df.columns = rename_cols + list(df.columns[len(rename_cols):])  # preserve extras if present

    # add encoded symbol if missing
    if 'symbol_encoded' not in df.columns:
        df['symbol_encoded'] = 0
    
    X_input = df[['open','high','low','close','volume','symbol_encoded']]
    scaler=None
    with open('Standard_scaler.pkl','rb') as f:
        scaler=pickle.load(f)
    
    X_scaled=scaler.transform(X_input.to_numpy())        
    dummy_target = np.zeros((X_scaled.shape[0], 1))
    X_scaled = np.hstack([X_scaled, dummy_target])  # now 7 columns
    X_scaled = X_scaled.reshape(1, 100, 7)

    return X_scaled,scaler 

def modelPredict(model,X_scaled):
    pred_scaled = model.predict(X_scaled)
    predicted_close = float(pred_scaled[0][0])    
    return predicted_close
       
