import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model_utils import model,get_stock_data,preprocess,modelPredict

def ui():
    st.header("💹Stock price prediction model")
    stock=st.selectbox("Select stock:",['ADANIENT.BSE','APOLLOHOSP.BSE','COALINDIA.BSE','ITC.BSE','RELIANCE.BSE'])
    if st.button("Predict"):
        
        df=get_stock_data(stock)
        data,sc=preprocess(df.copy())
        pred_close=modelPredict(model,data)
        
        df=df.sort_values(by='date')
        
        df_last_100 = df.tail(100).copy()
        df_last_100 = df_last_100.reset_index(drop=True)
        df_last_100["date"] = pd.to_datetime(df_last_100["date"])
        
        last_close = df_last_100["4. close"].iloc[-1]
        next_date = df_last_100["date"].iloc[-1] + pd.Timedelta(days=1)
        change=((pred_close - last_close) / last_close) * 100
        
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_last_100["date"], df_last_100["4. close"],
                color="gray", linewidth=2, label="Past 100 Days", alpha=0.8)
        
        ax.plot(
            [df_last_100["date"].iloc[-1], next_date],
            [last_close, pred_close],
            color="red", linewidth=2, marker='o',
            label="Predicted Next Day"
        )
        ax.set_title(f"{stock} - Last 100 Days + Next Day Prediction", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (₹)")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

        st.write(f"**Last close:** ₹{last_close:.2f}")
        st.write(f"**Predicted next close:** ₹{pred_close:.2f}")
        st.write(f"**Change:** {change:+.2f}%")
    
if __name__=='__main__':
    ui()
    
        