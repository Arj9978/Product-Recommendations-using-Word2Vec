import streamlit as st
import numpy as np
import pandas as pd
import joblib
import lightgbm

from utils import columns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('transaction_dataset.csv', index_col=0)
df = df.iloc[:,2:]
# Turn object variables into 'category' dtype for more computation efficiency
categories = df.select_dtypes('O').columns.astype('category')
# Drop the two categorical features
df.drop(df[categories], axis=1, inplace=True)
df.fillna(df.median(), inplace=True)
# Filtering the features with 0 variance
no_var = df.var() == 0
# Drop features with 0 variance --- these features will not help in the performance of the model
df.drop(df.var()[no_var].index, axis = 1, inplace = True)
drop = ['total transactions (including tnx to create contract',
        'total ether sent contracts', 
        'max val sent to contract',
        ' ERC20 avg val rec',
        ' ERC20 avg val rec',
        ' ERC20 max val rec', 
        ' ERC20 min val rec', 
        ' ERC20 uniq rec contract addr', 
        'max val sent', 
        ' ERC20 avg val sent',
        ' ERC20 min val sent', 
        ' ERC20 max val sent', 
        ' Total ERC20 tnxs', 
        'avg value sent to contract', 
        'Unique Sent To Addresses',
        'Unique Received From Addresses', 
        'total ether received', 
        ' ERC20 uniq sent token name', 
        'min value received', 
        'min val sent', 
        ' ERC20 uniq rec addr' ]
df.drop(drop, axis=1, inplace=True)
drops = ['min value sent to contract', ' ERC20 uniq sent addr.1', 'FLAG']
df.drop(drops, axis=1, inplace=True)

scaler = MinMaxScaler()
sc_df = scaler.fit_transform(df)

model = joblib.load('finalized_model.joblib')
st.title('Crypto Fraud Detection')
st.write("""### We need some information to Detect the Fraud""")

Avg_min_between_sent_tnx = st.slider("Avg min between sent tnx", 0, 430288, 215144)
Avg_min_between_received_tnx = st.slider("Avg min between received tnx", 0, 482176, 241088)
Time_Diff_between_first_and_last_Mins = st.slider("Time Diff between first and last (Mins)", 0, 1954861, 977430)
Sent_tnx = st.slider("Sent tnx", 0, 10000, 5000)
Received_Tnx = st.slider("Received Tnx", 0, 10000, 5000)
Number_of_Created_Contracts = st.slider("Number of Created Contracts", 0, 9995, 4998)
max_value_received = st.slider("max value received", 0, 800000, 400000)
avg_val_received = st.slider("avg val received", 0, 283619, 141809)
avg_val_sent = st.slider("avg val sent", 0, 12000, 6000)
total_Ether_sent = st.slider("total Ether sent", 0, 28580960, 14290480)
total_ether_balance = st.slider("total ether balance", -15605350, 14288640, -658355)
ERC20_total_Ether_received = 5e12
ERC20_total_ether_sent = 56e9
ERC20_total_Ether_sent_contract = st.slider("ERC20 total Ether sent contract", 0, 416000, 208000)
ERC20_uniq_sent_addr = st.slider("ERC20 uniq sent addr", 0, 6582, 3291)
ERC20_uniq_rec_token_name = st.slider("ERC20 uniq rec token name", 0, 737, 368)

columns = ['Avg min between sent tnx', 'Avg min between received tnx', 'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
           'Number of Created Contracts', 'max value received', 'avg val received', 'avg val sent', 'total Ether sent', 'total ether balance', 
           'ERC20 total Ether received', 'ERC20 total ether sent', 'ERC20 total Ether sent contract', 'ERC20 uniq sent addr', 'ERC20 uniq rec token name']

ok = st.button("Detect Fraud")
if ok: 
    # Create a DataFrame with the row data and columns matching the training data
    row = np.array([Avg_min_between_sent_tnx, Avg_min_between_received_tnx, Time_Diff_between_first_and_last_Mins, Sent_tnx, Received_Tnx,
                    Number_of_Created_Contracts, max_value_received, avg_val_received, avg_val_sent, total_Ether_sent, total_ether_balance,
                    ERC20_total_Ether_received, ERC20_total_ether_sent, ERC20_total_Ether_sent_contract, ERC20_uniq_sent_addr, ERC20_uniq_rec_token_name])

    # Create a DataFrame with the row data and columns matching the training data
    X = pd.DataFrame([row], columns=columns)

    # Align the columns with the training data columns
    X = scaler.transform(X)

    X = np.array(X)
    st.write(X)
    
    Detect_Fraud = model.predict(X)

    if Detect_Fraud==0 :
        st.subheader(f"No Fraud")
    else :
        st.subheader(f"Fraud Detected")
