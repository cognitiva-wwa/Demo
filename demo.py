# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:55:02 2022

@author: Marek
"""

import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pivottablejs
from pivottablejs import pivot_ui
import streamlit.components.v1 as components
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from scipy import stats
import statsmodels.api as sm
#from  sklearn.metrics import mean_absolute_percentage_error
import statsmodels.stats.stattools as sss
from google.oauth2 import service_account
from oauth2client.client import GoogleCredentials
from google.cloud import bigquery
import datetime
from datetime import datetime
import pickle
import db_dtypes
import csv 

st.set_page_config(layout = "wide")

#@st.cache(allow_output_mutation=True)
#def get_base64(bin_file):
#    with open(bin_file, 'rb') as f:
#        data = f.read()
#    return base64.b64encode(data).decode()

def load_data(nrows):
    DATA_URL="Wynik_3.txt"
    superstore = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    superstore['Order_Date'] = pd.to_datetime(superstore['Order_Date'], format='%Y-%m-%d')
    superstore['Ship_Date'] = pd.to_datetime(superstore['Ship_Date'], format='%Y-%m-%d')
    return superstore

@st.cache(allow_output_mutation=True)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [dict(row) for row in rows_raw]
    return rows

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()
train = pd.DataFrame() 
val = pd.DataFrame()
test = pd.DataFrame()

def split(df_w):
    column_indices = {name: i for i, name in enumerate(df_w.columns)}
    n = len(df_w)
    train = df_w[0:int(n*0.8)]
    val = df_w[int(n*0.8):int(n*0.9)]
    test = df_w[int(n*0.9):]
    num_features = df_w.shape[1]
    return train, val, test

def standard_ize(train, val, test):
    train_mean = train.mean()
    train_std = train.std()
  
    train_df = (train - train_mean) / train_std
    val_df = (val - train_mean) / train_std
    test_df = (test - train_mean) / train_std    
    return train_df, val_df, test_df

st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader('Manually upload your csv file there')

if uploaded_file:
   superstore = pd.read_csv(uploaded_file)
   df = pd.DataFrame(superstore)
   df['Order_Date'] = pd.to_datetime(df['Order_Date'])

st.subheader("Interactive dashboard for Eda - Exploratory data analysis")

st.sidebar.text('Import dataset:')
with st.sidebar:
     add_radio = st.radio(
     "Choose a method of import:",
     ("Empty", "Local data", "BigQuery"))

     page = st.sidebar.selectbox('Select page',
             ['Empty page', 'Main', 'Quantitative data analysis', 'Profit looses'])

if page == 'Empty page':
      if add_radio == "Empty":
         def empty():
             st.error('Data Error!!! Choose import metod: Local data or BigQuery')
             return
      if add_radio == "Local data":
         superstore = load_data(9995)
         df = superstore
        
      if add_radio == "BigQuery":
         credentials = service_account.Credentials.from_service_account_info(
         st.secrets["gcp_service_account"]
         )
         client = bigquery.Client(credentials=credentials)
         rows = run_query("SELECT * FROM predykcja-incomig-calls.Retail_Data_1.Retails_1")
         df =  pd.DataFrame(rows)
                  
if page == 'Empty page':
      st.sidebar.text('Numeric data analysis:')
      with st.sidebar:
           add_radio = st.radio(
           "Choose a method of analysis:",
           ("Empty", "Data profiles", "Pivot Tables"))
      if add_radio == "Empty":
         def empty():
             return
      if add_radio == "Data profiles":
         if add_radio == "Local data": 
              superstore = load_data(9995)
              df = superstore
         if add_radio == "BigQuery":
              credentials = service_account.Credentials.from_service_account_info(
              st.secrets["gcp_service_account"]
              )
              client = bigquery.Client(credentials=credentials) 
              rows = run_query("SELECT * FROM predykcja-incomig-calls.Retail_Data_1.Retails_1")
              df =  pd.DataFrame(rows)
              df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')
              df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], format='%Y-%m-%d')
            
         pr = ProfileReport(df, explorative=True)
         st.subheader("Pandas Profiling in Streamlit")
         st.write(df)
         st_profile_report(pr)
      if add_radio == "Pivot Tables":
         if add_radio == "Local data": 
              superstore = load_data(9995)
              df = superstore
         if add_radio == "BigQuery":
              credentials = service_account.Credentials.from_service_account_info(
              st.secrets["gcp_service_account"]
              )
              client = bigquery.Client(credentials=credentials) 
              rows = run_query("SELECT * FROM predykcja-incomig-calls.Retail_Data_1.Retails_1")
              df =  pd.DataFrame(rows)
              df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')
              df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], format='%Y-%m-%d')
            
         st.subheader("Pivot table in Streamlit")
         t = pivot_ui(df)
         with open(t.src) as t:
              components.html(t.read(), width=900, height=1000, scrolling=True)
if page == 'Empty page':
      col1, col2, col3 = st.columns(3)
      st.sidebar.text('Machine learning model - Regression:')
      with st.sidebar:
           add_radio = st.radio(
           "Choose a computation:",
           ("Empty", "Local - OLS model", "Time Series"))
      if add_radio == "Empty":
         def empty():
             return
      if add_radio == "Local - OLS model":

         df['Price'] = df['Sales']/df['Quantity']
         df.sort_values(by=['Price'], ascending=False)
         endog_df = df[['Sales']]
         exog_df = df[['Price']]
         exog_df = sm.add_constant(exog_df, prepend=True)
         pd.set_option('mode.chained_assignment', None)
         endog_df['Sales_log'] = np.log(endog_df['Sales']+1)
         product_cat_dummies= pd.get_dummies(df['Category']).iloc[:,1:]
         exog_df[product_cat_dummies.columns] = product_cat_dummies
         quad_endog = df[['Sales']]
         quad_exog = df[['Price']]
         quad_exog = sm.add_constant(quad_exog, prepend=True)
         quad_exog['Price^2'] = quad_exog['Price'] * quad_exog['Price']
         D_df = df.groupby("Order_Date",as_index=False)["Sales"].sum()
         D_df['Timestamp'] = pd.to_datetime(D_df['Order_Date'],format="%Y-%m-%d")
         D_df = D_df.sort_values(by=['Timestamp'], ascending=True)
         D_df['Acc_Sales'] = D_df['Sales'].cumsum()
         trend_endog = D_df[['Acc_Sales', 'Sales']]
         trend_exog = pd.DataFrame()
         trend_exog['day'] = (D_df['Timestamp'] - D_df['Timestamp'].values[0]).dt.days


         def forecast_accuracy(df, obs, pred):
             mfe = (df[obs] - df[pred]).mean()
             mae = abs(df[obs] - df[pred]).mean()
             mse = ((df[obs] - df[pred]) ** 2).mean()
             mape = (((df[obs] - df[pred]) / df[obs]) * 100).mean()

             return 'Mean Forcasting Error (MFE): '+str(mfe)+\
                    '\nMean Absolute Error (MAE): '+str(mae)+\
                    '\nMean Squared Error (MSE): '+str(mse)+\
                    '\nMean Absolute Percentage Error (MAPE): '+str(mape)

         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
             fig, axs = plt.subplots(1, 2, figsize=(18, 6))
             sns.histplot(data=endog_df, x='Sales', kde=True, ax=axs[0])
             sns.histplot(data=endog_df, x='Sales', kde=True, ax=axs[1])
             axs[0].set_title('Normality test for Sales:' +
                  '\n'+ str(stats.normaltest(endog_df['Sales'])) +
                  '\nSkewness: ' + str(stats.skew(endog_df['Sales'])) +
                  '\nKurtosis: ' + str(stats.kurtosis(endog_df['Sales']) + 3),color='white')
             axs[1].set_title('Normality test for log(Sales):' +
                   '\n'+ str(stats.normaltest(endog_df['Sales_log'])) +
                   '\n' + str(stats.jarque_bera(endog_df['Sales_log'])) +
                   '\nSkewness: ' + str(stats.skew(endog_df['Sales_log'])) +
                   '\nKurtosis: ' + str(stats.kurtosis(endog_df['Sales_log']) + 3),color='white')
             st.pyplot(fig)

         mod_log = sm.OLS(endog_df['Sales_log'], exog_df)
         res_log = mod_log.fit()
         st.text(res_log.summary())

         mod = sm.OLS(endog_df['Sales'], exog_df)
         res = mod.fit()
         
         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
             fig, axs = plt.subplots(1, 2, figsize=(18, 6))
             sns.histplot(data=res.resid, kde=True, ax=axs[0])
             sns.histplot(data=res_log.resid, kde=True, ax=axs[1])
             axs[0].set_title('Test residual distribution of Sales:' +
                 '\n'+ str(stats.normaltest(res.resid)) +
                 '\n' + str(stats.jarque_bera(res.resid)) +
                 '\nSkewness: ' + str(stats.skew(res.resid)) +
                 '\nKurtosis: ' + str(stats.kurtosis(res.resid) + 3) +
                 '\nAutocorrelation: ' + str(sss.durbin_watson(res.resid)), color='white')
             axs[1].set_title('Test residual distribution of log(Sales):' +
                 '\n'+ str(stats.normaltest(res_log.resid)) +
                 '\n' + str(stats.jarque_bera(res_log.resid)) +
                 '\nSkewness: ' + str(stats.skew(res_log.resid)) +
                 '\nKurtosis: ' + str(stats.kurtosis(res_log.resid) + 3) +
                 '\nAutocorrelation: ' + str(sss.durbin_watson(res_log.resid)), color='white')
             st.pyplot(fig)
         st.text(res.summary())

         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
             fig1 = plt.figure(figsize=(16,8))
             sns.regplot(data=df, x='Price', y='Sales').set_title('Price vs Sales - Observed Data',color='white')
             st.pyplot(fig1)
             fig2 = plt.figure(figsize=(16,8))
             sns.regplot(data=df, x='Profit', y='Sales').set_title('Profit vs Sales - Observed Data',color='white')
         st.pyplot(fig2)

         linear_mod = sm.OLS(trend_endog['Acc_Sales'], trend_exog)
         linear_res = linear_mod.fit()
         st.text(linear_res.summary())

         st.subheader('Prediction table and chart - linear model')
         linear_pred = linear_res.get_prediction().summary_frame()
         linear_pred.columns = ['pred_sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
         linear_pred[['obs_Sales', 'timestamp']] = D_df[['Acc_Sales', 'Timestamp']]
         st.dataframe(linear_pred)

         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
             fig3 = plt.figure(figsize=(16,8))
             sns.lineplot(data=linear_pred, x='timestamp', y='pred_sales', color='red', label='Prediction')
             sns.lineplot(data=linear_pred, x='timestamp', y='obs_Sales', color='blue', alpha=0.5, label='Observation')
             plt.fill_between(x=linear_pred['timestamp'], y1=linear_pred['ci_lower'], y2=linear_pred['ci_upper'], color='teal', alpha=0.2)
             plt.fill_between(x=linear_pred['timestamp'], y1=linear_pred['pi_lower'], y2=linear_pred['pi_upper'], color='skyblue', alpha=0.2)
             st.pyplot(fig3)
             st.text(forecast_accuracy(linear_pred, 'obs_Sales', 'pred_sales'))

         trend_exog = sm.add_constant(trend_exog, prepend=True)
         trend_exog['day^2'] = trend_exog['day'] ** 2
         nonlinear_mod = sm.OLS(trend_endog['Acc_Sales'], trend_exog)
         nonlinear_res = nonlinear_mod.fit()
         st.text(nonlinear_res.summary())
         
         st.subheader('Prediction table and chart - nonlinear model')
         nonlinear_pred = nonlinear_res.get_prediction().summary_frame()
         nonlinear_pred.columns = ['pred_sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
         nonlinear_pred[['obs_Sales', 'timestamp']] = D_df[['Acc_Sales', 'Timestamp']]
         st.dataframe(nonlinear_pred)

         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
             fig4 = plt.figure(figsize=(16,8))
             sns.lineplot(data=nonlinear_pred, x='timestamp', y='pred_sales', color='red', label='Prediction')
             sns.lineplot(data=nonlinear_pred, x='timestamp', y='obs_Sales', color='blue', alpha=0.5, label='Observation')
             plt.fill_between(x=nonlinear_pred['timestamp'], y1=nonlinear_pred['ci_lower'], y2=nonlinear_pred['ci_upper'], color='teal', alpha=0.2)
             plt.fill_between(x=nonlinear_pred['timestamp'], y1=nonlinear_pred['pi_lower'], y2=nonlinear_pred['pi_upper'], color='skyblue', alpha=0.2)
             st.pyplot(fig4)
             st.text(forecast_accuracy(nonlinear_pred, 'obs_Sales', 'pred_sales'))
         
         # if st.button('Save OLS model'):
         #    pickle_out = open("model_OLS.pkl", "wb")
         #    pickle.dump(nonlinear_mod, pickle_out)
         #    pickle_out.close()
            
      if add_radio == "Time Series":      
          
         store= df.copy()
         cols = ['Row_ID', 'Order_ID', 'Ship_Date', 'Ship_Mode', 'Customer_ID', 'Customer_Name', 'Country', 'Postal_Code', 'Product_Name', 'Quantity', 'Discount', 'Profit']
         store.drop(cols, axis=1, inplace=True)
         store = store.groupby('Order_Date')['Sales'].sum().reset_index()
         store = store.set_index('Order_Date')
         store.index = pd.to_datetime(store.index)
         y = store['Sales'].resample('MS').mean()
         train = y[:40]
         test = y[40:]
                    
         model_TS = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 1),seasonal_order=(1,1,1,12))
         results = model_TS.fit()
         st.text(results.summary())
         pre=results.predict(start = len(train), end = (len(y)-1),dynamic=True)
          
         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):   
              fig_cast, ax = plt.subplots(1, figsize=(10, 6))
              ax.set_title('Predict plot:', color = 'white')
              plt.xlabel('Date')
              plt.ylabel('Sales')
              ax.set_facecolor("#37383f")
              train.plot(legend=True, label='Train')            
              test.plot(legend=True, label= 'Test')
              pre.plot(legend=True, label='SARIMAX prediction')
              st.pyplot(fig_cast)
          
         #mape = mean_absolute_percentage_error(test, pre)
         #print('MAPE: %f' %mape)
          
         future_sale= results.predict(start = len(y), end = (len(y)+12),dynamic=True)
         st.subheader('Forecast table and chart - SARIMAX model')
         future_sale
 
         with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):   
              fig_fut, ax = plt.subplots(1, figsize=(10, 6))    
              ax.set_title('Forecast plot:', color = 'white')
              plt.xlabel('Date')
              plt.ylabel('Sales')
              ax.set_facecolor("#37383f")              
              y.plot(legend=True, label='Current Sale', figsize=(10,6))
              future_sale.plot(legend= True, label='Future Sale')
              st.pyplot(fig_fut)
               
         # if st.button('Save TS model'):
         #    pickle_out = open("model_TS.pkl", "wb")
         #    pickle.dump(model_TS, pickle_out)
         #    pickle_out.close()         
          
if page == 'Empty page':
      col1, col2 = st.columns([2,5])
      poz1, poz2, poz3 = st.columns(3)
      wide1, wide2 = st.columns([6,1])
      st.sidebar.text('What - If Analysis:')
      with st.sidebar:
            add_radio = st.radio(
            "Choose an analysis:",
            ("Empty", "What - If for TS model"))
      if add_radio == "Empty":
          def empty():
              return
                                     
      if add_radio == "What - If for TS model":
         with col1:
               with st.container():    
                    rlist = np.array([])
                    rlist = np.append(rlist, 'Empty Selector')
                    rlist = np.append(rlist,df['Region'].unique())
                    slist = np.array([])
                    slist = np.append(slist, 'Empty Selector')
                    clist = np.array([])
                    clist = np.append(clist, 'Empty Selector')
                      
                    sglist = np.array([])
                    sglist = np.append(sglist, 'Empty Selector')
                    sglist = np.append(sglist,df['Segment'].unique())
                    
                    calist = np.array([])
                    calist = np.append(calist, 'Empty Selector')
                    calist = np.append(calist,df['Category'].unique())
                    sclist = np.array([])
                    sclist = np.append(sclist, 'Empty Selector')
    
                    reg = st.selectbox("Select Region:", rlist, index = 0)
                    if reg != 'Empty Selector':
                       data_2= df[['State','Region']]
                       mask1= data_2['Region']==reg
                       df1= data_2[mask1]
                       slist = np.append(slist,df1['State'])
                       slist = np.unique(slist)
                    if reg == 'Empty Selector':
                       slist = np.append(slist,df['State'].unique())
                    sta = st.selectbox("Select State:", slist, index = 0)
                    if reg != 'Empty Selector' and sta != 'Empty Selector':
                       data_2= df[['State', 'City', 'Region']]
                       mask1= data_2['State']==sta
                       df1= data_2[mask1]
                       clist = np.append(clist,df1['City'])
                       clist = np.unique(clist) 
                    if reg != 'Empty Selector' and sta == 'Empty Selector':
                       data_2= df[['City', 'Region']]
                       mask1= data_2['Region']==reg
                       df1= data_2[mask1]
                       clist = np.append(clist,df1['City'])
                       clist = np.unique(clist) 
                    if reg == 'Empty Selector' and sta != 'Empty Selector':
                       data_2= df[['City', 'State']]
                       mask1= data_2['State']==sta
                       df1= data_2[mask1]
                       clist = np.append(clist,df1['City'])
                       clist = np.unique(clist)  
                    if reg == 'Empty Selector' and sta == 'Empty Selector':
                       clist = np.append(clist,df['City'].unique()) 
                    cty = st.selectbox("Select City:", clist, index = 0)
                    seg = st.selectbox("Select Segment:", sglist, index = 0)
                    cat = st.selectbox("Select Category:", calist, index = 0)
                    if cat != 'Empty Selector':
                       data_2= df[['Category','Sub_Category']]
                       mask1= data_2['Category']==cat
                       df1= data_2[mask1]
                       sclist = np.append(sclist,df1['Sub_Category'])
                       sclist = np.unique(sclist)
                    if cat == 'Empty Selector':
                       sclist = np.append(sclist,df['Sub_Category'].unique()) 
                    scat = st.selectbox("Select Sub_Category:", sclist, index = 0)
               with st.container():    
                    with poz1:
                            confirm = st.button('Confirm choice:')                       
                            region_lst = reg
                            state_lst = sta
                            city_lst = cty
                            segment_lst = seg
                            category_lst = cat
                            subcategory_lst = scat
                    with col2:       
                            rdf = df
                            if region_lst !='Empty Selector':
                               rdf = df[df['Region'].isin([region_lst])] 
                            if state_lst !='Empty Selector':
                               rdf = rdf[rdf['State'].isin([state_lst])]     
                            if city_lst !='Empty Selector':
                               rdf = rdf[rdf['City'].isin([city_lst])] 
                            if segment_lst !='Empty Selector':
                               rdf = rdf[rdf['Segment'].isin([segment_lst])] 
                            if category_lst !='Empty Selector':
                               rdf = rdf[rdf['Category'].isin([category_lst])] 
                            if subcategory_lst !='Empty Selector':
                               rdf = rdf[rdf['Sub_Category'].isin([subcategory_lst])] 
                    with wide1:                    
                            if confirm:
                               st.subheader('Filtered Data - Records number:  '+str(len(rdf)))
                               st.dataframe(rdf)
                               st.write('Choice confirmed!')
                    # with poz2:
                    #         load = st.button('Load TS model')
                    #         pickle_in = open('model_TS.pkl', 'rb')
                    #         model_TS = pickle.load(pickle_in)
                    #         if load:
                    #            st.write('TS model loaded!') 
                    with poz3:        
                            slider_val = st.slider("Forecasting range ( Months )", 1, 12, 6)
                            forecast = st.button("Start forecasting")
                    with col2:        
                            if forecast:
                                                                
                                filtered = rdf.copy()
                                cols = ['Row_ID', 'Order_ID', 'Ship_Date', 'Ship_Mode', 'Customer_ID', 'Customer_Name', 'Country', 'Postal_Code', 'Product_Name', 'Quantity', 'Discount', 'Profit']
                                filtered.drop(cols, axis=1, inplace=True)
                                filtered = filtered.groupby('Order_Date')['Sales'].sum().reset_index()
                                filtered = filtered.set_index('Order_Date')
                                filtered.index = pd.to_datetime(filtered.index)
                                yf = filtered['Sales'].resample('MS').mean()
                                train = yf[:40]
                                test = yf[40:]
                    
                                model_TS = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 1),seasonal_order=(1,1,1,12))
                               
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                
                                results_f = model_TS.fit()
                                  
                                future_sale = results_f.predict(start= len(yf), end=(len(yf)+slider_val))
                                with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):   
                                     fig_fut, ax = plt.subplots(1, figsize=(10, 6))    
                                     ax.set_title('Forecast plot:', color = 'white')
                                     plt.xlabel('Date')
                                     plt.ylabel('Filtered Sales')
                                     ax.set_facecolor("#37383f")              
                                     yf.plot(legend=True, label='Filtered Sales', figsize=(10,6))
                                     future_sale.plot(legend= True, label='Future Sale')
                                     st.pyplot(fig_fut)
                                  
if page == 'Empty page':
      col1, col2 = st.columns(2)
      st.sidebar.text('Statistical data analysis:')
      with st.sidebar:
           add_radio = st.radio(
           "Choose an analysis:",
           ("Empty", "Tables&Plots"))
           
           superstore_df = df
           
           superstore_df['Profit_rate'] = superstore_df['Profit']/superstore_df['Sales']
           mean_list = [superstore_df.sample(frac=0.2, replace=False, random_state=seed)['Profit_rate'].mean() for seed in range(35)]
           mean_df = pd.DataFrame({'Sample_Mean':mean_list})

           std_list = [superstore_df.sample(frac=0.15, replace=False, random_state=seed)['Profit_rate'].std() for seed in range(25)]
           std_df = pd.DataFrame({'Sample_Std':std_list})

           sample_list = [superstore_df.sample(frac=0.15, replace=False, random_state=seed) for seed in range(29)]
           p_list = [sample.loc[(sample['Profit_rate'] > 0.2),].shape[0]/sample.shape[0] for sample in sample_list]
           p_df = pd.DataFrame({'Sample_P':p_list})

           mean_lists1 = [[superstore_df.sample(n=n, replace=True, random_state=seed)['Profit_rate'].mean() for seed in range(35)] for n in [2, 100, 1000]]
           mean_df1 = pd.DataFrame({'Sample_Mean (n=2)':mean_lists1[0], 'Sample_Mean (n=100)':mean_lists1[1], 'Sample_Mean (n=1000)':mean_lists1[2]})

           sample_lists2 = [[superstore_df.sample(n=n, replace=True, random_state=seed) for seed in range(35)] for n in [2, 100, 1000]]
           p_lists = [[sample.loc[(sample['Profit_rate'] > 0.2),].shape[0]/sample.shape[0] for sample in sample_list2] for sample_list2 in sample_lists2]
           p_df1 = pd.DataFrame({'Sample_P (n=2)':p_lists[0], 'Sample_P (n=100)':p_lists[1], 'Sample_P (n=1000)':p_lists[2]})

      if add_radio == "Empty":
         def empty():
             return
      if add_radio == "Tables&Plots":
              
         with st.container():
             with col1:
                  fig1 = plt.figure(figsize=(12,6))
                  sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
                  fig1.suptitle('Frequency distribution of sample means', color='white', fontsize=20)
                  ax = sns.histplot(data=mean_df, x='Sample_Mean', kde=True)
                  plt.tick_params(colors='white')
                  ax.set_xlabel("Sample_Mean", color="white")
                  ax.set_ylabel("Count", color="white")
                  st.pyplot(fig1)
             with col2:
                  st.text('List of sample means.')
                  st.dataframe(mean_df, height = 100)
                  st.text(' ')
         with st.container():
             with col1:
                  st.text(' ')
                  fig2 = plt.figure(figsize=(20,10))
                  sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
                  fig2.suptitle('Frequency distribution of sample standard deviation', color='white',  fontsize=35)
                  ax = sns.histplot(data=std_df, x='Sample_Std', kde=True)
                  plt.tick_params(colors='white')
                  ax.set_xlabel("Sample_Std", color="white")
                  ax.set_ylabel("Count", color="white")
                  st.pyplot(fig2)
             with col2:
                  st.text(' ')
                  st.text(' ')
                  st.text(' ')
                  st.text(' ')
                  st.text(' ')
                  st.text('List of sample standard deviations.')
                  st.dataframe(std_df, height = 100)
         with st.container():
             with col1:
                 with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
                      fig6, axs = plt.subplots(1, 3, figsize=(18, 6))
                      fig6.suptitle('Frequency distribution of sample proportion for n = 2, 100, 1000.', color='white', fontsize=35)
                      sns.histplot(data=p_df1, x='Sample_P (n=2)', color='teal', kde=True, ax=axs[0])
                      sns.histplot(data=p_df1, x="Sample_P (n=100)", color='skyblue', kde=True, ax=axs[1])
                      sns.histplot(data=p_df1, x="Sample_P (n=1000)", color='red', kde=True, ax=axs[2])
                      st.pyplot(fig6)
             with col2:
                  st.text(' ')
                  st.text(' ')
                  st.text(' ')
                  st.text(' ')
                  st.text(' ')
                  st.text('List of sample proportions for n = 2, 100, 1000.')
                  st.dataframe(p_df1, height = 100)
                  st.text(' ')              
         with st.container():
             with col1:
                  st.text(' ')
                  fig3 = plt.figure(figsize=(12,6))
                  sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
                  fig3.suptitle('Frequency distribution of sample proportion', color='white', fontsize=20)
                  ax = sns.histplot(data=p_df, x='Sample_P', kde=True)
                  plt.tick_params(colors='white')
                  ax.set_xlabel("Sample_P", color="white")
                  ax.set_ylabel("Count", color="white")
                  st.pyplot(fig3)
             with col2:
                  st.text('List of sample proportions.')
                  st.dataframe(p_df, height = 100)
                  st.text(' ')
         with st.container():
             with col1:
                 with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
                       fig5, axs = plt.subplots(1, 3, figsize=(18, 6))
                       fig5.suptitle('Frequency distribution of sample means for n = 2, 100, 1000.', color='white', fontsize=35)
                       sns.histplot(data=mean_df1, x='Sample_Mean (n=2)', color='teal', kde=True, ax=axs[0])
                       sns.histplot(data=mean_df1, x="Sample_Mean (n=100)", color='skyblue', kde=True, ax=axs[1])
                       sns.histplot(data=mean_df1, x="Sample_Mean (n=1000)", color='red', kde=True, ax=axs[2])
                       st.pyplot(fig5)
                 with col2:
                      st.text(' ')
                      st.text(' ')
                      st.text(' ')
                      st.text(' ')
                      st.text(' ')
                      st.text('List of sample means for n = 2, 100, 1000.')
                      st.dataframe(mean_df1, height = 100)              
         
         df_nnum = df[['Sales', 'Discount', 'Profit']]
        
         with col1:
              fig4 = plt.figure(figsize=(12,6))
              sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
              fig4.suptitle('Frequency distribution of Profie_rate', color='white', fontsize=20)
              ax = sns.histplot(data=superstore_df, x='Profit_rate', kde=True)
              plt.tick_params(colors='white')
              ax.set_xlabel("Profit_rate", color="white")
              ax.set_ylabel("Count", color="white")
              st.pyplot(fig4)        
         with col2:
              st.text(' ')
         with col2:
             with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
                  fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharex=True, sharey=True)
                  fig.suptitle('Checking Distribution and Outliers for Sales, Profit and Discount', color='white', fontsize=35)
                  plt.subplot(3,3,1)
                  plt.hist(df_nnum['Sales'], bins=200, color='#F05454')
                  plt.xlim(0,1000)
                  plt.subplot(3,3,2)
                  sns.boxplot(df_nnum['Sales'], color='#F05454')
                  plt.subplot(3,3,3)
                  sns.kdeplot(x=df_nnum['Sales'], color='#F05454')
                  plt.subplot(3,3,4)
                  plt.hist(df_nnum['Profit'], bins=200, color='#30475E')
                  plt.xlim(-250,300)
                  plt.subplot(3,3,5)
                  sns.boxplot(df_nnum['Profit'], color='#30475E')
                  plt.subplot(3,3,6)
                  sns.kdeplot(x=df_nnum['Profit'], color='#30475E')
                  plt.subplot(3,3,7)
                  plt.hist(df_nnum['Discount'], bins=10, color='#006400')
                  plt.xlim(0,1)
                  plt.subplot(3,3,8)
                  sns.boxplot(df_nnum['Discount'], color='#006400')
                  plt.subplot(3,3,9)
                  sns.kdeplot(x=df_nnum['Discount'], color='#006400')
             st.pyplot(fig)

if page == 'Main':
   col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
   pol3, pol4 = st.columns(2)
   if add_radio == "Local data": 
      superstore = load_data(9995)
      df = superstore
   if add_radio == "BigQuery":
      credentials = service_account.Credentials.from_service_account_info(
      st.secrets["gcp_service_account"]
      )
      client = bigquery.Client(credentials=credentials) 
      rows = run_query("SELECT * FROM predykcja-incomig-calls.Retail_Data_1.Retails_1")
      df =  pd.DataFrame(rows)
      df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')
      df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], format='%Y-%m-%d')
   
   with col1:
        st.text('Total Sales $')
        s = df['Sales']
        s = s.sum()
        my_string1 = '{:,.2f}'.format(s.sum())
        st.text(my_string1)
   with col2:
        st.text('Total Profit $')
        p = df['Profit']
        p = p.sum()
        my_string2 = '{:,.2f}'.format(p.sum())
        st.text(my_string2)
   with col3:
        st.text('Profit Ratio %')
        pr = df['Profit_rate'] = df['Profit']/df['Sales']
        pr = round(pr.sum()/100,2)
        st.text(pr)
   with col4:
        st.text('Profit per Order $')        
        df_ord_sum = df[['Order_ID', 'Profit']].groupby('Order_ID').sum().reset_index()
        ord = df_ord_sum['Profit']/len(df_ord_sum['Order_ID'])
        ord = round(ord.sum(),2)
        st.text(ord)
   with col5:
        st.text('Sales per Customer $')        
        df_cust_sum = df[['Customer_ID', 'Sales']].groupby('Customer_ID').sum().reset_index()
        cust = df_cust_sum['Sales']/len(df_cust_sum['Customer_ID'])
        my_string3 = '{:,.2f}'.format(cust.sum())
        st.text(my_string3)
   with col6:
        st.text('Avg. Discount per Product %')
        df_disc_mean = df[['Product_Name', 'Discount']].groupby('Product_Name').mean().reset_index()
        disc = df_disc_mean['Discount']/len(df_disc_mean['Product_Name'])*100
        disc = round(disc.sum(),2)
        st.text(disc)
   with col7:
        st.text('Quantity pcs')        
        q = df['Quantity']
        q = q.sum()
        my_string4 = '{:,.2f}'.format(q.sum())
        st.text(my_string4)
        
   total_month1 = df.groupby(['Segment', pd.Grouper(key='Order_Date', freq='M')])['Sales'].sum().reset_index() 
   segments = df['Segment'].unique()
   for i, s in enumerate(segments):
       t = len(segments)
       if i == 0:
          with pol3:
                                st.text("Total Sales (per month) for each Segment")
                                col = ['#cccccc']*len(segments) 
                                col_pall = ['#e60073', '#ff3333', '#ff944d'] 
                                fig1 = plt.figure(figsize = (18, 9))   
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})  
                                color_list = col.copy()
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month1, x = 'Order_Date', y = 'Sales', hue = 'Segment', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                                st.pyplot(fig1)
       if i == 1:
          with pol4:
                                st.text('.')
                                col = ['#cccccc']*len(segments) 
                                col_pall = ['#e60073', '#ff3333', '#ff944d']   
                                fig2 = plt.figure(figsize = (18, 9))   
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})  
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month1, x = 'Order_Date', y = 'Sales', hue = 'Segment', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                                st.pyplot(fig2)
       if i == t-1:
          with pol3:
                                st.text('.')
                                col = ['#cccccc']*len(segments) 
                                col_pall = ['#e60073', '#ff3333', '#ff944d']   
                                fig3 = plt.figure(figsize = (18, 9))   
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})  
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month1, x = 'Order_Date', y = 'Sales', hue = 'Segment', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                                st.pyplot(fig3)        
        

   total_month1 = df.groupby(['Category', pd.Grouper(key='Order_Date', freq='M')])['Sales'].sum().reset_index() 
   categories = df['Category'].unique() 
   with st.container():
           for i, s in enumerate(categories):
               c = len(categories)
               if i == 0:
                  with pol3:
                                st.text("Total Sales (per month) for each Category")
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
                                col = ['#cccccc']*len(categories) 
                                fig4 = plt.figure(figsize = (18, 9))
                                col_pall = ['#e60073', '#ff3333', '#ff944d']   
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month1, x = 'Order_Date', y = 'Sales', hue = 'Category', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')    
                                st.pyplot(fig4)
               if i == 1:
                  with pol4:
                                st.text(' ')
                                st.text(' ')       
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')       
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')       
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                
                                st.text('.')
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
                                col = ['#cccccc']*len(categories) 
                                fig5 = plt.figure(figsize = (18, 9))
                                col_pall = ['#e60073', '#ff3333', '#ff944d']   
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month1, x = 'Order_Date', y = 'Sales', hue = 'Category', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')    
                        
                                st.pyplot(fig5)
               if i == c-1:
                  with pol3:
                                st.text('.')
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"}) 
                                col = ['#cccccc']*len(categories) 
                                fig6 = plt.figure(figsize = (18, 9))
                                col_pall = ['#e60073', '#ff3333', '#ff944d']   
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month1, x = 'Order_Date', y = 'Sales', hue = 'Category', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                        
                                st.pyplot(fig6)        
           
   total_month = df.groupby(['Region', pd.Grouper(key='Order_Date', freq='M')])['Sales'].sum().reset_index() 
   regions = df['Region'].unique()     
   with st.container(): 
           for i, s in enumerate(regions):
               r = len(regions)
               if i == 0:
                  with pol3:
                                st.text("Total Sales (per month) for each Region")
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})      
                                col = ['#cccccc']*len(regions)
                                col_pall = ['#e60073', '#ff3333', '#ff944d', '#339900'] 
                                fig7 = plt.figure(figsize = (18, 9))
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month, x = 'Order_Date', y = 'Sales', hue = 'Region', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                        
                                st.pyplot(fig7)
               if i == 1:
                  with pol4:
                                st.text(' ')
                                st.text(' ')       
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')       
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')       
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')
                                st.text(' ')

                                st.text('.')
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})      
                                col = ['#cccccc']*len(regions) 
                                col_pall = ['#e60073', '#ff3333', '#ff944d', '#339900'] 
                                fig8 = plt.figure(figsize = (18, 9))
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month, x = 'Order_Date', y = 'Sales', hue = 'Region', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                         
                                st.pyplot(fig8)
               if i == 2:
                  with pol3:
                                st.text('.')
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})      
                                col = ['#cccccc']*len(regions) 
                                col_pall = ['#e60073', '#ff3333', '#ff944d', '#339900'] 
                                fig9 = plt.figure(figsize = (18, 9))
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month, x = 'Order_Date', y = 'Sales', hue = 'Region', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                        
                                st.pyplot(fig9)
               if i == r-1:
                  with pol4:
                                st.text('.')
                                sns.set(rc={'axes.facecolor':"#37383f", 'figure.facecolor':"#37383f"})      
                                col = ['#cccccc']*len(regions) 
                                col_pall = ['#e60073', '#ff3333', '#ff944d', '#339900'] 
                                fig10 = plt.figure(figsize = (18, 9))
                                color_list = col.copy() 
                                color_list[i] = col_pall[i] 
                                sns.lineplot(data = total_month, x = 'Order_Date', y = 'Sales', hue = 'Region', palette = color_list, legend=False)
                                plt.title(s, color="white", fontsize = 35) 
                                plt.xlabel(None)
                                plt.ylabel(None)
                                plt.tick_params(colors='white')
                         
                                st.pyplot(fig10)

   pol1, pol2 = st.columns([6, 1])
   
   with pol1:
       us_state = {
       "Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA",
       "Colorado": "CO","Connecticut": "CT","Delaware": "DE","Florida": "FL","Georgia": "GA",
       "Hawaii": "HI","Idaho": "ID","Illinois": "IL","Indiana": "IN","Iowa": "IA","Kansas": "KS",
       "Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI",
       "Minnesota": "MN","Mississippi": "MS","Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV",
       "New Hampshire": "NH","New Jersey": "NJ","New Mexico": "NM","New York": "NY","North Carolina": "NC",
       "North Dakota": "ND","Ohio": "OH","Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","Rhode Island": "RI",
       "South Carolina": "SC","South Dakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT",
       "Virginia": "VA","Washington": "WA","West Virginia": "WV","Wisconsin": "WI","Wyoming": "WY",
       "District of Columbia": "DC","American Samoa": "AS","Guam": "GU","Northern Mariana Islands": "MP","Puerto Rico": "PR",
       "United States Minor Outlying Islands": "UM","U.S. Virgin Islands": "VI",
       }
       df['State Code']= df['State'].apply(lambda x: us_state[x])
       data_2= df[['Sales','State Code','Region']]

       def transaction_2(data_2, region):
           mask= data_2['Region']==region
           df1= data_2[mask]
           df1= df1.groupby('State Code', as_index=False)['Sales'].sum()
           fig =px.choropleth(df1,locations='State Code',color='Sales',
                          locationmode="USA-states", scope='usa', template = "plotly_dark")                          
           fig.update_layout(title_text=f'Total Sale by States in Region: ({region})', title_x=0.5)               
           return fig

       transaction_fig_2= []
       for g in df['Region'].unique().tolist():
           transaction_fig_2.append(transaction_2(data_2,g))
       option = st.selectbox(
      'Chose Region of interest: 0 - South, 1 - West, 2 - Central, 3 - East',
      (0, 1, 2, 3))
       transaction_fig_2[option]     
       
       
   with pol1:
        DATA_URL="us-states.json"
        json_file_path =  DATA_URL
        with open(json_file_path, 'r') as j:
             contents = json.loads(j.read())

             superstore2_df = df
             states_df= superstore2_df.groupby("State",as_index=False)["Sales"].sum()
             fig_map= go.Figure(data=go.Choropleth(locations =states_df['State'], 
                                      z = states_df['Sales'],              
                                      geojson=contents,                    
                                      colorscale = 'peach',                
                                      colorbar_tickprefix = 'Billion US$', 
                                      colorbar_title = 'Sales',
                                      featureidkey='properties.name'
                                    )                                      
                 )
        fig_map.update_layout(geo_scope="usa", template = "plotly_dark",
        title={'text': 'Sales by states', 
              'y':0.9,
              'x':0.5,
              'xanchor': 'center',
              'yanchor': 'top'})

        fig_map
        
   pos1, pos2 = st.columns([2, 1])
        
   def ones(x):
       return 1
   df['Row_ID'] = df['Row_ID'].apply(ones)
       

   with pos1:
            st.subheader('Frequency occurences:')
            fig1 = px.sunburst(data_frame = df[['Category','Sub_Category','Row_ID']].groupby(['Category','Sub_Category']).sum().reset_index(),
                          path=['Category', 'Sub_Category'], values='Row_ID', title = 'Frequency of category occurences.',
                          color_discrete_sequence=px.colors.sequential.Electric)
            st.plotly_chart(fig1, sharing="streamlit", use_container_width=False)

   with pos2:
           
            st.text(' ')
            st.text(' ')       
            st.text(' ')
            st.text(' ')
            
            fig2 = px.sunburst(data_frame = df[['Region','City','Row_ID']].groupby(['Region','City']).sum().reset_index(),
                           path=['Region', 'City'], values='Row_ID', title = 'Frequency of regions and cities occurences.',
                           color_discrete_sequence=px.colors.sequential.Mint)
            st.plotly_chart(fig2, sharing="streamlit", use_container_width=False)
            
   with pos1:
            st.subheader('Top 20 products:')
            fig3 = px.sunburst(data_frame = df[['Category','Product_Name','Row_ID']].groupby(['Category','Product_Name']).sum().reset_index().sort_values('Row_ID',ascending=False).head(20),
                           path=['Category', 'Product_Name'], values='Row_ID', title = 'Top 20 products and their distribution across categories',
                           color_discrete_sequence=px.colors.sequential.Agsunset)
            st.plotly_chart(fig3, sharing="streamlit", use_container_width=False)


if page == 'Quantitative data analysis':
   if add_radio == "Local data": 
      superstore = load_data(9995)
      df = superstore
   if add_radio == "BigQuery":
      credentials = service_account.Credentials.from_service_account_info(
      st.secrets["gcp_service_account"]
      )
      client = bigquery.Client(credentials=credentials) 
      rows = run_query("SELECT * FROM predykcja-incomig-calls.Retail_Data_1.Retails_1")
      df =  pd.DataFrame(rows)
      
   df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')
   df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], format='%Y-%m-%d')
      
   col1, col2 = st.columns(2)

   df['Year'] = pd.to_datetime(df['Order_Date']).dt.year
   df['mm.order_date'] = pd.to_datetime(df['Order_Date']).dt.month
   df['dd.order_date'] = pd.to_datetime(df['Order_Date']).dt.day

   df_copy = (df.copy(deep=True)).reset_index()
   df_copy.rename(columns = {'year' : 'Year'},inplace=True)
   df_prod = pd.pivot_table(df_copy, values = ['Sales', 'Profit'], index = ['Year', 'Category', 'Sub_Category'],
                           aggfunc=sum).sort_values(['Year', 'Profit'], ascending = False)

   df_prod_copy = df_prod.copy(deep=True).reset_index().drop(['Sales','Category'], axis=1)
   df_prod_copy = df_prod_copy.groupby('Sub_Category')['Profit'].agg('sum')
   df_prod_copy = pd.DataFrame(df_prod_copy).sort_values(by = ['Profit'] , ascending = False).reset_index()

   sub_c = df_prod_copy['Sub_Category']
   profit = df_prod_copy['Profit']

   with col1:
        fig1, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig1.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):
             col = 'Ship_Mode'
             plt.subplot(3,1,1)
             plt.title('Average of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).mean()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig1)
   with col2:
        fig2, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig2.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):                          
             col = 'Segment'
             plt.subplot(3,1,1)
             plt.title('Average of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).mean()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig2)
   with col1:
        fig3, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig3.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):                                
             col = 'Region'
             plt.subplot(3,1,1)
             plt.title('Average of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).mean()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig3)
   with col2:
        fig4, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig4.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):                                             
             col = 'Category'
             plt.subplot(3,1,1)
             plt.title('Average of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).mean()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig4)             
   with col1:
        fig5, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig5.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):
             col = 'Ship_Mode'
             plt.subplot(3,1,1)
             plt.title('Sum of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).sum()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig5)
   with col2:
        fig6, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig6.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):                         
             col = 'Segment'
             plt.subplot(3,1,1)
             plt.title('Sum of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).sum()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig6)
   with col1:
        fig7, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig7.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):                                          
             col = 'Region'
             plt.subplot(3,1,1)
             plt.title('Sum of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).sum()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig7)
   with col2:
        fig8, axis = plt.subplots(figsize=(20, 35), sharex=True, sharey=True)
        axis = fig8.add_subplot(121)
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#131314'}):                                 
             col = 'Category'
             plt.subplot(3,1,1)
             plt.title('Sum of profit by ' + col + '.', color = 'white', fontsize = 35)
             df_cat_mean = df[[col, 'Profit']].groupby(col).sum()
             plt.bar(df_cat_mean.index, df_cat_mean['Profit'], color=['#518D95', '#96547B', '#8C8466', '#5F5E8F'])
             st.pyplot(fig8)                         
   with col2:

            df_for_plot = df[['Year','Sales','Quantity','Profit']].copy(deep=True)
            df_for_plot = df_for_plot.groupby(['Year']).sum()
            df_for_plot = df_for_plot.reset_index()
            fig, (ax1,ax3) = plt.subplots(1,2)
            ax1.grid()
            fig.set_figwidth(12)
            fig.set_figheight(4)
            color = '#0024be'
            ax1.set_xlabel('Year', fontsize = 20)
            ax1.set_ylabel('sales', color = color, fontsize = 20)
            ax1.plot(df_for_plot['Year'],df_for_plot['Sales'],
                    marker = 'o',
                    markersize = 6,
                    color = '#0024be',
                    markerfacecolor = '#2500fb',
                    markeredgecolor = '#0bbdc6',
                    markeredgewidth = 3)
            ax1.xaxis.label.set_color('white')
            ax1.yaxis.label.set_color('white')
            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='white')

            ax2 = ax1.twinx()

            color = 'darkgreen'
            ax2.set_ylabel('quantity', color = color, fontsize = 20)
            ax2.plot(df_for_plot['Year'],df_for_plot['Quantity'],
                    marker = 'o',
                    markersize = 6,
                    color = 'darkgreen',
                    markerfacecolor = 'lawngreen',
                    markeredgecolor = 'darkgreen',
                    markeredgewidth = 3)
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
            ax2.tick_params(axis='x', colors='white')
            ax2.tick_params(axis='y', colors='white')
            
            fig.tight_layout()

            color = '#0024be'
            ax3.grid()
            ax3.set_xlabel('Year', fontsize = 20)
            ax3.set_ylabel('sales', color = color, fontsize = 20)
            ax3.plot(df_for_plot['Year'],df_for_plot['Sales'],
                   marker = 'o',
                   markersize = 6,
                   color = '#0024be',
                   markerfacecolor = '#2500fb',
                   markeredgecolor = '#0bbdc6',
                   markeredgewidth = 3)
            ax3.xaxis.label.set_color('white')
            ax3.yaxis.label.set_color('white')
            ax3.tick_params(axis='x', colors='white')
            ax3.tick_params(axis='y', colors='white')
            
            ax4 = ax3.twinx()

            color = '#ec0014'
            ax4.set_ylabel('Profit', color = color, fontsize = 20)
            ax4.plot(df_for_plot['Year'],df_for_plot['Profit'],
                   marker = 'o',
                   markersize = 6,
                   color = '#ec0014',
                   markerfacecolor = '#b7000f',
                   markeredgecolor = '#f96b00',
                   markeredgewidth = 3)
            ax4.xaxis.label.set_color('white')
            ax4.yaxis.label.set_color('white')
            ax4.tick_params(axis='x', colors='white')
            ax4.tick_params(axis='y', colors='white')
            
            fig.tight_layout()
            st.pyplot(fig)

            st.subheader('Conclusions:')
            st.caption("-The graph shows that since 2015, the company's profit, quantity and sales have been growing.")
            st.caption('-Sales declined in 2015, but profits did not fall.')
            st.caption('-The largest increase in profits and sales was in 2016.')
            st.caption('-In 2017, the growth of the store went down, perhaps the store reached its limit? It is necessary to consider other indicators and analyze the activities of competitors.')

   with col1:
            i = 1
            df_for_plot['Sales % to the previous year'] = None
            for x in df_for_plot['Sales']:
                if x == df_for_plot['Sales'].loc[0]:
                    continue
                df_for_plot.loc[i, 'Sales % to the previous year'] = str(round(((x / df_for_plot['Sales'].loc[i-1]*100)-100),2))+'%'
                i = i + 1

            i = 1
            df_for_plot['Profit % to the previous year'] = None
            for x in df_for_plot['Profit']:
                if x == df_for_plot['Profit'].loc[0]:
                    continue
                df_for_plot.loc[i, 'Profit % to the previous year'] = str(round(((x / df_for_plot['Profit'].loc[i-1]*100)-100),2))+'%'
                i = i + 1
            df_for_plot

   with col1:
        df['Year'] = pd.to_datetime(df['Order_Date']).dt.year
        df_sort = df.sort_values('Year', axis = 0, ascending = True,inplace = True, na_position ='last')
        ylist = df['Year'].unique()

        df_num = df[['Ship_Mode', 'Sales', 'Quantity', 'Discount', 'Profit']]
        df_num_shipmode_change = df_num.replace({'Same Day': 0, 'First Class': 1, 'Second Class': 2, 'Standard Class': 3})
        df_num_shipmode_change['Year'] = pd.to_datetime(df['Order_Date']).dt.year
        df_sales_profit = df_num_shipmode_change[['Sales', 'Profit', 'Year']].set_index(pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')).sort_index()
        df_sales_profit = df_sales_profit.groupby('Order_Date').mean()

        year_f = str(st.selectbox("Select year from:", ylist, index = 0))
        year_t = str(st.selectbox("Select year to:", ylist))
        
        year_f_dt = pd.to_datetime(year_f)
        year_f_dt = year_f_dt.year
        year_t_dt = pd.to_datetime(year_t)
        year_t_dt = year_t_dt.year
        
        if year_f != year_t:
           st.text('')
           st.text('')
           st.text('')
           st.text('') 
           with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#37383f'}): 
                fig, ax = plt.subplots(1, figsize=(20, 8))
                fig.suptitle('Average Sales and Profit over Time Period', color = 'white', fontsize = 35)
                ax.plot(df_sales_profit[(df_sales_profit['Year'] >= year_f_dt) & (df_sales_profit['Year'] <= year_t_dt)].index, 'Sales', data=df_sales_profit[(df_sales_profit['Year'] >= year_f_dt) & (df_sales_profit['Year'] <= year_t_dt)], linewidth=2)
                ax.plot(df_sales_profit[(df_sales_profit['Year'] >= year_f_dt) & (df_sales_profit['Year'] <= year_t_dt)].index, 'Profit', data=df_sales_profit[(df_sales_profit['Year'] >= year_f_dt) & (df_sales_profit['Year'] <= year_t_dt)], linewidth=2)
           st.pyplot(fig)
        if year_f == year_t:
           st.text('')
           st.text('')
           st.text('')
           st.text('') 
           with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#37383f'}):
                fig, ax = plt.subplots(1, figsize=(20, 8))
                fig.suptitle('Average Sales and Profit over Time Period', color = 'white', fontsize = 35)
                ax.plot(df_sales_profit[df_sales_profit['Year'] == year_t_dt].index, 'Sales', data=df_sales_profit[df_sales_profit['Year'] == year_t_dt], linewidth=2)
                ax.plot(df_sales_profit[df_sales_profit['Year'] == year_t_dt].index, 'Profit', data=df_sales_profit[df_sales_profit['Year'] == year_t_dt], linewidth=2)
           st.pyplot(fig)
   
   with col2:
        with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'#37383f'}):       
             fig, ax = plt.subplots(1, figsize=(20, 8))        
             time_df = df[['Order_Date', 'Sales', 'Profit']].sort_values(by='Order_Date')
             time_df['MonthYr'] = pd.to_datetime(df['Order_Date']).dt.to_period('M')
             time_df_avg = time_df.groupby('MonthYr').agg({'Sales':'mean', 'Profit':'mean'}).reset_index()
             time_df['Year'] = pd.to_datetime(time_df['Order_Date']).dt.year
             
             if year_f == year_t:
                ax.plot(time_df_avg[time_df['Year'] == year_t_dt].index, time_df_avg['Sales'][time_df['Year'] == year_t_dt], color='steelblue', label='Sales', linewidth=3)
                ax.plot(time_df_avg[time_df['Year'] == year_t_dt].index, time_df_avg['Profit'][time_df['Year'] == year_t_dt], color='darkorange', label='Profit', linewidth=3)
                plt.xlabel('Order_Date (Year-Month)', color = 'white')
                labels = time_df_avg['MonthYr'].values
                plt.xticks(range(1,time_df_avg.shape[0]+1), labels=labels)
                plt.xticks(rotation=90)
                plt.ylim([-100, 350])
                plt.ylabel('Sales/Profit Average', color = 'white')
                plt.legend(labelcolor = 'white', fontsize=14)
                plt.title('Sales & Profit Average Over Time (per Month)', color = 'white', fontsize = 35)       
             if year_f != year_t:
                ax.plot(time_df_avg[(time_df['Year'] >= year_f_dt) & (time_df['Year'] <= year_t_dt)].index, time_df_avg['Sales'][(time_df['Year'] >= year_f_dt) & (time_df['Year'] <= year_t_dt)], color='steelblue', label='Sales', linewidth=3)
                ax.plot(time_df_avg[(time_df['Year'] >= year_f_dt) & (time_df['Year'] <= year_t_dt)].index, time_df_avg['Profit'][(time_df['Year'] >= year_f_dt) & (time_df['Year'] <= year_t_dt)], color='darkorange', label='Profit', linewidth=3)
                plt.xlabel('Order_Date (Year-Month)', color = 'white')
                labels = time_df_avg['MonthYr'].values
                plt.xticks(range(1,time_df_avg.shape[0]+1), labels=labels)
                plt.xticks(rotation=90)
                plt.ylim([-100, 350])
                plt.ylabel('Sales/Profit Average', color = 'white')
                plt.legend(labelcolor = 'white', fontsize=14)
                plt.title('Sales & Profit Average Over Time (per Month)', color = 'white', fontsize = 35)        
                 
        st.pyplot(fig)
             
   with col1:
        fig, ax5 = plt.subplots(figsize=(15, 9))
        ax5.barh(sub_c, profit, color = "#128277")

        for s in ['top', 'bottom', 'left', 'right']:
            ax5.spines[s].set_visible(False)

        ax5.xaxis.set_ticks_position('none')
        ax5.yaxis.set_ticks_position('none')

        ax5.xaxis.set_tick_params(pad=5)
        ax5.yaxis.set_tick_params(pad=10)
        
        ax5.tick_params(axis='x', colors='white')
        ax5.tick_params(axis='y', colors='white')
        
        ax5.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.5,
                alpha=0.2)

        ax5.invert_yaxis()

        for i in ax5.patches:
                txt = plt.text(i.get_width()+0.2, i.get_y()+0.5,
                     str(int(i.get_width())),
                     fontsize=20, fontweight='bold',
                     color='white')
                txt.set_path_effects([pe.Stroke(linewidth=3, foreground='black'),
                               pe.Normal()])

        ax5.set_title('Profit by sub-category',
                     loc='center', fontsize = 35, color = 'white')

        st.pyplot(fig)

   with col2:
        df_prod = pd.pivot_table(df_copy, values = ['Sales', 'Profit'], index = ['Year', 'Category', 'Sub_Category'],
                  aggfunc='sum').sort_values(['Year', 'Profit'], ascending = False)
        df_prod_copy = df_prod.copy(deep=True).reset_index().drop(['Sales','Sub_Category'], axis=1)
        df_prod_copy = df_prod_copy.groupby([ 'Year', 'Category']).agg({'Profit' : 'sum'}).reset_index()
        profit = df_prod_copy['Profit']
        sub_c = df_prod_copy['Category']
        
        with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
             plt1 = plt.figure(figsize=(15,8))
             ax111 = plt1.add_subplot(211)
             ax222 = plt1.add_subplot(212)        
             plt1.set_figwidth(15)
             plt1.set_figheight(8)
             ax111.set_ylabel("Profit", fontsize = 25)
             ax111.set_title('Profit by year and category', color='white', fontsize = 35)
             ax111.legend(df_prod_copy['Category'].unique(), loc = "lower left")
             ax111.grid(color='gray', linestyle='dotted')

             ax111.plot(df_prod_copy['Year'][df_prod_copy['Category']=='Furniture'],
                        df_prod_copy['Profit'][df_prod_copy['Category']=='Furniture'],
                        color = '#232122',
                        marker = "d",
                        label = 'Furniture', linewidth = 2.5)
             ax111.plot(df_prod_copy['Year'][df_prod_copy['Category']=='Technology'],
                        df_prod_copy['Profit'][df_prod_copy['Category']=='Technology'],
                        color = '#A5C05B',
                        marker = "d",
                        label = "Technology",linewidth = 2.5)
             ax111.plot(df_prod_copy['Year'][df_prod_copy['Category']=='Office Supplies'],
                        df_prod_copy['Profit'][df_prod_copy['Category']=='Office Supplies'],
                        color = '#7BA4A8',
                        marker = "d",
                        label = 'Office Supplies', linewidth = 2.5)
             ax111.legend(df_prod_copy['Category'].unique(), loc = "upper left")
             ax111.legend(bbox_to_anchor =(1,1), ncol = 2, labelcolor='white')


             df_prod_copy = df[['Year','Category','Quantity']].copy(deep=True)
             df_prod_copy =  df_prod_copy.groupby(['Year', 'Category']).agg({'Quantity' : 'sum'}).reset_index()
            
             ax222.set_xlabel("Year", fontsize = 25)
             ax222.set_ylabel("Quantity", fontsize = 25)
             ax222.set_title('Quantity by year and category', color='white', fontsize = 35)
             plt.legend(labelcolor='linecolor')
             ax222.legend(df_prod_copy['Category'].unique(), loc = "lower left")
             ax222.grid(color='gray', linestyle='dotted')


             ax222.plot(df_prod_copy['Year'][df_prod_copy['Category']=='Furniture'],
                        df_prod_copy['Quantity'][df_prod_copy['Category']=='Furniture'],
                        color = '#232122',
                        marker = "d",
                        label = 'Furniture', linewidth = 2.5)
             ax222.plot(df_prod_copy['Year'][df_prod_copy['Category']=='Technology'],
                        df_prod_copy['Quantity'][df_prod_copy['Category']=='Technology'],
                        color = '#A5C05B',
                        marker = "d",
                        label = "Technology", linewidth = 2.5)
             ax222.plot(df_prod_copy['Year'][df_prod_copy['Category']=='Office Supplies'],
                        df_prod_copy['Quantity'][df_prod_copy['Category']=='Office Supplies'],
                        color = '#7BA4A8',
                        marker = "d",
                        label = 'Office Supplies', linewidth = 2.5)
             ax222.legend(bbox_to_anchor =(1,1), ncol = 2, labelcolor='white')
        st.pyplot(plt1)

   st.subheader('Conclusions:')
   st.caption('categories "Office Supplies" and "Technology" bring the most profit.Technology is the fastest growing category (profits have more than doubled since 2014), furniture sales tend to rise and fall each year, staying at the same level.')
   st.caption('Top 3 subcategories by profit: "Copiers", "Phones", "Accessories". Despite the highest profit, the "Technology" category is not sold in huge quantities (increase by 1k in three years), the "Office supplies" category increased in quantity from 4.5k to 7.6k, being the top seller.')

#3. Profit looses

if page == 'Profit looses':
   if add_radio == "Local data": 
      superstore = load_data(9995)
      df = superstore
   if add_radio == "BigQuery":
      credentials = service_account.Credentials.from_service_account_info(
      st.secrets["gcp_service_account"]
      )
      client = bigquery.Client(credentials=credentials) 
      rows = run_query("SELECT * FROM predykcja-incomig-calls.Retail_Data_1.Retails_1")
      df =  pd.DataFrame(rows)
      df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')
      df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], format='%Y-%m-%d')
   col1, col2 = st.columns(2)

   df['Year'] = pd.to_datetime(df['Order_Date']).dt.year
   df['mm.order_date'] = pd.to_datetime(df['Order_Date']).dt.month
   df['dd.order_date'] = pd.to_datetime(df['Order_Date']).dt.day

   df_copy = (df.copy(deep=True)).reset_index()
   df_prod = pd.pivot_table(df_copy, values = ['Sales', 'Profit'], index = ['Year', 'Category', 'Sub_Category'],
                           aggfunc=sum).sort_values(['Year', 'Profit'], ascending = False)

   df_neg_val = df_prod.loc[df_prod.Profit < 0, :]
   df_neg_val = df_neg_val.reset_index()
   df_neg_val['Profit'] = df_neg_val['Profit'] * (-1)

   with col1:
        with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
               def make_autopct(values):
                   def my_autopct(pct):
                       total = sum(values)
                       val = int(round(pct*total/100.0))
                       return '{p:.1f}% ({v:d})'.format(p=pct,v=val)
                   return my_autopct
               
               fig1 = plt.figure(figsize=(6,6))
               ax = plt.subplot(111)
               explode = (0.20, 0.10, 0.10)
               plot1 = df_neg_val.groupby(['Category']).sum().plot.pie(y = 'Profit',ax=ax, autopct=make_autopct(df_neg_val['Profit']),
                                                    explode = explode, colors = ['#518D95','#96547B','#8C8466'],
                                                    wedgeprops = {"edgecolor" : "Black",'linewidth' : 2,'antialiased': True},
                                                    radius = 1.7, labeldistance = 1.1,startangle=220, textprops = {'color':"white", 'fontsize':'20', 'fontweight':'bold'})
               plt.legend(bbox_to_anchor=(1,1.3),loc="upper left", labelcolor='white')
               plt.title('Profit losses by category', loc = 'center', pad = 65.0, color='white', fontsize = 20)
               plt.ylabel("")
               st.pyplot(fig1)

   with col2:
        with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
               fig2 = plt.figure(figsize=(15,8))
               ax = plt.subplot(222)
               df_neg_val['Profit'] = df_neg_val['Profit'] * (-1)
               plot2 = df_neg_val.pivot(index='Year',columns = 'Sub_Category',
                                       values = 'Profit').plot.bar(ax=ax,color =
                                       ['#518D95','#96547B','#8C8466','#5F5E8F'])
        
               plt.title('Profit looses', fontsize = 15, color='white')
               plt.xlabel("Year", fontsize = 10)
               plt.ylabel("Profit", fontsize = 10)
               plt.legend(labelcolor='white')
               st.pyplot(fig2)
               
   region_profit = df.groupby('City')['Profit'].sum().sort_values(ascending = False)
   negative_region = region_profit[region_profit < 0]
   top_negative_region = negative_region.tail(10)

   with col1:
        with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
               fig3 = plt.figure(figsize=(25,10))
               ax = plt.subplot(333)            
               sns.barplot(top_negative_region.index, top_negative_region, ax=ax)
               for i,v in enumerate(top_negative_region):
                   plt.text(i-0.2,v - 800,str(int(v)), color='white')
               plt.title('Top 10 lowest profit made cities', fontsize = 15, color='white')
               plt.xticks(rotation = 90);
               st.pyplot(fig3)
   with col2:
       st.subheader('Conclusions:')
       st.caption('Most of the losses are in the category "Furniture" (82.5%). The amount of losses is 21.4k.')
       st.caption('Every year the store sells tables at a loss. Perhaps the size of the discount affected the fact that the company loses money on the sale of tables. The company also loses profit on the sale of: machines, bookcases, supplies.')
       
   grouped_df = df.groupby('State').sum()[['Profit']].reset_index().sort_values(by='Profit', ascending=False)
   negative_state = grouped_df[grouped_df['Profit'] < 0]
   with col1:
        st.text('Negative Profit States Analysis')
        st.dataframe(negative_state)
        
   states_neg_pro = ['Oregon', 'Florida', 'Arizona', 'Tenessee', 'Colorado', 'North Carolina', 'Illinois', 'Pennsylvania', 'Ohio', 'Texas']
   states_neg_pro_df = df[df['State'].isin(states_neg_pro)]     
   with col1:
        states_neg_pro_df1 = states_neg_pro_df[states_neg_pro_df['Category']=='Technology']
        with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
            g = sns.lmplot(
                data=states_neg_pro_df1,
                x="Sales", y="Profit", col='Sub_Category',
                height=5, aspect=1, col_wrap=2
                )
            g.add_legend()
            st.pyplot(g)
        
   with col2:
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')   
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
       st.text('')
   
       states_neg_pro_df2 = states_neg_pro_df[states_neg_pro_df['Category']=='Furniture']
       with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
            g = sns.lmplot(
                data=states_neg_pro_df2,
                x="Sales", y="Profit", col='Sub_Category', hue='Region', hue_order=['South', 'East', 'Central', 'West'],
                height=5, aspect=1, col_wrap=2
                )
            g.add_legend(labelcolor='white')
            st.pyplot(g)
   with col1:
       states_neg_pro_df3 = states_neg_pro_df[states_neg_pro_df['Category']=='Office Supplies']
       with plt.rc_context({'axes.facecolor':"#37383f", 'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'figure.facecolor':'#37383f'}):
            g = sns.lmplot(
                data=states_neg_pro_df3,
                x="Sales", y="Profit", col='Sub_Category', hue='Region', hue_order=['South', 'East', 'Central', 'West'],
                height=5, aspect=1, col_wrap=3
                )
            g.add_legend(labelcolor='white')
            st.pyplot(g)
                    
