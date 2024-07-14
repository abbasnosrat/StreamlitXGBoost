import streamlit as st
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jdatetime
from matplotlib.style import use
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from tqdm.auto import trange
from stqdm import stqdm

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def to_georgian(date):
    date  = date.split("/")
    date = jdatetime.date(year=int(date[0]), month=int(date[1]), day=int(date[2])).togregorian()
    return date


def make_ticks(year,month):
    ticks = []
    for i in range(len(year)):
        ticks.append(f"{int(year[i])}/{int(month[i])}")
    return np.array(ticks)

def train_booster(df_t, params, train_size=-5):

    dg = df_t.groupby(["Year", "Month"]).agg({"SaleAmount":"sum"}).reset_index()\
        .sort_values(["Year","Month"],ignore_index=True).rename(columns={"SaleAmount":"y"})
    
    for i in range(int(params["n_lags"])):
   
        dg[f"y_lag_{i+1}"] = dg["y"].shift((i+1))
    dg = dg.fillna(0)

    y = dg["y"].to_numpy()
    X = dg.drop(columns="y").to_numpy()
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    param_dist = {
            'max_depth': stats.randint(1, 100),
            'learning_rate': stats.uniform(0.01, 0.1),
            'subsample': stats.uniform(0.5, 0.5),
            'n_estimators':stats.randint(1, 200)
        }

        # Create the XGBoost model object
    xgb_model = xgb.XGBRegressor()

        # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=30, cv=5, scoring="neg_mean_absolute_error")

        # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X_train, y_train)

    pred = random_search.predict(X_test)
    
    mse= mean_absolute_percentage_error(y_test, pred)
    
    return random_search, mse



def ar_prediction(model:XGBRegressor, X_init:np.ndarray, horizon:int):
    Xs = []
    preds = []
    X_init = X_init.copy()
    # print(X_init)
    year = np.array(X_init[0]).reshape(1)
    month = np.array(X_init[1]).reshape(1)
    lags = X_init[2:]
    for i in range(horizon):
        X_pred = np.concatenate([year, month, lags]).reshape([1,-1])
        # print(X_pred)
        pred = model.predict(X_pred)
        Xs.append(X_pred.copy())
        preds.append(pred.copy())
        lags = np.concatenate([lags[1:], np.array(pred).reshape(-1)])
        month += 1
        if month > 12:
            month = 1
            year += 1
            
    return np.concatenate(Xs, axis=0), np.concatenate(preds).reshape(-1)


st.write("""
         # XGBoost
         """)
st.sidebar.write("Controls")
file = st.sidebar.file_uploader("Upload Your Dataset", type=".csv")

# df = pd.read_csv("SalesData.csv") if file is None else pd.read_csv(file)
try:
    df = pd.read_csv(file)
    got_data = True
except: got_data = False

if got_data:
    products = list(df.GoodName.unique())
    product = st.sidebar.selectbox(label="Please select a product", options=products)
    horizon = int(st.sidebar.slider(label="Select Prediction Horizon", min_value=2, max_value=30, value=5))
    test_size_manual = st.sidebar.number_input(label="Select Test Size", min_value=0, max_value=30, value=0)
    manual = st.sidebar.checkbox("Manual Mode")
    n_lags_manual=float(st.sidebar.text_input(label="Select Number of lags", value=1))
    

    df_t = df.query(f"GoodName == '{product}'").reset_index(drop=True)

    df_t["Year"] = df_t["StrFactDate"].apply(lambda d: int(d.split("/")[0]))
    df_t["Month"] = df_t["StrFactDate"].apply(lambda d: int(d.split("/")[1]))
    train_size = -5 if test_size_manual == 0 else -test_size_manual
    if not manual:
        mses = []
        models = []
        lags  = range(10)
        # lags.set_description("Tuning the model. Please be patient")
        for lag in lags:
            
            model, mse = train_booster(df_t, {"n_lags": lag}, train_size)
            mses.append(mse)
            models.append(model)
        
        best = {"n_lags": np.argmin(mses)}
        best_params = models[int(best["n_lags"])].best_params_
    else:
        best = {"n_lags": n_lags_manual}
        model, mse = train_booster(df_t, {"n_lags": n_lags_manual}, train_size)
        best_params = model.best_params_







    n_lags = int(best["n_lags"])



    dg = df_t.groupby(["Year", "Month"]).agg({"SaleAmount":"sum"}).reset_index()\
        .sort_values(["Year","Month"],ignore_index=True).rename(columns={"SaleAmount":"y"})
    
    for i in range(n_lags):
        dg[f"y_lag_{i+1}"] = dg["y"].shift((i+1))

    dg = dg.fillna(0)
    y = dg["y"].to_numpy()
    X = dg.drop(columns="y").to_numpy()
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    pred = model.predict(X)
    ticks = make_ticks(X[:,0], X[:,1])
    fig_tuned = go.Figure()
    fig_tuned.add_trace(go.Scatter(x=ticks, y=y, mode='markers', name='observations', marker=dict(color='black')))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred, mode='lines', name='predictions'))

    fig_tuned.add_vline(x=ticks[train_size], line=dict(dash='dash', color='black'), name='split')

    fig_tuned.update_layout(title='One Step Ahead Prediction', xaxis_title='Date', yaxis_title='Sales Amount')
    st.plotly_chart(fig_tuned)
    # plt.plot(ticks,y, ".k", label="observations")
    # plt.plot(pred, label="prediction")

    # plt.axvline(len(y) + train_size, linestyle ="--", color="k",label="Split")

    # plt.xticks(ticks, rotation=270)
    # plt.title(f"n_lags = {n_lags}")
    # plt.legend()
    # plt.savefig("ost_pred.jpg")
    
    model = XGBRegressor(**best_params)
    model.fit(X, y)
    Xs, preds = ar_prediction(model, X[-1], horizon)
    ticks = make_ticks(Xs[:,0], Xs[:,1])
    fig_ar = go.Figure()
    fig_ar.add_trace(go.Scatter(x=ticks, y=preds, mode='lines', name='predictions'))
    fig_ar.update_layout(title='AutoRegressive Prediction', xaxis_title='Date', yaxis_title='Sales Amount')
    st.plotly_chart(fig_ar)
    # plt.figure()
    df_pred = pd.DataFrame({"Date":ticks, "Yhat":preds})
    csv = convert_df(df_pred)
    st.download_button(
    "Download Results",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

    # plt.plot(preds, label="prediction")
    # plt.savefig("ar_pred.jpg")
else:
    
    st.write("Please uppload your data")
    df = pd.read_csv("SalesData.csv")[["GoodName", "StrFactDate", "SaleAmount"]]
    csv = convert_df(df)
    st.download_button("Sample Data", csv, "SampleData.csv","text/csv",
    key='download-csv')