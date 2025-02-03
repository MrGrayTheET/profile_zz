from sc_loader import sierra_charts as sc
from ts_analysis.momentum_signals import  momentum as mom
from finance_models import ml_model, feature_builder
from finance_models.utils import clean_data
from ts_analysis.momentum_signals import momentum as momo
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np

sch = sc()
ticker = 'CL_F'
momo_cl = momo(sch.get_chart(ticker, formatted=True), ticker,intraday_period='30min',
               daily_len=5,daily_ATR=5, intraday_ATR=12,
               intraday_length=10, kama_periods=12, kama_fast_slow=[3,30], kama_daily=9)

momo_cl.daily_trend_signal()


ml_df = pd.DataFrame(index=momo_cl.data.index)
stationary_cols = ['vol_signal',
       'RVOL', 'RVOL_ma', 'momentum', 'mom_sma_s', 'mom_sma_l', 'momXsma1',
       'momXsma2', 'mSMA1xmSMA2', 'Volume']

nonstationary_cols = ['Open', 'High', 'Low', 'PD_High', 'PD_Low','PD_Close','PD_Open', 'High_3d', 'Low_3d', 'High_5d',
       'Low_5d', 'High_10d', 'Low_10d', 'KAMA', '']

for i in nonstationary_cols:

    ml_df[f'{i}_x'] = (momo_cl.data.Close - momo_cl.data[i])/momo_cl.data.Close



ml_df[stationary_cols] = momo_cl.data[stationary_cols]
daily_fwd_ret = momo_cl.data_1d.Close.pct_change(-2)

target = momo_cl.data.Close.pct_change(-2)
ml_df['target'] = target
ml_df['prev_ret'] = momo_cl.data.Close.pct_change()
ml_df['EMA_x'] = momo_cl.data.EMA1_x
ml_df['EMA_x2'] = momo_cl.data.EMA2_x
ml_df['daily_momentum'] = momo_cl.data['daily_momentum']
ml_df['daily_kama_x'] = (momo_cl.data.Close - momo_cl.data['daily_KAMA'])/momo_cl.data.Close

features=['Low_x', 'High_x',
          'vol_signal', 'RVOL','daily_kama_x',
          'daily_momentum','mSMA1xmSMA2']

ml_df_2 = clean_data(ml_df[features+['target']])

cols = ml_df.columns
cl_mom_mod = ml_model.ml_model(ml_df_2,features=features, target_column='target')
gbr_params = {'max_features': [4,5,6],
              'learning_rate':[0.02],
              'n_estimators':[200],
              'subsample':[0.6],
              'random_state':[42]}

gbr_model = cl_mom_mod.tree_model(gbr_params, gbr=True)









