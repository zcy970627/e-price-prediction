from tensorflow.keras.models import load_model
import pickle
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import datetime as dt
from apps import extract_data
#Load NN models
model_uni_day = load_model('models/uni_model_1day.h5')
model_uni_hour = load_model('models/uni_model_1hour.h5')
model_uni_week = load_model('models/uni_model_1week.h5')

model_multi_hour= load_model('models/multi_model_1hour.h5')
model_multi_day= load_model('models/multi_model_1dayl.h5')
model_multi_week= load_model('models/multi_model_1week.h5')
#Load the used scaler
scalerfile = 'models/scaler_uni_hourly.sav'
scaler_uni_hourly = pickle.load(open(scalerfile, 'rb'))

scalerfile_1 = 'models/scaler_multi_hourly.sav'
scaler_multi_hourly = pickle.load(open(scalerfile_1, 'rb'))
scaler_multi_hourly.clip=False

# scaler_uni_daily = 'models/scaler_uni.sav'
# scaler_uni_daily = pickle.load(open(scaler_uni_daily, 'rb'))

# scaler_multi_daily = 'models/scaler_multi.sav'
# scaler_multi_daily = pickle.load(open(scaler_multi_daily, 'rb'))

# #Load data
end_date = str(date.today())
start_date = str(date.today() - timedelta(days=30))

freq_elec = 'Base'
freq_weather = 24
idx = pd.date_range(start_date, str(datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=1)), freq='D')
elec_price = extract_data.get_elecprice(start_date, end_date, idx, freq_elec)
leipzig_weather = extract_data.get_weather(start_date, end_date, freq_weather)
oil_price = extract_data.get_oil_price(start_date, end_date, idx)
gas_price = extract_data.get_gas_price(start_date, end_date, idx)
wti_prices = extract_data.get_wti_oil_price(start_date, end_date, idx)
total_df_daily = pd.concat([leipzig_weather, oil_price, gas_price, wti_prices, elec_price], axis=1)
total_df_daily.to_csv('dataset/daily_historical_data.csv')


freq_elec='Hours'
freq_weather = 1
idx = pd.date_range(start_date, str(datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=1)), freq='1H')
elec_price = extract_data.get_elecprice(start_date, end_date, idx, freq_elec)
leipzig_weather = extract_data.get_weather(start_date, end_date, freq_weather)
oil_price = extract_data.get_oil_price(start_date, end_date, idx)
gas_price = extract_data.get_gas_price(start_date, end_date, idx)
wti_prices = extract_data.get_wti_oil_price(start_date, end_date, idx)
total_df_hourly = pd.concat([leipzig_weather, oil_price, gas_price, wti_prices, elec_price], axis=1)
total_df_hourly.to_csv('dataset/hourly_historical_data.csv')



data_extracted_hourly=total_df_hourly
data_extracted_daily= total_df_daily

data_list_daily=list(data_extracted_hourly)
data_list_hourly=list(data_extracted_daily)

data_list_daily=data_list_daily[-1:]+ data_list_daily[6:-1]
data_list_hourly=data_list_hourly[-1:]+ data_list_hourly[6:-1]

#Only give the last month of data of electricity price to model and scale them
# Start with hourly data for univariante
data_prediction_univariante_hourly=data_extracted_hourly[data_list_hourly[0:1]][-24*7*3:]
data_prediction_univariante_hourly=scaler_uni_hourly.fit_transform(data_prediction_univariante_hourly)
data_prediction_univariante_hourly = np.expand_dims(data_prediction_univariante_hourly, axis=0)

#daily data for univariante
data_prediction_univariante_daily=data_extracted_daily[data_list_daily[0:1]][-24*7*3:]
# data_prediction_univariante_daily=scaler_uni_daily.fit_transform(data_prediction_univariante_daily)
# data_prediction_univariante_daily = np.expand_dims(data_prediction_univariante_daily, axis=0)

#Hourly data for multi variante
data_prediction_mutlivariante_hourly=data_extracted_hourly[data_list_hourly[0:5]][-24*7*3:]
# data_prediction_mutlivariante_hourly=scaler_multi_hourly.fit_transform(data_prediction_mutlivariante_hourly)

#Daily data for mutli variante
# data_prediction_mutlivariante_daily=data_extracted_daily[data_list_daily[0:1]][-24*7*3:]
# data_prediction_mutlivariante_daily=scaler_multi_daily.fit_transform(data_prediction_mutlivariante_daily)


# data_list=data_list[-24*7*3:]
# data_list=scaler_multi.fit_transform(data_list)



def one_hour(mutivariate_model=False):
    if (mutivariate_model==False):
        hour_prediction = model_uni_hour.predict(np.array(data_prediction_univariante_hourly[: ,-24:,:]))
        hour_prediction = scaler_uni_hourly.inverse_transform(hour_prediction)
    else:
        hour_prediction = model_multi_hour.predict(np.array(data_prediction_mutlivariante_hourly[:,-24:,:]))
    return hour_prediction

def one_day(mutivariate_model=False,daily=False):
    if (mutivariate_model == False):
        if (daily==False):
            day_prediction = model_uni_day.predict(np.array(data_prediction_univariante_hourly[:,-24*7:,:]))
            day_prediction = scaler_uni_hourly.inverse_transform(day_prediction)
        else:
            day_prediction = model_multi_day.predict(np.array(data_prediction_univariante_daily[:, -24 * 7:, :]))
    else:
        if (daily==False):
            day_prediction = model_uni_day.predict(np.array(data_prediction_mutlivariante_hourly[:,-24*7:, :]))
            day_prediction = scaler_multi_hourly.inverse_transform(day_prediction)
        # else:
        #     day_prediction = model_uni_day.predict(np.array(data_prediction_mutlivariante_daily[:,-24*7:, :]))
        #     day_prediction = scaler_multi.daily.inverse_transform(day_prediction)
    print("Done day prediction")
    return day_prediction

def one_week(mutivariate_model=False):
    if (mutivariate_model == False):
        week_prediction = model_uni_week.predict(np.array(data_prediction_univariante_hourly))
    # else:
    #     week_prediction = model_multi_week.predict(data_prediction_mutlivariante_daily)
        print("Done Week prediction")
    return week_prediction