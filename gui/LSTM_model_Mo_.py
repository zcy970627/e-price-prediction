# API data extraction
import tensorflow as tf
import keras.models
import requests
import numpy as np
from wwo_hist import retrieve_hist_data
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import seaborn as sns
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# API electricity data
def get_elecprice(start_date,end_date):
    r = requests.get('https://coop.eikon.tum.de/mbt/mbt.json')
    response = r.json()
    url = 'https://api.montelnews.com/spot/getprices'
    headers = {
'Authorization': 'Bearer ' + r.json()['access_token']}
    params = {'SpotKey': 14, 'Fields': ['hours'], 'FromDate': start_date, 'ToDate': end_date, 'Currency': 'eur',
              'SortType': 'ascending'}
    r = requests.get(url, headers=headers, params=params)
    r_dictionary = r.json()
    df = pd.DataFrame(r_dictionary['Elements'])
    # Create DataFrame
    # elec_price: data frame containing the price of elactricity each hour
    data = {'start_datetime': [], 'elec_price': []}
    elec_price = pd.DataFrame(data)
    # Add missing hour
    for index, row in df.iterrows():
        current_day = row['Date']
        current_timespan = row['TimeSpans']
        for i in current_timespan:
            current_time = i['TimeSpan'][0:5]
            Time1 = current_day[0:10] + ' ' + current_time + ':00'
            try:
                start_datetime = datetime.strptime(Time1, '%Y-%m-%d %H:%M:%S')
            except:
                hours_added = dt.timedelta(hours=1)
                start_datetime = start_datetime + hours_added
            current_price = i['Value']
            elec_price.loc[len(elec_price.index)] = [start_datetime, current_price]
    elec_price['start_datetime'] = pd.to_datetime(elec_price['start_datetime'], infer_datetime_format='%Y-%m-%d %H:%M:%S')
    # add missing dates
    idx = pd.date_range(start_date, str(datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=1)), freq='1H')
    elec_price = elec_price.set_index('start_datetime')
    elec_price=elec_price.loc[~elec_price.index.duplicated(), :]
    elec_price = elec_price.reindex(idx, fill_value=np.nan)
    elec_price['elec_price'] = elec_price['elec_price'].interpolate(method='polynomial', order=2,limit_direction='both').ffill().bfill()
    elec_price = elec_price.drop(elec_price.index[len(elec_price) - 1])
    return elec_price

#extract oil prices
#you have a problem of interpolation
def get_oil_price(start_date,end_date):
    brent_oil_prices = pd.read_excel('https://www.eia.gov/dnav/pet/hist_xls/RBRTED.xls', sheet_name='Data 1',skiprows=20, names=['Date', 'Brent_Prices'])
    brent_oil_prices = brent_oil_prices[(brent_oil_prices['Date'] >= start_date)]
    brent_oil_prices = brent_oil_prices[(brent_oil_prices['Date'] <= end_date)]
    #add missing dates
    idx = pd.date_range(start_date, str( datetime. strptime(end_date, '%Y-%m-%d')+dt.timedelta(days=1)), freq='1H')
    brent_oil_prices = brent_oil_prices.set_index('Date')
    brent_oil_prices = brent_oil_prices.reindex(idx, fill_value=np.nan)
    brent_oil_prices['Brent_Prices']=brent_oil_prices['Brent_Prices'].interpolate(method='polynomial', order=2, limit_direction= 'both').ffill().bfill()
    brent_oil_prices = brent_oil_prices.drop(brent_oil_prices.index[len(brent_oil_prices) - 1])
    return brent_oil_prices

def get_gas_price(start_date,end_date):
    gas_prices = pd.read_excel('https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls', sheet_name='Data 1', skiprows=2,names=['Date', 'Henry_Hub_Price'])
    gas_prices = gas_prices[(gas_prices['Date'] >= start_date)]
    gas_prices = gas_prices[(gas_prices['Date'] <= end_date)]
    #add missing dates
    idx = pd.date_range(start_date, str( datetime. strptime(end_date, '%Y-%m-%d')+dt.timedelta(days=1)), freq='1H')
    gas_prices = gas_prices.set_index('Date')
    gas_prices = gas_prices.reindex(idx, fill_value=np.nan)
    gas_prices['Henry_Hub_Price']=gas_prices['Henry_Hub_Price'].interpolate(method='polynomial', order=2, limit_direction= 'both').ffill().bfill()
    gas_prices=gas_prices.drop(gas_prices.index[len(gas_prices)-1])
    return gas_prices

def get_wti_oil_price(start_date,end_date):
    wti_prices =  pd.read_excel('https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls', sheet_name='Data 1', skiprows=2801, names=['Date', 'WTI_Prices'])
    wti_prices = wti_prices[(wti_prices['Date'] >= start_date)]
    wti_prices = wti_prices[(wti_prices['Date'] <= end_date)]
    #add missing dates
    idx = pd.date_range(start_date, str( datetime. strptime(end_date, '%Y-%m-%d')+dt.timedelta(days=1)), freq='1H')
    wti_prices = wti_prices.set_index('Date')
    wti_prices = wti_prices.reindex(idx, fill_value=np.nan)
    wti_prices['WTI_Prices']=wti_prices['WTI_Prices'].interpolate(method='polynomial', order=2, limit_direction= 'both').ffill().bfill()
    wti_prices=wti_prices.drop(wti_prices.index[len(wti_prices)-1])
    return wti_prices

#extract weather infos
def get_weather(start_date,end_date):
    api_key = '468e02981d20483d88a181015212508'  # YOUR API KEY
    location_list = ['leipzig']
    frequency = 1
    hist_weather_data = retrieve_hist_data(api_key,
                                          location_list,
                                          start_date,
                                          end_date,
                                          frequency,
                                          location_label=False,
                                          export_csv=True,
                                          store_df=True)

    leipzig_weather = hist_weather_data.drop(
        ['totalSnow_cm', 'moon_illumination', 'moonrise', 'moonset', 'sunrise', 'sunset', 'DewPointC', 'FeelsLikeC',
          'WindChillC', 'visibility', 'winddirDegree', 'location','pressure','precipMM','cloudcover','sunHour'], axis=1)
    leipzig_weather['date_time'] = pd.to_datetime(leipzig_weather['date_time'],     infer_datetime_format='%Y-%m-%d %H:%M:%S')
    leipzig_weather = leipzig_weather.set_index('date_time')
    return leipzig_weather

# Define a function to plot different types of time-series
def plot_series(df=None, column=None, series=pd.Series([]),label=None, ylabel=None, title=None, start=0, end=None):
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=16)
    if column:
        ax.plot(df[column][start:end], label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if series.any():
        ax.plot(series, label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if label:
        ax.legend(fontsize=16)
    if title:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax

def gen_timeseries_dataset(df, input_length, shift = 1, sequence_stride=1, start_index = 0):
    data = np.array(df)
    input = []
    label = []
    i = start_index
    while i <= data.shape[0] - input_length - shift:
        input.append(data[i:i+input_length, :])
        label.append(data[i+input_length+shift-1, 0])
        i += sequence_stride

    return np.array(input), np.array(label)



#second model : this program uses an artificial reccurent neural network called long short term memory (LSTM)
#to predict the electricity price using the past one year prices
def LSTM_model(df, input_length, shift = 1, train_and_val = 0.9, test = 0.1, val = 0.2):
    n = len(df)
    # change the indexes to ints (0,1,2,3)
    df['idx'] = list(range(n))
    df.set_index('idx', inplace=True, drop=True)
    scaler = MinMaxScaler(copy=False)
    scaler_elec = MinMaxScaler(copy=False)
    scaler_elec.fit(np.array(df['elec_price']).reshape(-1, 1))
    df = scaler.fit_transform(df)
    # Split data into train, test and val sets
    train_and_val_df = df[0:int(n * train_and_val)]
    test_df = df[int(n * train_and_val):]

    #train_df = scaler.fit_transform(train_df)
    # Generate timeseries dataset
    gen_dataset_input, gen_dataset_label = gen_timeseries_dataset(train_and_val_df, input_length, shift = shift)
    gen_dataset_label.resize(gen_dataset_label.shape[0],1)

    # Create the model
    model = keras.Sequential()
    model.add(LSTM(128, activation = 'sigmoid',input_shape=(gen_dataset_input.shape[1], gen_dataset_input.shape[2])))
    model.add(Dense(gen_dataset_label.shape[1],  activation = 'sigmoid'))
    # Compile and fit it
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(gen_dataset_input, gen_dataset_label, epochs=10,  batch_size = 128, validation_split = val,
                        callbacks=[early_stopping], verbose = 1)
    #plt.plot(history.history['loss'], label='Training loss')
    #plt.plot(history.history['val_loss'], label='Validation loss')
    #plt.legend()
    #plt.show()
    test_inputs, test_labels = gen_timeseries_dataset(test_df, input_length)
    test_labels.resize((test_labels.shape[0], 1), refcheck= False)
    evaluation = model.evaluate(test_inputs, test_labels, verbose = 1)
    plt.plot(test_labels, label='labels')
    plt.plot(model.predict(test_inputs), label='prediction')
    plt.legend()
    plt.show()
    print("Mean squared error: %f", mean_squared_error(scaler_elec.inverse_transform(test_labels), scaler_elec.inverse_transform(model.predict(test_inputs))))
    return model


if __name__ == "__main__":
    start_date= "2016-0-01"
    end_date= '2021-08-30'
    idx = pd.date_range(start=start_date, end='2021-08-30', freq='1H')
    leipzig_weather = get_weather(start_date, end_date)
    oil_price=get_oil_price(start_date,end_date)
    gas_price = get_gas_price(start_date, end_date)
    wti_prices=get_wti_oil_price(start_date,end_date)
    elec_price= get_elecprice(start_date,end_date)

    #check corrolation
    total_df=pd.concat([leipzig_weather, oil_price,gas_price,wti_prices,elec_price], axis=1)
    total_df= total_df[['elec_price','tempC','windspeedKmph','Brent_Prices','Henry_Hub_Price','WTI_Prices']]

    model=LSTM_model(total_df, 24*7*3, 24*7)
    model.save('LSTM_model.h5')
    # plot electricity price
    #ax = plot_series(df=elec_price, column='price', ylabel='electricity price(eur/mwh)', title='Electricity price')
    #plt.show()
