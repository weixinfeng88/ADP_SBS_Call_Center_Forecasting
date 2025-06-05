import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM
from keras.layers import  Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import datetime
from datetime import time
import numpy as np

def calculate_mape(actual, predicted):
    """
    计算 MAPE（平均绝对百分比误差）

    参数:
    
    
    actual (array-like): 实际值数组。
    predicted (array-like): 预测值数组。

    返回:
    float: MAPE 值。
    """
    # 将输入转换为 numpy 数组
    actual = np.array(actual)
    predicted = np.array(predicted)

    # 避免除以零的情况
    if np.any(actual == 0):
        raise ValueError("实际值中包含零，无法计算 MAPE。")
    # 计算 MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape
def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    if not np.any(mask):
        return np.nan  # Or return 0 or raise a different warning

    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return mape

def ETL(month_list,data,holiday_data,full_dates,unique_times):
    '''
    month_list is the list you want to forecast one and half month in advanced, for example
    , if you want to forecast May, 2025, please input ['2025-03-15'], also you can include multiple value in the list
    , like ['2024-11-15','2024-12-15','2025-01-15','2025-02-15','2025-03-15'];
    data is the source data in dataframe;
    holiday_data is the holiday data in dataframe
    full_dates are all the Business days you want to feed into Model for training;
    unique_times are the intervals need to forecast for every day;
    '''
    matrix = []

    for time,time1,time2 in zip(unique_times,np.roll(unique_times, shift=-1),np.roll(unique_times, shift=1)):
        # 过滤出当前时间点的数据
        time_data = data[data['time'] == time].set_index('date')['offered']    
        # full_dates #= pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq='B')  # 仅工作日
        full_dates = full_dates.difference(holiday_datetime)
        #time_data = time_data.groupby(time_data.index).sum()
        time_data = time_data.reindex(full_dates)  # 重新索引，缺失数据用NaN填充    
        shifted_data = pd.concat([time_data.shift(i) for i in range(start_time,end_time)], axis=1)
        
        
        add_data = data[data['time'] == time].set_index('date')[add_column]
        add_data = add_data.groupby(add_data.index).sum()
        add_data = add_data.reindex(full_dates)  # 重新索引，缺失数据用NaN填充    
        add_data_ = pd.concat([add_data.shift(i) for i in range(start_time,end_time)], axis=1)
        
        
        time_data1 = data[data['time'] == time1].set_index('date')['offered']    
        time_data1 = time_data1.groupby(time_data1.index).sum()
        time_data1 = time_data1.reindex(full_dates)  # 重新索引，缺失数据用NaN填充    
        shifted_data1 = pd.concat([time_data1.shift(i) for i in range(start_time,end_time)], axis=1)
        
        time_data2 = data[data['time'] == time2].set_index('date')['offered']    
        time_data2 = time_data2.groupby(time_data2.index).sum()
        time_data2 = time_data2.reindex(full_dates)  # 重新索引，缺失数据用NaN填充    
        shifted_data2 = pd.concat([time_data2.shift(i) for i in range(start_time,end_time)], axis=1)
        df=time_data.to_frame(name='offer')
        df_last_year = df.copy()
        df_last_year.index = df_last_year.index + pd.DateOffset(years=1)
        df_last_year = df_last_year.rename(columns={'offer': 'offered_last_year'})
        df_last_two_year = df.copy()
        df_last_two_year.index = df_last_two_year.index + pd.DateOffset(years=2)
        df_last_two_year = df_last_two_year.rename(columns={'offer': 'offered_last_two_year'})        
        # Step 3: Join on index
        result = df.join(df_last_year, how='left').join(df_last_two_year,how='left')
        year_data=result[['offered_last_year','offered_last_two_year']]
        #year_data = time_data.shift(251)
        # 将当前时间点、前半小时和后半小时的数据拼接
        combined_data = pd.concat([time_data,add_data_, shifted_data,shifted_data1,shifted_data2,year_data], axis=1)
        combined_data = combined_data.merge(date_column, left_index=True, right_index=True, how='left')

        # 创建日期+时间列
        #datetime_column = np.array([pd.Timestamp(date) + pd.Timedelta(hours=time.hour, minutes=time.minute) for date in full_dates]).reshape(-1, 1)
        datetime_column = np.array([pd.Timestamp(date) + pd.Timedelta(hours=time.hour, minutes=time.minute) for date in combined_data.index]).reshape(-1, 1)
        # 将日期+时间列添加到数据中
        combined_data_with_datetime = np.hstack([datetime_column,combined_data.to_numpy()])
        
        # 将结果存入矩阵
        matrix.append(combined_data_with_datetime)
    
    max_rows = max(arr.shape[0] for arr in matrix)
    
    # 将每个时间点的数据填充到最大行数
    for i in range(len(matrix)):
        if matrix[i].shape[0] < max_rows:
            padding = np.full((max_rows - matrix[i].shape[0], 11), np.nan)  # 用NaN填充（10列数据 + 1列时间）
            matrix[i] = np.vstack([matrix[i], padding])
    
    # 将矩阵堆叠成一个大的二维数组
    matrix = np.vstack(matrix)
    
    matrix = pd.DataFrame(matrix,columns = ['datetime','offered'] +add_columns +  [f'freq_{i}' for i in range(start_time+1,end_time+1) ]\
                          + [f'freq_last{i}' for i in range(start_time+1,end_time+1) ] + \
                          [f'freq_next{i}' for i in range(start_time+1,end_time+1) ]+ \
                           ['years_data_1','years_data_2','fiscalYear', 'ficalQuarter', 'fiscalMonth', 'fiscalWeek', 'DOW']).sort_values('datetime').reset_index(drop = True)
    
    matrix['hour'] = matrix.datetime.map(lambda x:x.hour)
    matrix['minute'] = matrix.datetime.map(lambda x:x.minute)
    matrix = matrix.sort_values('datetime').reset_index(drop = True)
    
    read_data = matrix.iloc[:]
    
    read_data = read_data[~read_data.offered.isna()].reset_index(drop = True)
    # Quarter: 'Q1' → 1, ..., 'Q4' → 4
    quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    read_data['ficalQuarter'] = read_data['ficalQuarter'].map(quarter_map)
    year_map={'FY 2021':1 ,'FY 2022':2,'FY 2023':3,'FY 2024':4, 'FY 2025':5}
    read_data['fiscalYear']=read_data['fiscalYear'].map(year_map)
    # Month: 'January' → 1, ..., 'December' → 12
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    read_data['fiscalMonth'] = read_data['fiscalMonth'].map(month_map)
    
    # Day of week: 'Monday' → 0, ..., 'Sunday' → 6
    dow_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    read_data['DOW'] = read_data['DOW'].map(dow_map)
    read_data['fiscalWeek']=read_data['fiscalWeek'].apply(lambda x:pd.to_numeric(x))
    # Step 2: Apply cyclical encoding
    
    read_data['ficalQuarter_sin'] = np.sin(2 * np.pi * read_data['ficalQuarter'] / 4)
    read_data['ficalQuarter_cos'] = np.cos(2 * np.pi * read_data['ficalQuarter'] / 4)
    
    read_data['fiscalMonth_sin'] = np.sin(2 * np.pi * read_data['fiscalMonth'] / 12)
    read_data['fiscalMonth_cos'] = np.cos(2 * np.pi * read_data['fiscalMonth'] / 12)
    
    read_data['fiscalWeek_sin'] = np.sin(2 * np.pi * read_data['fiscalWeek'] / 52)
    read_data['fiscalWeek_cos'] = np.cos(2 * np.pi * read_data['fiscalWeek'] / 52)
    
    read_data['DOW_sin'] = np.sin(2 * np.pi * read_data['DOW'] / 7)
    read_data['DOW_cos'] = np.cos(2 * np.pi * read_data['DOW'] / 7)

    
    
    # Step 3: Drop original raw categorical time features
    
    read_data.drop(['ficalQuarter', 'fiscalMonth', 'fiscalWeek', 'DOW'], axis=1, inplace=True)
    fill_na = read_data.iloc[:,1:].apply(lambda x:pd.to_numeric(x))

    fill_na = fill_na.interpolate(method='linear')
    
    fill_na['datetime'] = read_data.datetime
    fill_na['date'] = fill_na.datetime.dt.date
    fill_na = fill_na.dropna().reset_index(drop = True)
    return fill_na

def Train_Model(train_dataset,test_dataset,unique_times):
    all_feature = pd.DataFrame()
    all_predict=pd.DataFrame()
    for (hour,minute) in [(i.hour,i.minute) for i in unique_times]:
        now_lstm_data = train_dataset[(train_dataset.hour == hour)&(train_dataset.minute == minute)]
        now_fill_na_predict = test_dataset[(test_dataset.hour == hour)&(test_dataset.minute == minute)]    
        now_time_list = now_lstm_data.datetime.to_numpy()
        cols=  ['offered', 'datetime'] + [col for col in now_lstm_data.columns if col not in ['offered', 'datetime','date']]
        cols_=[col for col in now_lstm_data.columns if col not in ['offered', 'datetime','date']]
        date_cols=['fiscalYear','ficalQuarter_sin','ficalQuarter_cos'
              ,'fiscalMonth_sin','fiscalMonth_cos','fiscalWeek_sin','DOW_cos','DOW_sin','hour','minute','datetime','date','offered']
        cal_cols=['fiscalYear','ficalQuarter_sin','ficalQuarter_cos'
              ,'fiscalMonth_sin','fiscalMonth_cos','fiscalWeek_sin','DOW_cos','DOW_sin']
        cols_offered=[col for col in now_lstm_data.columns if col not in date_cols]

        now_lstm_data = now_lstm_data.reset_index(drop = True)
        # 归一化数据
        
        # Separate the parts
        to_scale_df = now_lstm_data[cols_offered]
        keep_df = now_lstm_data[cal_cols]
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_y.fit_transform(now_lstm_data[['offered']])
        scaled_array = scaler_x.fit_transform(to_scale_df)
        scaled_df = pd.DataFrame(scaled_array, columns=cols_offered)
        # Concatenate back together
        X_combined = pd.concat([keep_df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
        # 设置时间步长
        time_step = start_time
        # 创建训练数据集
        X_train_1, y_train =X_combined [:-start_time] , scaled_data[:-start_time]
        X_test_1, y_test =X_combined[-start_time:]  , scaled_data[-start_time:]
        X_train_1 = np.array(X_train_1)
        y_train = np.array(y_train)
        X_test_1 = np.array(X_test_1)
        y_test = np.array(y_test)
        X_test_1 = X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[1], 1)
    
        # 重塑输入数据为 [samples, time steps, features] 格式
        X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 1)
        X_train_1 = X_train_1.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test_1 = X_test_1.astype(np.float32)
        y_test = y_test.astype(np.float32)
        # 构建LSTM模型
        model_1 = Sequential()
        model_1.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model_1.add(Dense(25))
        model_1.add(LSTM(20, return_sequences=False))
        model_1.add(Dense(1))
    
        # 编译模型
        model_1.compile(optimizer='adam', loss='mean_squared_error')
        model_1.fit(X_train_1, y_train, batch_size=32, epochs=20)
        train_predict_1 = model_1.predict(X_train_1)
        train_predict_1 = scaler_y.inverse_transform(train_predict_1)
        True_data_1 = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        print(calculate_mape(train_predict_1, True_data_1))
        return_y_1 =  model_1.predict(X_test_1)
        return_y_pred_1 = scaler_y.inverse_transform(np.array(return_y_1).reshape(-1, 1))
        now_predict_1 = pd.DataFrame(return_y_pred_1,columns = ['predict'])
        now_predict_1['datetime'] = now_time_list[-start_time:]
        now_lstm_data = now_lstm_data[['offered'] +cols_].reset_index(drop = True)        
        # 设置时间步长
        time_step = start_time
        # 创建训练数据集
        X_train_2 = X_combined[:-start_time]
        X_test_2 = X_combined[-start_time:]
        y_train = scaled_data[:-start_time]
        y_test = scaled_data[-start_time:]
        
        X_train_2 = np.array(X_train_2)
        y_train = np.array(y_train)
        X_test_2 = np.array(X_test_2)
        y_test = np.array(y_test)
        X_test_2 = X_test_2.reshape(X_test_2.shape[0], X_test_2.shape[1], 1)
    
        # 重塑输入数据为 [samples, time steps, features] 格式
        X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1], 1)
        X_train_2 = X_train_2.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test_2 = X_test_2.astype(np.float32)
        y_test = y_test.astype(np.float32)
        # 构建LSTM模型
        model_2 = Sequential()
        model_2.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model_2.add(LSTM(50, return_sequences=False))
        model_2.add(Dense(25))
        model_2.add(Dense(1))
    
        # 编译模型
        model_2.compile(optimizer='adam', loss='mean_squared_error')
        model_2.fit(X_train_2, y_train, batch_size=32, epochs=20)
        train_predict_2 = model_2.predict(X_train_2)
        train_predict_2 = scaler_y.inverse_transform(train_predict_2)
        True_data_2 = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        print(calculate_mape(train_predict_2, True_data_2))
        return_y_2 =  model_2.predict(X_test_2)
        return_y_pred_2 = scaler_y.inverse_transform(np.array(return_y_2).reshape(-1, 1))
        now_predict_2 = pd.DataFrame(return_y_pred_2,columns = ['predict'])
        now_predict_2['datetime'] = now_time_list[-start_time:]
        print(calculate_mape(y_test, return_y_pred_2))
        if calculate_mape(y_test, return_y_pred_1) < calculate_mape(y_test, return_y_pred_2):
            all_predict = pd.concat([now_predict_1,all_predict])
            future_X = now_fill_na_predict[cols_].to_numpy()
            future_data = scaler_y.inverse_transform(np.array(\
                            model_1.predict(future_X.reshape(future_X.shape[0], future_X.shape[1], 1))).reshape(-1, 1))
            future_data = pd.DataFrame(future_data,columns = ['predict'])
            all_feature = pd.concat([all_feature,future_data])
        else:
            all_predict = pd.concat([now_predict_2,all_predict])
            future_X = now_fill_na_predict[cols_].to_numpy()
            future_data = scaler_y.inverse_transform(np.array(\
                            model_2.predict(future_X.reshape(future_X.shape[0], future_X.shape[1], 1))).reshape(-1, 1))
            future_data = pd.DataFrame(future_data,columns = ['predict'])
            all_feature = pd.concat([all_feature,future_data])
        #print('Interval Hour:{}, Minute:{} has been trained'.format(hour,minute)
    all_predict = pd.merge(all_predict,lstm_data[['offered','datetime']],on = ['datetime'])
    all_predict = all_predict.sort_values('datetime').reset_index(drop = True)
    #all_feature.to_excel(r'return_month_{}.xlsx'.format(month_number))
    all_predict['diff'] = (all_predict.predict - all_predict.offered)
    all_predict['hour'] = all_predict.datetime.dt.hour
    pd.merge(all_predict,lstm_data[['offered','datetime']],on = ['datetime'])
    calculate_mape(all_predict.offered,all_predict.predict)
    
    time_ = []
    for (hour,minute) in fill_na_predict[['hour','minute']].drop_duplicates().to_numpy()[:]:
        now_fill_na_predict = fill_na_predict[(fill_na_predict.hour == hour)&(fill_na_predict.minute == minute)]
        time_.append(now_fill_na_predict.datetime.to_numpy())
    all_feature['datetime'] =  np.hstack(time_)
    all_feature.predict = all_feature.predict.map(lambda x:max(0,x))
    data_['conversation_start_interval_tmst']=pd.to_datetime(data_['conversation_start_interval_tmst'])
    all_feature['datetime']=pd.to_datetime(all_feature['datetime'])
    all_feature = pd.merge(all_feature,data_[['offered','conversation_start_interval_tmst']],left_on = ['datetime'],right_on=['conversation_start_interval_tmst'])
    all_feature = all_feature.sort_values('datetime').reset_index(drop = True)
    all_feature['mape']=np.abs(all_feature['offered']-all_feature['predict'])/(all_feature['offered'])*100
    return all_predict, all_feature

month_number=['2025-01-15']
#data input
data = pd.read_excel('/workspace/ADP_SBS_Call_Center_Forecasting/Data/MidWest_WFM_Stat_2025_05_16.xlsx')
data.columns = ['conversation_start_interval_tmst', 'Time', 'offered', 'actans',
               'actabn', 'absActHt', 'absActSa', 'parent', 'child', 'fiscalDate',
               'fiscalYear', 'ficalQuarter', 'fiscalMonth', 'fiscalWeek', 'DOW']
data = data[~data .offered.isna()]
data_=data.copy()
data = data[data.conversation_start_interval_tmst<=pd.to_datetime('{} 00:00:00'.format(month_number))]
holiday_data = pd.read_excel(r'/workspace/ADP_SBS_Call_Center_Forecasting/Data/holiday.xlsx')
#month_list=['2025-02-15']
unique_dates = data['conversation_start_interval_tmst'].dt.date.unique()
start_time = 35 + 23
end_time = start_time+36 #36 is the train length
full_dates_train = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq='B')  # 仅工作日
full_dates_test = pd.date_range(start=unique_dates.min(), end=unique_dates.max()+ pd.offsets.BDay(start_time), freq='B')  # 仅工作日
unique_times_Midwest = [ datetime.time(10, 0),
datetime.time(10, 30), datetime.time(11, 0), datetime.time(11, 30),
datetime.time(12, 0), datetime.time(12, 30), datetime.time(13, 0),
datetime.time(13, 30), datetime.time(14, 0), datetime.time(14, 30),
datetime.time(15, 0), datetime.time(15, 30), datetime.time(16, 0),
datetime.time(16, 30), datetime.time(17, 0), datetime.time(17, 30),datetime.time(18,0),datetime.time(18,30),datetime.time(19,0),datetime.time(19,30),datetime.time(20,0),datetime.time(20,30)
]
data.columns = ['conversation_start_interval_tmst', 'Time', 'offered', 'actans',
       'actabn', 'absActHt', 'absActSa', 'parent', 'child', 'fiscalDate',
       'fiscalYear', 'ficalQuarter', 'fiscalMonth', 'fiscalWeek', 'DOW']

data = data[~data .offered.isna()]
# Month one and half need to set up 
data_=data.copy()
data = data[data.conversation_start_interval_tmst<=pd.to_datetime('{} 00:00:00'.format(month_number))]
holiday_data = pd.read_excel(r'workspace/ADP_SBS_Call_Center_Forecasting/Data/holiday.xlsx')
holiday_data['~is_holiday'] = 0
Holiday_name = ['Christmas Day', 'Columbus Day',
       'Independence Day', 'Labor Day', 'Martin Luther King Jr. Day',
       'Memorial Day', "New Year's Day", "Presidents' Day", 'Thanksgiving Day',
       'Veterans Day']
holiday_data['date'] = holiday_data.Date.dt.date
data['datetime'] = pd.to_datetime(data.conversation_start_interval_tmst)
data['date'] = data.datetime.dt.date
data.datetime = data.datetime.dt.floor('T')
data['datetime'] = data['datetime'].apply(
    lambda x: x.ceil('30T') if x.minute == 29 else x
)
data['datetime'] = data['datetime'].apply(
    lambda x: x.ceil('H') if x.minute == 59 else x
)
data['time'] = data.datetime.dt.time
data = data.sort_values('datetime').reset_index(drop = True)
data = data.sort_values(by='datetime')
date_column=data[['date',
   'fiscalYear', 'ficalQuarter', 'fiscalMonth', 'fiscalWeek', 'DOW']]
date_column=date_column.set_index('date')
date_column = date_column[~date_column.index.duplicated(keep='first')]
date_column=date_column.sort_values('date')
# 获取所有唯一的日期
unique_dates = data['date'].unique()
data = pd.merge(data,holiday_data,on = 'date',how = 'left').fillna(1)
data = data[data['~is_holiday'] == 1]

holiday_datetime = holiday_data.date.to_numpy()

start_time = 35+23 # 58 how many days in advance
end_time = start_time+36 #36 is the train length


add_column = ['actans','actabn','absActHt','absActSa']

add_columns =  [f'actans_{i}' for i in range(start_time+1,end_time+1) ] +\
                [f'actabn_{i}' for i in range(start_time+1,end_time+1) ] +\
                [f'absActHt_{i}' for i in range(start_time+1,end_time+1) ] +\
                [f'absActSa_{i}' for i in range(start_time+1,end_time+1) ]


# 遍历每个时间点
fill_na=ETL(month_list,data,holiday_data,full_dates_train,unique_times_Midwest)
fill_na_predict=ETL(month_list,data,holiday_data,full_dates_test,unique_times_Midwest)
lstm_data = fill_na.dropna(axis = 0) 
lstm_data
    
for i in lstm_data.columns:
    try:
        fill_na_predict[i]
    except:
        fill_na_predict[i] = 0
all_predict,all_feature=Train_Model(lstm_data,fill_na_predict,unique_times_Midwest)
month_number=month_list[-1]
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
all_predict.to_excel(f'workspace/Midwest_CS_Test_Improved_{month_number}_{timestamp}.xlsx')
all_feature.to_excel(f'workspace/Midwest_CS_Target_Improved_{month_number}_{timestamp}.xlsx')