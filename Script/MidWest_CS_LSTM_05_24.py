import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

# set up the interval need to forecast here
unique_times = [ datetime.time(8, 0),datetime.time(8, 30),datetime.time(9, 0)
                ,datetime.time(9, 30),datetime.time(10, 0),
datetime.time(10, 30), datetime.time(11, 0), datetime.time(11, 30),
datetime.time(12, 0), datetime.time(12, 30), datetime.time(13, 0),
datetime.time(13, 30), datetime.time(14, 0), datetime.time(14, 30),
datetime.time(15, 0), datetime.time(15, 30), datetime.time(16, 0),
datetime.time(16, 30), datetime.time(17, 0), datetime.time(17, 30),datetime.time(18,0),datetime.time(18,30),datetime.time(19,0),datetime.time(19,30),datetime.time(20,0)
]

for month_number in ['2024-11-15']:
#for month_number in ['2024-11-15','2024-12-15','2025-01-15','2025-02-15','2025-03-15']:
#for month_number in ['2024-11-15','2024-12-15','2025-01-15']:
    #ETL Process
    data = pd.read_excel('workspace/MidWest_WFM_Stat_2025_05_16.xlsx')
    pd.set_option('display.max_rows', 100)
    data.columns = ['conversation_start_interval_tmst', 'Time', 'offered', 'actans',
           'actabn', 'absActHt', 'absActSa', 'parent', 'child', 'fiscalDate',
           'fiscalYear', 'ficalQuarter', 'fiscalMonth', 'fiscalWeek', 'DOW']
    
    data = data[~data .offered.isna()]
    
    #data = data[data .offered != 0]
    # Month one and half need to set up 
    data_=data
    data = data[data.conversation_start_interval_tmst<=pd.to_datetime('{} 00:00:00'.format(month_number))]
    holiday_data = pd.read_excel(r'workspace/holiday.xlsx')
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
    
    # 获取所有唯一的时间点（如每天的09:00, 09:30等）
    #unique_times = data['time'].unique()
    
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
    
    matrix = []
    
    # 遍历每个时间点
    for time,time1,time2 in zip(unique_times,np.roll(unique_times, shift=-1),np.roll(unique_times, shift=1)):
        # 过滤出当前时间点的数据
        time_data = data[data['time'] == time].set_index('date')['offered']    
        full_dates = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq='B')  # 仅工作日
        full_dates = full_dates.difference(holiday_datetime)
        time_data = time_data.groupby(time_data.index).sum()
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
        # Step 3: Join on index
        result = df.join(df_last_year, how='left')
        year_data=result['offered_last_year'] 
        #year_data = time_data.shift(251)
        # 将当前时间点、前半小时和后半小时的数据拼接
        combined_data = pd.concat([time_data,add_data_, shifted_data,shifted_data1,shifted_data2,year_data], axis=1)
        
        # 创建日期+时间列
        datetime_column = np.array([pd.Timestamp(date) + pd.Timedelta(hours=time.hour, minutes=time.minute) for date in full_dates]).reshape(-1, 1)
        
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
                           ['years_data']).sort_values('datetime').reset_index(drop = True)
    
    matrix['year'] = matrix.datetime.map(lambda x:x.year)
    matrix['month'] = matrix.datetime.map(lambda x:x.month)
    matrix['day'] = matrix.datetime.map(lambda x:x.day)
    matrix['hour'] = matrix.datetime.map(lambda x:x.hour)
    matrix['minute'] = matrix.datetime.map(lambda x:x.minute)
    
    matrix = matrix.sort_values('datetime').reset_index(drop = True)
    
    read_data = matrix.iloc[:]
    read_data = read_data[~read_data.offered.isna()].reset_index(drop = True)
    
    fill_na = read_data.iloc[:,1:].apply(lambda x:pd.to_numeric(x))
    
    
    
    fill_na = fill_na.interpolate(method='linear')
    
    fill_na['datetime'] = read_data.datetime
    fill_na['date'] = fill_na.datetime.dt.date
    fill_na = fill_na.dropna().reset_index(drop = True)
    
    fill_na = pd.concat([fill_na,pd.get_dummies(fill_na['month'],prefix = 'month').astype(int)],axis = 1)
    fill_na = pd.concat([fill_na,pd.get_dummies(fill_na['year'],prefix = 'year').astype(int)],axis = 1)
    fill_na['week_of_year'] = fill_na['datetime'].dt.isocalendar().week
    fill_na['weekday'] = fill_na['datetime'].dt.day_name()
    fill_na = pd.concat([fill_na,pd.get_dummies(fill_na['week_of_year'],prefix = 'week_of_year').astype(int)],axis = 1)
    fill_na = pd.concat([fill_na,pd.get_dummies(fill_na['weekday'],prefix = 'weekday').astype(int)],axis = 1)
    
    matrix_predict = []
    
    # 遍历每个时间点
    for time,time1,time2 in zip(unique_times,np.roll(unique_times, shift=-1),np.roll(unique_times, shift=1)):
        # 过滤出当前时间点的数据
        time_data = data[data['time'] == time].set_index('date')['offered']    
        full_dates = pd.date_range(start=unique_dates.min(), end=unique_dates.max()+ pd.offsets.BDay(start_time), freq='B')  # 仅工作日
        full_dates = full_dates.difference(holiday_datetime)
        time_data = time_data.groupby(time_data.index).sum()    
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
        # Step 3: Join on index
        result = df.join(df_last_year, how='left')
        year_data=result['offered_last_year']
        #year_data = time_data.shift(251)
    
        # 将当前时间点、前半小时和后半小时的数据拼接
        combined_data = pd.concat([time_data,add_data_, shifted_data,shifted_data1,shifted_data2,year_data], axis=1)
        
        # 创建日期+时间列
        datetime_column = np.array([pd.Timestamp(date) + pd.Timedelta(hours=time.hour, minutes=time.minute) for date in full_dates]).reshape(-1, 1)
        
        # 将日期+时间列添加到数据中
        combined_data_with_datetime = np.hstack([datetime_column,combined_data.to_numpy()])[-start_time:]
        
        
        
        # 将结果存入矩阵
        matrix_predict.append(combined_data_with_datetime)
    
    max_rows = max(arr.shape[0] for arr in matrix_predict)
    # 将每个时间点的数据填充到最大行数
    for i in range(len(matrix_predict)):
        if matrix_predict[i].shape[0] < max_rows:
            padding = np.full((max_rows - matrix_predict[i].shape[0], 11), np.nan)  # 用NaN填充（10列数据 + 1列时间）
            matrix_predict[i] = np.vstack([matrix_predict[i], padding])
    
    # 将矩阵堆叠成一个大的二维数组
    matrix_predict = np.vstack(matrix_predict)
    
    matrix_predict = pd.DataFrame(matrix_predict,columns = ['datetime','offered'] + add_columns+ [f'freq_{i}' for i in range(start_time+1,end_time+1) ]\
                          + [f'freq_last{i}' for i in range(start_time+1,end_time+1) ] + [f'freq_next{i}' for i in range(start_time+1,end_time+1) ]+ \
                           ['years_data']).sort_values('datetime').reset_index(drop = True)
    
    matrix_predict['year'] = matrix_predict.datetime.map(lambda x:x.year)
    matrix_predict['month'] = matrix_predict.datetime.map(lambda x:x.month)
    matrix_predict['day'] = matrix_predict.datetime.map(lambda x:x.day)
    matrix_predict['hour'] = matrix_predict.datetime.map(lambda x:x.hour)
    matrix_predict['minute'] = matrix_predict.datetime.map(lambda x:x.minute)
    
    matrix_predict = matrix_predict.sort_values('datetime').reset_index(drop = True)
    
    read_data_predict = matrix_predict.iloc[:]
    
    
    
    
    fill_na_predict = read_data_predict.iloc[:,2:].apply(lambda x:pd.to_numeric(x))
    
    fill_na_predict = fill_na_predict.interpolate(method='linear')
    fill_na_predict = fill_na_predict.fillna(1)
    
    fill_na_predict['datetime'] = read_data_predict.datetime
    fill_na_predict['date'] = fill_na_predict.datetime.dt.date
    
    fill_na_predict = pd.concat([fill_na_predict,pd.get_dummies(fill_na_predict['month'],prefix = 'month').astype(int)],axis = 1)
    fill_na_predict = pd.concat([fill_na_predict,pd.get_dummies(fill_na_predict['year'], prefix = 'year').astype(int)],axis = 1)
    fill_na_predict['week_of_year'] = fill_na_predict['datetime'].dt.isocalendar().week
    fill_na_predict['weekday'] = fill_na_predict['datetime'].dt.day_name()
    fill_na_predict = pd.concat([fill_na_predict,pd.get_dummies(fill_na_predict['week_of_year'],prefix = 'week_of_year').astype(int)],axis = 1)
    fill_na_predict = pd.concat([fill_na_predict,pd.get_dummies(fill_na_predict['weekday'], prefix = 'weekday').astype(int)],axis = 1)
    
    fill_na_predict
    
    X_columns = [f'freq_{i}' for i in range(start_time+1,end_time+1) ] + [f'freq_last{i}' for i in range(start_time+1,end_time+1) ] +\
                [f'freq_next{i}' for i in range(start_time+1,end_time+1) ] + ['years_data'] + [f'month_{i}' for i in range(1,13)] +\
                [f'year_{i}' for i in np.unique(fill_na.year)] +\
                [f'week_of_year_{i}' for i in fill_na.week_of_year.drop_duplicates().tolist()] +\
                    add_columns
                
    X_columns2 = [f'freq_{i}' for i in range(start_time+1,end_time+1) ] + [f'freq_last{i}' for i in range(start_time+1,end_time+1) ] +\
                [f'freq_next{i}' for i in range(start_time+1,end_time+1) ] + ['years_data'] + [f'month_{i}' for i in range(1,13)] +\
                [f'year_{i}' for i in np.unique(fill_na.year)] +\
                [f'week_of_year_{i}' for i in fill_na.week_of_year.drop_duplicates().tolist()] +\
                    add_columns
    
    
    lstm_data = fill_na.dropna(axis = 0)
    
    lstm_data
    
    for i in X_columns:
        try:
            fill_na_predict[i]
        except:
            fill_na_predict[i] = 0
    
    lstm_data = lstm_data[lstm_data.year>=2021].reset_index(drop = True)
    
    week_dict = {'Friday':5, 'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4}
    
    lstm_data['week_num'] = lstm_data['weekday'].map(lambda x:week_dict[x])
    fill_na_predict['week_num'] = fill_na_predict['weekday'].map(lambda x:week_dict[x])
    all_feature = pd.DataFrame()
    all_predict=pd.DataFrame()
    #for week_ in [1,2,3,4,5]:
    for (hour,minute) in [(i.hour,i.minute) for i in unique_times]:
        now_lstm_data = lstm_data[(lstm_data.hour == hour)&(lstm_data.minute == minute)]
        #now_lstm_data = now_lstm_data[now_lstm_data.week_num == week_]
        now_fill_na_predict = fill_na_predict[(fill_na_predict.hour == hour)&(fill_na_predict.minute == minute)]
        #now_fill_na_predict = now_fill_na_predict[now_fill_na_predict.week_num == week_]
    
        now_time_list = now_lstm_data.datetime.to_numpy()
        now_lstm_data = now_lstm_data[['offered']+['datetime'] +X_columns].reset_index(drop = True)
        # 归一化数据
    
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(now_lstm_data[['offered']][:-start_time])
        # 设置时间步长
        time_step = start_time
        # 创建训练数据集
        X_train_1, y_train =now_lstm_data[X_columns][:-start_time] , scaled_data
    
        X_test_1, y_test =now_lstm_data[X_columns][-start_time:] , now_lstm_data[['offered']][-start_time:]
    
        X_train_1 = np.array(X_train_1)
        y_train = np.array(y_train)
        X_test_1 = np.array(X_test_1)
        y_test = np.array(y_test)
        X_test_1 = X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[1], 1)
    
        # 重塑输入数据为 [samples, time steps, features] 格式
        X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 1)
        # 构建LSTM模型
        model_1 = Sequential()
        model_1.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model_1.add(LSTM(50, return_sequences=False))
        model_1.add(Dense(25))
        model_1.add(Dense(1))
    
        # 编译模型
        model_1.compile(optimizer='adam', loss='mean_squared_error')
        model_1.fit(X_train_1, y_train, batch_size=32, epochs=20)
        train_predict_1 = model_1.predict(X_train_1)
        train_predict_1 = scaler.inverse_transform(train_predict_1)
        True_data_1 = scaler.inverse_transform(y_train.reshape(-1, 1))
        print(calculate_mape(train_predict_1, True_data_1))
        return_y_1 =  model_1.predict(X_test_1)
        return_y_pred_1 = scaler.inverse_transform(np.array(return_y_1).reshape(-1, 1))
        now_predict_1 = pd.DataFrame(return_y_pred_1,columns = ['predict'])
        now_predict_1['datetime'] = now_time_list[-start_time:]
        now_lstm_data = now_lstm_data[['offered'] +X_columns2].reset_index(drop = True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(now_lstm_data[['offered']][:-start_time])
        # 设置时间步长
        time_step = start_time
        # 创建训练数据集
    
        X_train_2, y_train =now_lstm_data[X_columns2][:-start_time] , scaled_data
    
        X_test_2, y_test =now_lstm_data[X_columns2][-start_time:] , now_lstm_data[['offered']][-start_time:]
    
        X_train_2 = np.array(X_train_2)
        y_train = np.array(y_train)
        X_test_2 = np.array(X_test_2)
        y_test = np.array(y_test)
        X_test_2 = X_test_2.reshape(X_test_2.shape[0], X_test_2.shape[1], 1)
    
        # 重塑输入数据为 [samples, time steps, features] 格式
        X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1], 1)
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
        train_predict_2 = scaler.inverse_transform(train_predict_2)
        True_data_2 = scaler.inverse_transform(y_train.reshape(-1, 1))
        print(calculate_mape(train_predict_2, True_data_2))
        return_y_2 =  model_2.predict(X_test_2)
        return_y_pred_2 = scaler.inverse_transform(np.array(return_y_2).reshape(-1, 1))
        now_predict_2 = pd.DataFrame(return_y_pred_2,columns = ['predict'])
        now_predict_2['datetime'] = now_time_list[-start_time:]
        print(calculate_mape(y_test, return_y_pred_2))
        if calculate_mape(y_test, return_y_pred_1) < calculate_mape(y_test, return_y_pred_2):
            all_predict = pd.concat([now_predict_1,all_predict])
            future_X = now_fill_na_predict[X_columns].to_numpy()
            future_data = scaler.inverse_transform(np.array(\
                            model_1.predict(future_X.reshape(future_X.shape[0], future_X.shape[1], 1))).reshape(-1, 1))
            future_data = pd.DataFrame(future_data,columns = ['predict'])
            all_feature = pd.concat([all_feature,future_data])
        else:
            all_predict = pd.concat([now_predict_2,all_predict])
            future_X = now_fill_na_predict[X_columns2].to_numpy()
            future_data = scaler.inverse_transform(np.array(\
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
    all_predict.to_excel(r'workspace/MidWest_CS_Test_05_17_{}.xlsx'.format(month_number))
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
    all_feature.to_excel(r'workspace/MidWest_CS_Target_05_17_{}.xlsx'.format(month_number))