import os
import csv
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Data Acquisition and Reading
def data_acquisition():
    print("{:=^100s}".format(" Acquiring and reading data... "))
    dataframes = [pd.read_csv(f"./DS_Dataset/{file}") for file in os.listdir("./DS_Dataset")]
    source_data = pd.concat(dataframes, ignore_index=True)
    x_test_data = pd.read_csv(f"./upload.csv")
    print("done")
    return source_data, x_test_data

# 2. Data Preprocessing
def data_preprocessing(data, x_test):
    print("{:=^100s}".format(" Preprocessing data... "))

    # Drop rows with missing values
    data = data.dropna()

    # Min-Max Scaler
    scaler = MinMaxScaler()
    columns_to_scale = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    # Convert DateTime column to datetime format
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
    # data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')

    aggregated_data = aggregate_data(data)

    # Data splitting
    # x_train, y_train, x_test, y_weatherData = split_data(data)
    # x_train = aggregate_data(x_train)
    # y_train = aggregate_data(y_train, is_target=True)
    # x_test = aggregate_data(x_test)
    # y_weatherData = aggregate_data(y_weatherData)
    # x_train.to_csv('x_train.csv', index=False)
    # y_train.to_csv('y_train.csv', index=False)

    # x_test['Date'] = x_test['DateTime'].dt.date
    # x_test['Year'] = x_test['DateTime'].dt.year
    # x_test['Month'] = x_test['DateTime'].dt.month
    # x_test['Day'] = x_test['DateTime'].dt.day
    # x_test['LocationCode'] = x_test['LocationCode'].astype(str).str.zfill(2)

    # x_test['date_serial'] = (
    #     x_test['Year'].astype(str).str.zfill(4) +
    #     x_test['Month'].astype(str).str.zfill(2) +
    #     x_test['Day'].astype(str).str.zfill(2) +
    #     x_test['LocationCode']
    # )
    
    # x_test = x_test.drop(columns=['Year', 'Month', 'Day'])
    # unique = x_test["date_serial"].unique()
    # print(len(unique))

    # x_test = add_serial_number(x_test)
    # x_test.to_csv('x_test.csv', index=False)
    # y_weatherData = y_weatherData.drop(['Power(mW)'], axis=1)
    # y_weatherData.to_csv('y_weatherData.csv', index=False)

    aggregated_data = add_serial_number(aggregated_data)

    processed_x_test, real_train_data = generate_x_test(x_test, aggregated_data)
    x_train = real_train_data.drop(columns=['Power(mW)'])
    y_train = real_train_data['Power(mW)']
    print("done")
    return x_train, y_train, processed_x_test

def split_data(data):
    '''
    Treat the morning data with missing afternoon data days as test input, and the rest as training data.
    '''
#     print("{:=^100s}".format(" Splitting data into train and test sets... "))
#     data['Date'] = data['DateTime'].dt.date
#     data['Time'] = data['DateTime'].dt.time

#     x_train, y_train, x_test, y_weatherData = [], [], [], []

#     for (location, date), group in data.groupby(['LocationCode', 'Date']):
#         morning_data = group[group['DateTime'].dt.hour < 9]
#         afternoon_data = group[(group['DateTime'].dt.hour >= 9) & (group['DateTime'].dt.hour < 17)]
        
#         if afternoon_data.empty and not morning_data.empty:
#             x_test.append(morning_data.drop(['Time', 'Date'], axis=1))
#         else:
#             x_train.append(morning_data.drop(['Time', 'Date'], axis=1))
#             y_train.append(afternoon_data[['LocationCode', 'DateTime', 'Power(mW)']])
#             y_weatherData.append(afternoon_data.drop(['Time', 'Date'], axis=1))

#     x_train = pd.concat(x_train).reset_index(drop=True)
#     y_train = pd.concat(y_train).reset_index(drop=True)
#     x_test = pd.concat(x_test).reset_index(drop=True)
#     y_weatherData = pd.concat(y_weatherData).reset_index(drop=True)

    return 0 # x_train, y_train, x_test, y_weatherData

def aggregate_data(data, is_target=False):
    print("{:=^100s}".format(" Aggregating data... "))

    if not pd.api.types.is_datetime64_any_dtype(data['DateTime']):
        data['DateTime'] = pd.to_datetime(data['DateTime'])

    if is_target:
        aggregated_data = (
            data.groupby(['LocationCode', pd.Grouper(key='DateTime', freq='10min')])
            .agg({'Power(mW)': 'sum'})
            .reset_index()
        )
    else:
        aggregated_data = (
            data.groupby(['LocationCode', pd.Grouper(key='DateTime', freq='10min')])
            .agg({
                'WindSpeed(m/s)': 'mean',
                'Pressure(hpa)': 'mean',
                'Temperature(°C)': 'mean',
                'Humidity(%)': 'mean',
                'Sunlight(Lux)': 'mean',
                'Power(mW)': 'sum'
            })
            .reset_index()
        )

    aggregated_data = aggregated_data.dropna(how='all')

    print("done")
    return aggregated_data

def add_serial_number(data):
    print("{:=^100s}".format(" Adding Serial Number... "))
    data['Year'] = data['DateTime'].dt.year
    data['Month'] = data['DateTime'].dt.month
    data['Day'] = data['DateTime'].dt.day
    data['Hour'] = data['DateTime'].dt.hour
    data['Minute'] = data['DateTime'].dt.minute
    
    data['LocationCode'] = data['LocationCode'].astype(str).str.zfill(2)

    data['序號'] = (
        data['Year'].astype(str).str.zfill(4) +
        data['Month'].astype(str).str.zfill(2) +
        data['Day'].astype(str).str.zfill(2) +
        data['Hour'].astype(str).str.zfill(2) +
        data['Minute'].astype(str).str.zfill(2) +
        data['LocationCode']
    )
    
    data = data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

    print("done")
    return data

def generate_x_test(upload, aggregated_data):
    print("{:=^100s}".format(" Processing x_test... "))

    upload['MatchKey'] = upload['序號'].astype(str).str[:12]
    aggregated_data['MatchKey'] = aggregated_data['序號'].str[:12]

    matched_data = pd.merge(upload, aggregated_data, on='MatchKey', how='inner', suffixes=('_upload', '_aggregated'))
    matched_indices = matched_data.index

    x_test = matched_data.drop(columns=['答案', '序號_aggregated', 'MatchKey', 'Power(mW)'])
    x_test.rename(columns={'序號_upload': '序號'}, inplace=True)

    real_train_data = aggregated_data.drop(index=matched_indices)
    real_train_data = aggregated_data.drop(columns=['MatchKey'])

    print("done")
    return x_test, real_train_data

# 3. Feature Engineering
def feature_engineering(x_train, y_train, x_test):
    print("{:=^100s}".format(" Performing feature engineering... "))

    x_train_scaled = x_train
    y_train_scaled = y_train
    x_test_scaled = x_test

    print("done")
    return x_train_scaled, y_train_scaled, x_test_scaled

# 4. Model Training
def model_training(x_train, y_train):
    print("{:=^100s}".format(" Training model... "))

    features = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
    # target = 'Power(mW)'
    x_train = x_train[features]
    x_splitted_train, x_splitted_test, y_splitted_train, y_splitted_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Define and train the model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(x_splitted_train, y_splitted_train)

    # Evaluate the model
    y_pred = model.predict(x_splitted_test)
    mae = mean_absolute_error(y_splitted_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    print("done")
    return model

# 5. Model Evaluation
def model_evaluation(model, test_data):
    print("{:=^100s}".format(" Evaluating model... "))
    features = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
    # Test and evaluate the model
    result = model.predict(test_data[features])
    result = pd.DataFrame(result, columns=["答案"])
    test_data["答案"] = result["答案"]

    aggregated_result = (
        test_data.groupby('序號', sort=False)['答案']
        .mean()
        .reset_index()
    )
    # print(aggregated_result)
    # aggregated_result.to_csv('result1216.csv', index=False)
    # results = model.evaluate(...)
    # return results
    print("done")
    return aggregated_result

# Main program entry point
if __name__ == "__main__":
    # 1. Acquire data
    source_data, x_test = data_acquisition()
    
    # 2. Preprocess data
    x_train, y_train, processed_x_test = data_preprocessing(source_data, x_test)

    # 3. Perform feature engineering
    x_train_scaled, y_train_scaled, x_test_scaled = feature_engineering(x_train, y_train, processed_x_test)

    # 4. Train the model
    model = model_training(x_train_scaled, y_train_scaled)
    
    # 5. Evaluate the model
    evaluation_results = model_evaluation(model, x_test_scaled)

    result = x_test.merge(evaluation_results[['序號', '答案']], on='序號', how='left')
    result['答案_y'] = result['答案_y'].fillna(0)   # bad solution
    result.rename(columns={'答案_y': '答案'}, inplace=True)
    result = result.drop(columns=['MatchKey', '答案_x'])
    result.to_csv('result.csv', index=False)

    # output_df = pd.read_csv('output.csv')
    # upload_df = pd.read_csv('upload(no answer).csv')

    # upload_df['答案'] = output_df['答案']

    # # 檢查合併後的數據
    # print(upload_df.head())
    # upload_df.to_csv('output_and.csv', index=False)

    print("{:=^100s}".format(" Process complete! "))
