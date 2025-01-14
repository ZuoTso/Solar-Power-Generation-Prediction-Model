import os
import csv
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

# 3. Feature Engineering
def feature_engineering(x_train, y_train, x_test):
    print("{:=^100s}".format(" Performing feature engineering... "))

    # Extract hours and minutes as new features
    x_train['Hour'] = x_train['DateTime'].dt.hour
    x_train['Minute'] = x_train['DateTime'].dt.minute
    x_test['Hour'] = x_test['DateTime'].dt.hour
    x_test['Minute'] = x_test['DateTime'].dt.minute
    scaler = MinMaxScaler()
    columns_to_scale = ['Minute', 'Hour']
    x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
    x_test[columns_to_scale] = scaler.fit_transform(x_test[columns_to_scale])

    #drop DateTime
    x_train = x_train.drop(columns=['DateTime'])
    x_test = x_test.drop(columns=['DateTime'])

    x_train_scaled = x_train
    y_train_scaled = y_train
    x_test_scaled = x_test

    return x_train_scaled, y_train_scaled, x_test_scaled

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

    aggregated_data = add_serial_number(aggregated_data)

    all_processed_x_test = []
    all_x_train = []
    all_y_train = []
    x_test_chunks = [x_test.iloc[i:i + 48] for i in range(0, len(x_test), 48)]
    for idx, chunk in enumerate(x_test_chunks):
        processed_x_test, real_train_data = generate_x_test(chunk, aggregated_data)
        x_train = real_train_data.drop(columns=['Power(mW)'])
        y_train = real_train_data['Power(mW)']

        # 3. Perform feature engineering
        x_train_scaled, y_train_scaled, x_test_scaled = feature_engineering(x_train, y_train, processed_x_test)

        all_x_train.append(x_train_scaled)
        all_y_train.append(y_train_scaled)
        all_processed_x_test.append(x_test_scaled)

        print("done")

    return all_x_train, all_y_train, all_processed_x_test

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

    return data

def generate_x_test(upload, aggregated_data):
    print("{:=^100s}".format(" Processing x_test... "))

    upload = upload.copy()
    upload['MatchKey'] = upload['序號'].astype(str).str[:12]
    aggregated_data['MatchKey'] = aggregated_data['序號'].str[:12]

    matched_data = pd.merge(upload, aggregated_data, on='MatchKey', how='inner', suffixes=('_upload', '_aggregated'))
    
    # Handling missing matching data
    if matched_data.empty:
        print("{:=^100s}".format("No matching data for {upload['DateTime'].iloc[0].date()} at location {upload['LocationCode'].iloc[0]}. Filling with average values..."))
        
        # Extract time information from serial number
        upload['DateTime'] = pd.to_datetime(upload['DateTime'], format='%Y-%m-%d %H:%M:%S')
        upload['Week'] = upload['DateTime'].dt.isocalendar().week
        upload['Hour'] = upload['DateTime'].dt.hour
        upload['Minute'] = upload['DateTime'].dt.minute
        
        # Find the average of the same time period
        week = upload['Week'].iloc[0]
        hour = upload['Hour'].iloc[0]
        minute = upload['Minute'].iloc[0]

        # Filter out data with the same LocationCode, Week, Hour and Minute
        filtered_data = aggregated_data[
            (aggregated_data['DateTime'].dt.isocalendar().week == week) &
            (aggregated_data['DateTime'].dt.hour == hour) &
            (aggregated_data['DateTime'].dt.minute == minute)
        ]

        # Calculate average
        if not filtered_data.empty:
            avg_values = filtered_data.mean()
            for col in ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']:
                upload[col] = avg_values[col]
        else:
            print("{:=^100s}".format("Warning: No data available to fill for Week {week}, Hour {hour}, Minute {minute}."))
            upload[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']] = np.nan  # 可根據需求調整為 0 或其他值

        return upload, aggregated_data

    # If the match is successful, process the matched data
    matched_indices = matched_data.index
    x_test = matched_data.drop(columns=['答案', '序號_aggregated', 'MatchKey', 'Power(mW)'])
    x_test.rename(columns={'序號_upload': '序號'}, inplace=True)

    real_train_data = aggregated_data.drop(index=matched_indices)
    real_train_data = real_train_data.drop(columns=['MatchKey'])

    print("done")
    return x_test, real_train_data

# 4. Model Training
def model_training(x_train, y_train):
    print("{:=^100s}".format(" Training model... "))

    features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Hour', 'Minute']
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
    return None  # Replace with trained model

# source_data, x_test = data_acquisition()
# data.sort_values(by=["LocationCode","DateTime"], inplace=True)
# print(data.head(20))
# print(data["LocationCode"].dtype)

# 1. Acquire data
source_data, x_test = data_acquisition()

# 2. Preprocess data
x_train, y_train, processed_x_test = data_preprocessing(source_data, x_test)

# 4. Train the model
# model = model_training(x_train, y_train)