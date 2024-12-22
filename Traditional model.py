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
    # print("{:=^100s}".format(" Performing feature engineering... "))

    # Extract hours and minutes as new features
    x_train['Hour'] = x_train['DateTime'].dt.hour
    x_train['Minute'] = x_train['DateTime'].dt.minute
    x_test['Hour'] = x_test['DateTime'].dt.hour
    x_test['Minute'] = x_test['DateTime'].dt.minute
    scaler = MinMaxScaler()
    columns_to_scale = ['Minute', 'Hour']
    x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
    if x_test[columns_to_scale].empty:
        print("x_test is empty, skipping scaling.")
    else:
        x_test[columns_to_scale] = scaler.fit_transform(x_test[columns_to_scale])

    #drop DateTime
    x_train = x_train.drop(columns=['DateTime'])
    x_test = x_test.drop(columns=['DateTime'])

    x_train_scaled = x_train
    y_train_scaled = y_train
    x_test_scaled = x_test

    # print("done")
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

    all_processed_x_test = []
    all_x_train = []
    all_y_train = []
    x_test_chunks = [x_test.iloc[i:i + 48] for i in range(0, len(x_test), 48)]
    for idx, chunk in enumerate(x_test_chunks):
        processed_x_test, real_train_data = generate_x_test(chunk, aggregated_data)
        x_train = real_train_data.drop(columns=['Power(mW)'])
        y_train = real_train_data['Power(mW)']

        # 3. Perform feature engineering
        if processed_x_test.empty:
            print(idx)
        x_train_scaled, y_train_scaled, x_test_scaled = feature_engineering(x_train, y_train, processed_x_test)

        all_x_train.append(x_train_scaled)
        all_y_train.append(y_train_scaled)
        all_processed_x_test.append(x_test_scaled)

    print("done")

    return all_x_train, all_y_train, all_processed_x_test

def split_data(data):
    '''
    Treat the morning data with missing afternoon data days as test input, and the rest as training data.
    
    print("{:=^100s}".format(" Splitting data into train and test sets... "))
    data['Date'] = data['DateTime'].dt.date
    data['Time'] = data['DateTime'].dt.time

    x_train, y_train, x_test, y_weatherData = [], [], [], []

    for (location, date), group in data.groupby(['LocationCode', 'Date']):
        morning_data = group[group['DateTime'].dt.hour < 9]
        afternoon_data = group[(group['DateTime'].dt.hour >= 9) & (group['DateTime'].dt.hour < 17)]
        
        if afternoon_data.empty and not morning_data.empty:
            x_test.append(morning_data.drop(['Time', 'Date'], axis=1))
        else:
            x_train.append(morning_data.drop(['Time', 'Date'], axis=1))
            y_train.append(afternoon_data[['LocationCode', 'DateTime', 'Power(mW)']])
            y_weatherData.append(afternoon_data.drop(['Time', 'Date'], axis=1))

    x_train = pd.concat(x_train).reset_index(drop=True)
    y_train = pd.concat(y_train).reset_index(drop=True)
    x_test = pd.concat(x_test).reset_index(drop=True)
    y_weatherData = pd.concat(y_weatherData).reset_index(drop=True)
    '''

    return 0 # x_train, y_train, x_test, y_weatherData

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
    # print("{:=^100s}".format(" Processing x_test... "))

    upload = upload.copy()
    aggregated_data = aggregated_data.copy()

    upload['MatchKey'] = upload['序號'].astype(str).str[:12]
    aggregated_data['MatchKey'] = aggregated_data['序號'].str[:12]

    matched_data = pd.merge(upload, aggregated_data, on='MatchKey', how='inner', suffixes=('_upload', '_aggregated'))
    
    # Handling missing matching data
    if matched_data.empty:
        # 從序號中提取時間和地點信息
        upload['序號'] = upload['序號'].astype(str)
        upload['Year'] = upload['序號'].str[:4].astype(int)
        upload['Month'] = upload['序號'].str[4:6].astype(int)
        upload['Day'] = upload['序號'].str[6:8].astype(int)
        upload['Hour'] = upload['序號'].str[8:10].astype(int)
        upload['Minute'] = upload['序號'].str[10:12].astype(int)
        upload['LocationCode'] = upload['序號'].str[12:14]

        # Create DateTime column
        upload['DateTime'] = pd.to_datetime(upload[['Year', 'Month', 'Day', 'Hour', 'Minute']])

        # Fill missing data with average from preceding and following three days
        filled_data = []
        for _, row in upload.iterrows():
            target_time = row['DateTime']

            # Define time range for +/- 3 days
            start_date = target_time - pd.Timedelta(days=3)
            end_date = target_time + pd.Timedelta(days=3)

            # Filter aggregated_data for the same hour and minute in the time range
            relevant_data = aggregated_data[
                (aggregated_data['DateTime'] >= start_date) &
                (aggregated_data['DateTime'] <= end_date) &
                (aggregated_data['DateTime'].dt.hour == target_time.hour) &
                (aggregated_data['DateTime'].dt.minute == target_time.minute)
            ]

            if not relevant_data.empty:
                avg_values = relevant_data.select_dtypes(include=[np.number]).mean()
                filled_row = row.to_dict()
                for col in avg_values.index:
                    filled_row[col] = avg_values[col]
                filled_data.append(filled_row)
            else:
                print(f"Warning: No data available to fill for {target_time}. Filling with NaN.")
                filled_row = row.to_dict()
                for col in aggregated_data.select_dtypes(include=[np.number]).columns:
                    filled_row[col] = np.nan
                filled_data.append(filled_row)

        upload = pd.DataFrame(filled_data)
        x_test = upload.drop(columns=['答案', 'MatchKey', 'Power(mW)', 'Year', 'Month', 'Day', 'Hour', 'Minute'])
        real_train_data = aggregated_data.drop(columns=['MatchKey'])

        print("Data filled with averages from +/- 3 days.")
        return x_test, real_train_data

    # If the match is successful, process the matched data
    matched_indices = matched_data.index
    x_test = matched_data.drop(columns=['答案', '序號_aggregated', 'MatchKey', 'Power(mW)'])
    x_test.rename(columns={'序號_upload': '序號'}, inplace=True)

    real_train_data = aggregated_data.drop(index=matched_indices)
    real_train_data = real_train_data.drop(columns=['MatchKey'])

    # print("done")
    return x_test, real_train_data

# All 'temp' are for testing and can be deleted after testing.
# 4. Model Training Random Forest temp
def model_training(all_x_train, all_y_train):
    print("{:=^100s}".format(" Training model... "))

    features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Hour', 'Minute']

    # Define the Random Forest model
    random_forest_model = RandomForestRegressor(random_state=42, n_estimators=100)

    # List to store models for each dataset
    trained_models = []

    for i, (x_train, y_train) in enumerate(zip(all_x_train, all_y_train)):

        # if i != 1: # specific_idx temp
        #     continue  # 跳過不需要的組 temp
        print(f"Dataset {i + 1}:")

        # Filter the features
        x_train = x_train[features]
        x_splitted_train, x_splitted_test, y_splitted_train, y_splitted_test = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Train and evaluate the Random Forest model
        # print("{:=^100s}".format(" Training Random Forest... "))
        random_forest_model.fit(x_splitted_train, y_splitted_train)
        y_pred = random_forest_model.predict(x_splitted_test)
        mae = mean_absolute_error(y_splitted_test, y_pred)
        print(f"Random Forest {i + 1} Split MAE: {mae:.2f}")

        # Append the trained model for this dataset
        trained_models.append({'Random Forest': random_forest_model})


        # for a in range(199):    # temp
        #     trained_models.append({'Random Forest': random_forest_model})   # temp
        # return trained_models # temp

    print("done")

    return trained_models

# 4. Model Training
def model_training_stop(all_x_train, all_y_train):
    print("{:=^100s}".format(" Training model... "))

    features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Hour', 'Minute']

    # Define base models
    base_models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
    }
    # Dictionary to store models for each dataset
    trained_models = []

    for i, (x_train, y_train) in enumerate(zip(all_x_train, all_y_train)):
        print(f"Dataset {i + 1}:")
        
        # Filter the features
        x_train = x_train[features]
        x_splitted_train, x_splitted_test, y_splitted_train, y_splitted_test = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Train and evaluate each base model
        dataset_models = {}
        # Train and evaluate each base model with cross-validation
        for name, model in base_models.items():
            # cross-validation
            # cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            # mean_cv_score = -np.mean(cv_scores)
            # print(f"{name} CV MAE: {mean_cv_score:.2f}")
            # cross-validation end

            # Fit the model for final evaluation on split data
            model.fit(x_splitted_train, y_splitted_train)
            y_pred = model.predict(x_splitted_test)
            mae = mean_absolute_error(y_splitted_test, y_pred)
            print(f"{name} Split MAE: {mae:.2f}")
            dataset_models[name] = model

        # Stacking model
        print("{:=^100s}".format(" Stacking models... "))
        estimators = [(name, model) for name, model in base_models.items()]
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0)  # Ridge regression as the meta-model
        )

        # Evaluate stacking model with cross-validation
        # cv_scores = cross_val_score(stacking_model, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        # mean_cv_score = -np.mean(cv_scores)
        # print(f"Stacking Model CV MAE: {mean_cv_score:.2f}")
        # Evaluate stacking model with cross-validation end

        # Fit and evaluate stacking model on split data
        stacking_model.fit(x_splitted_train, y_splitted_train)
        y_pred = stacking_model.predict(x_splitted_test)
        mae = mean_absolute_error(y_splitted_test, y_pred)
        print(f"Stacking Model Split MAE: {mae:.2f}")
        dataset_models['Stacking Model'] = stacking_model

        # Append models for this dataset
        trained_models.append(dataset_models)

    print("done")

    return trained_models

# 5. Model Evaluation
def model_evaluation(trained_models, all_test_data):
    print("{:=^100s}".format(" Evaluating model... "))
    features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Hour', 'Minute']

    # Store aggregated results for all datasets
    aggregated_results = []
    for i, (models, test_data) in enumerate(zip(trained_models, all_test_data)):
        # print(f"Evaluating on Test Dataset {i + 1}:")
        test_data = test_data.copy()
        
        # Evaluate the stacking model
        # stacking_model = models['Stacking Model']
        stacking_model = models['Random Forest']
        test_data["答案"] = stacking_model.predict(test_data[features])
        # Aggregate results by "序號"
        aggregated_result = (
            test_data.groupby('序號', sort=False)['答案']
            .mean()
            .round(2)
            .reset_index()
        )

        aggregated_results.append(aggregated_result)

        # print(f"Dataset {i + 1} evaluation completed.")

    print("done")
    return aggregated_results

def combine_aggregated_results(aggregated_results):
    print("{:=^100s}".format(" Combining Results... "))
    
    # Concatenate all aggregated results
    combined_results = pd.concat(aggregated_results, ignore_index=True)
    
    print("Combining completed.")
    return combined_results

# Main program entry point
if __name__ == "__main__":
    # 1. Acquire data
    source_data, x_test = data_acquisition()
    
    # 2. Preprocess data
    x_train, y_train, processed_x_test = data_preprocessing(source_data, x_test)

    # 3. Perform feature engineering

    # 4. Train the model
    models = model_training(x_train, y_train)
    
    # 5. Evaluate the model
    evaluation_results = model_evaluation(models, processed_x_test)

    result = combine_aggregated_results(evaluation_results)

    print(result)

    # result = x_test.merge(evaluation_results[['序號', '答案']], on='序號', how='left')
    # result['答案_y'] = result['答案_y'].fillna(result['答案_y'].mean()) # bad solution
    # result.rename(columns={'答案_y': '答案'}, inplace=True)
    # result = result.drop(columns=['MatchKey', '答案_x'])
    result.to_csv('result.csv', index=False)

    # output_df = pd.read_csv('output.csv')
    # upload_df = pd.read_csv('upload(no answer).csv')

    # upload_df['答案'] = output_df['答案']

    # # 檢查合併後的數據
    # print(upload_df.head())
    # upload_df.to_csv('output_and.csv', index=False)

    print("{:=^100s}".format(" Process complete! "))
