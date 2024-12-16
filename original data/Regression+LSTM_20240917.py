#%% 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

import numpy as np
import pandas as pd
import os
#%%
#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  #LSTM往前看的筆數
ForecastNum = 48  #預測筆數

#載入訓練資料
# x_train = pd.read_csv('x_train.csv')
# y_train = pd.read_csv('y_train.csv')
# x_test = pd.read_csv('x_test.csv')
SourceData = pd.read_csv('x_train.csv', encoding='utf-8')

#迴歸分析 選擇要留下來的資料欄位
#(風速,大氣壓力,溫度,濕度,光照度)
#(發電量)
Regression_X_train = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
Regression_y_train = SourceData[['Power(mW)']].values

#LSTM 選擇要留下來的資料欄位
#(風速,大氣壓力,溫度,濕度,光照度)
AllOutPut = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values

#正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []

#設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i-LookBackNum:i, :])
    y_train.append(AllOutPut_MinMax[i, :])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
#(samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

#============================建置&訓練「LSTM模型」============================
#建置LSTM模型
regressor = Sequential()
regressor.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 5)))
regressor.add(LSTM(units=64))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units=5))
regressor.compile(optimizer='adam', loss='mean_squared_error')

#開始訓練
regressor.fit(X_train, y_train, epochs=100, batch_size=128)

#保存模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save('WheatherLSTM_' + NowDateTime + '.h5')
print('Model Saved')


#============================建置&訓練「回歸模型」========================
#開始迴歸分析(對發電量做迴歸)
RegressionModel = LinearRegression()
RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

#儲存回歸模型
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
joblib.dump(RegressionModel, 'WheatherRegression_' + NowDateTime)
print(NowDateTime)
#取得截距
print('截距: ', RegressionModel.intercept_)

#取得係數
print('係數 : ', RegressionModel.coef_)

#取得R平方
print('R squared: ', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

#%%
#============================預測數據============================
#載入模型
print(NowDateTime)
regressor = load_model('WheatherLSTM_2024-09-23T01_10_50Z.h5')
Regression = joblib.load('WheatherRegression_2024-09-23T01_10_53Z')

#載入測試資料
DataName = os.getcwd() + r'\ExampleTestData\upload.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')

# 推測 EXquestion 可以從 x_test 推算出來
# 假設 x_test 是您的測試數據集，其中包含 DateTime 和 localcode 欄位
# 先轉換 DateTime 欄位為 datetime 格式
x_test["DateTime"] = pd.to_datetime(x_test["DateTime"])

# 選取 DateTime 和 localcode 欄位
x_test["Date"] = x_test["DateTime"].dt.date
x_test["Time"] = x_test["DateTime"].dt.strftime('%H%M')

# 定義過濾條件：時間在 09:00 - 16:50 之間，且每 10 分鐘為一個區間
x_test_filtered = x_test[(x_test["DateTime"].dt.hour >= 9) & (x_test["DateTime"].dt.hour < 17)]
x_test_filtered = x_test_filtered[(x_test_filtered["DateTime"].dt.minute % 10 == 0)]

# 生成代碼：'YYYYMMDDHHMMLocationCode'
x_test_filtered['Code'] = x_test_filtered['DateTime'].dt.strftime('%Y%m%d%H%M') + x_test_filtered['localcode'].astype(str).str.zfill(2)

# 顯示結果
print(x_test_filtered[['DateTime', 'localcode', 'Code']])
# EXquestion = SourceData[['Serial']].values

x_test = pd.read_csv('x_test.csv')  # 讀取測試資料
EXquestion = x_test['Serial'].values  # 假設 Serial 欄位是用來推算 EXquestion 的

inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值(天氣參數)
PredictPower = []  # 存放預測值(發電量)

count = 0
while(count < len(EXquestion)):
    print('count : ', count)
    LocationCode = int(EXquestion[count])
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = '0' + str(LocationCode)

    DataName = os.getcwd() + '\\ExampleTrainData(IncompleteAVG)\\IncompleteAvgDATA_' + strLocationCode + '.csv'
    SourceData = pd.read_csv(DataName, encoding='utf-8')
    ReferTitle = SourceData[['Serial']].values
    ReferData = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
    
    inputs = []  # 重置存放參考資料

    #找到相同的一天，把12個資料都加進inputs
    for DaysCount in range(len(ReferTitle)):
        if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
            TempData = ReferData[DaysCount].reshape(1, -1)
            TempData = LSTM_MinMaxModel.transform(TempData)
            inputs.append(TempData)

    #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
    for i in range(ForecastNum):
        #將新的預測值加入參考資料(用自己的預測值往前看)
        if i > 0:
            inputs.append(PredictOutput[i - 1].reshape(1, 5))

        #切出新的參考資料12筆(往前看12筆)
        X_test = []
        X_test.append(inputs[0 + i:LookBackNum + i])

        #Reshaping
        NewTest = np.array(X_test)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 5))

        predicted = regressor.predict(NewTest)
        PredictOutput.append(predicted)
        PredictPower.append(np.round(Regression.predict(predicted), 2).flatten())

    #每次預測都要預測48個，因此加48個會切到下一天
    count += 48

#寫預測結果寫成新的CSV檔案
df = pd.DataFrame(PredictPower, columns=['答案'])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv('output.csv', index=False)
print('Output CSV File Saved')
#%%