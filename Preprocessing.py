import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

print("Start Preprocesssing..")
path_train = "../phase1_training/"
path_test = "../phase1_test/"
path= "../Pre/"

def missing_weather(type):
    if type == "train":
        df = pd.read_csv("train.csv")
    else:
        df = pd.read_csv("test.csv")

    # fill up precipitation, rel_huanitiy
    temp_pre = df["precipitation"]
    temp_rel = df["rel_humidity"]

    # step 1 : copy data from latest data
    for i in range(len(df)):
        if str(temp_pre[i]) == "nan":
            temp_pre[i] = temp_pre[i - 3]
            # print(temp_pre[i])

    for i in range(len(df)):
        if str(temp_rel[i]) == "nan":
            temp_rel[i] = temp_rel[i - 3]
            # print(temp_pre[i])

    # step 2 : if there is no data from afternoon : fill na with 0 and median

    df["precipitation"] = df["precipitation"].fillna(0)
    df["rel_humidity"] = df["rel_humidity"].fillna(df["rel_humidity"].median())

    # make time window
    df["time_window"] = df["time"].apply(lambda x: build_time_window(x))

    if type == "train":
        df.to_csv("train.csv")
    else:
        df.to_csv("test.csv")


def build_time_window(x):
    import datetime

    time_window = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    time_next = time_window + datetime.timedelta(minutes=20)

    return '[' + str(time_window) + ',' + str(time_next) + ')'


# join with weather dataset and training dataset
def generate_train(filename):
    df_volume = pd.read_csv(filename)

    df_volume["time"] = pd.to_datetime(df_volume["time"])

    df_volume = df_volume.sort_values(by=['tollgate_id', 'direction', 'time'])
    df_volume["am_pm"] = df_volume["hour"].apply(lambda x: ampm(x))

    df_volume["period_num"] = df_volume["hour"].apply(lambda x: calc_period_num(x))

    # assume that volume increasing every 20 mins from starting point.
    df_volume["period_num"] = df_volume["period_num"] + df_volume["miniute"].apply(lambda x: x / 20)

    df_volume["hour1"] = df_volume["hour"].apply(lambda x: x / 3 * 3)
    df_weather = pd.read_csv("../Pre/weather_feature.csv")[["date", "hour1", "precipitation", "rel_humidity"]]

    df_volume = df_volume.merge(df_weather, on=["date", "hour1"], how="left")  # join left
    df_volume = df_volume.drop("hour1", axis=1)

    return df_volume


# load weather data
def weather():
    path = "../weather/"
    df_weather = pd.read_csv(path + "weather_July_01_Oct_17_table7.csv")
    df_weather = df_weather.append(
        pd.read_csv(path + "weather_Oct_18_Oct_24_table7.csv"))

    # will aggregate with training dataset's hour
    df_weather["hour1"] = df_weather["hour"]

    df_weather[
        ["date", "hour1", "pressure", "sea_pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity",
         "precipitation"]].to_csv("../Pre/weather_feature.csv", index=False)


# to handle exception of date
def exception():
    # generate date information
    # col  'holiday'    0: workday   1: weekend   2: holiday
    # col   'first_last_workday'     1: first workday of week  2: last workday of week  0: other
    # col  'day_of_week'     1: Mon  2: Tues    ...  7 : Sun

    df_date = pd.DataFrame(columns=('date', 'holiday', 'first_last_workday', 'day_of_week'))
    df_date.loc[0] = ['2016-09-19', '0', '0', '1']
    df_date.loc[1] = ['2016-09-20', '0', '0', '2']
    df_date.loc[2] = ['2016-09-21', '0', '0', '3']
    df_date.loc[3] = ['2016-09-22', '0', '0', '4']
    df_date.loc[4] = ['2016-09-23', '0', '2', '5']
    df_date.loc[5] = ['2016-09-24', '1', '0', '6']
    df_date.loc[6] = ['2016-09-25', '1', '0', '7']
    df_date.loc[7] = ['2016-09-26', '0', '1', '1']
    df_date.loc[8] = ['2016-09-27', '0', '0', '2']
    df_date.loc[9] = ['2016-09-28', '0', '0', '3']
    df_date.loc[10] = ['2016-09-29', '0', '0', '4']
    df_date.loc[11] = ['2016-09-30', '0', '2', '5']
    df_date.loc[12] = ['2016-10-01', '2', '0', '6']
    df_date.loc[13] = ['2016-10-02', '2', '0', '7']
    df_date.loc[14] = ['2016-10-03', '2', '0', '1']
    df_date.loc[15] = ['2016-10-04', '2', '0', '2']
    df_date.loc[16] = ['2016-10-05', '2', '0', '3']
    df_date.loc[17] = ['2016-10-06', '2', '0', '4']
    df_date.loc[18] = ['2016-10-07', '2', '0', '5']
    df_date.loc[19] = ['2016-10-08', '0', '1', '6']
    df_date.loc[20] = ['2016-10-09', '0', '0', '7']
    df_date.loc[21] = ['2016-10-10', '0', '0', '1']
    df_date.loc[22] = ['2016-10-11', '0', '0', '2']
    df_date.loc[23] = ['2016-10-12', '0', '0', '3']
    df_date.loc[24] = ['2016-10-13', '0', '0', '4']
    df_date.loc[25] = ['2016-10-14', '0', '2', '5']
    df_date.loc[26] = ['2016-10-15', '1', '0', '6']
    df_date.loc[27] = ['2016-10-16', '1', '0', '7']
    df_date.loc[28] = ['2016-10-17', '0', '1', '1']
    df_date.loc[29] = ['2016-10-18', '0', '0', '2']
    df_date.loc[30] = ['2016-10-19', '0', '0', '3']
    df_date.loc[31] = ['2016-10-20', '0', '0', '4']
    df_date.loc[32] = ['2016-10-21', '0', '2', '5']
    df_date.loc[33] = ['2016-10-22', '1', '0', '6']
    df_date.loc[34] = ['2016-10-23', '1', '0', '7']
    df_date.loc[35] = ['2016-10-24', '0', '1', '1']
    df_date.loc[36] = ['2016-10-25', '0', '0', '2']
    df_date.loc[37] = ['2016-10-26', '0', '0', '3']
    df_date.loc[38] = ['2016-10-27', '0', '0', '4']
    df_date.loc[39] = ['2016-10-28', '0', '2', '5']
    df_date.loc[40] = ['2016-10-29', '1', '0', '6']
    df_date.loc[41] = ['2016-10-30', '1', '0', '7']
    df_date.loc[42] = ['2016-10-31', '0', '1', '1']
    df_date.loc[43] = ['2016-11-01', '0', '0', '2']

    # print(df_date)
    # print("date finish preprocesing")
    df_date.to_csv("execption_date.csv", index=False)


def run(df, rng):
    # count the volume for speciic time period and direction

    rng_length = len(rng)
    result_dfs = []

    # id-direction
    # 1-entry, 1-exit, 2-entry, 3-entry and 3-exit)

    # 0:entry, 1: exit
    # tolid_direc=[(1,0),(1,1),(2,0),(3,0),(3,1)]

    for this_tollgate_id in range(1, 4):
        for this_direction in range(2):

            if this_tollgate_id == 2 and this_direction == 1:
                continue

            time_start_list = []
            volume_list = []
            direction_list = []
            tollgate_id_list = []

            this_df = df[(df.tollgate_id == this_tollgate_id) & (df.direction == this_direction)]

            if len(this_df) > 0:
                for ind in range(rng_length - 1):
                    this_df_time_window = this_df[(this_df.time >= rng[ind]) & (this_df.time < rng[ind + 1])]
                    volume_list.append(len(this_df_time_window))

                    time_start_list.append(rng[ind])

                result_df = pd.DataFrame({'time_start': time_start_list,
                                          'volume': volume_list,
                                          'direction': [this_direction] * (rng_length - 1),
                                          'tollgate_id': [this_tollgate_id] * (rng_length - 1),
                                          }
                                         )

                result_dfs.append(result_df)

    d = pd.concat(result_dfs)

    if type == 'test':
        d['hour'] = d['time_start'].apply(lambda x: x.hour)
        d = d[d.hour.isin([6, 7, 15, 16])]

    # print(d)
    return d


# split time_start -> time, hour, min
def df_filter(df_volume):
    df_volume["time"] = df_volume["time_start"]
    # df_volume["date"] = df_volume["time"].apply(lambda x: pd.to_datetime(x[: 10]))
    df_volume["date"] = df_volume["time"].apply(lambda x: x.date())
    df_volume["hour"] = df_volume["time"].apply(lambda x: x.hour)  # x.hour
    df_volume["miniute"] = df_volume["time"].apply(lambda x: x.minute)  # x.minute

    df_volume["time_window"] = df_volume["time"]

    df_volume = df_volume[["tollgate_id", "time_window", "direction", "volume", "time", "date", "hour", "miniute"]]
    return df_volume


# return 1 if is am
def ampm(x):
    if (x <= 12):
        return 1
    return 0


##get period
def calc_period_num(x):
    # Rush hour  == 1, otherwide 4
    if x == 17 or x == 8:
        return 1
    return 4

# join with weather dataset and training dataset
def get_train():
    path = "../Pre/"
    print(os.getcwd())
    df1 = generate_train(path +"0_train_filter.csv")
    df2 = generate_train(path + "5_train_filter.csv")
    df3 = generate_train(path + "10_train_filter.csv")
    df4 = generate_train(path + "15_train_filter.csv")

    df_total_list = [df1, df2, df3, df4]
    df_total = pd.concat(df_total_list)

    df_total.to_csv("train.csv")


# join with weather dataset and training dataset step3 : for test
def get_test():
    path = "../Pre/"
    df1 = generate_train(path + "0_test_filter.csv")
    df1.to_csv("test.csv", index=False)


def create_timeseiries():
    df_train = pd.read_csv("train.csv")
    df_grouped = df_train.groupby(["tollgate_id", "direction"])
    ts_feature = []

    for key, data in df_grouped:
        # print(key)

        data = pd.DataFrame.fillna(data, 0)
        data["hour"] = data["time"].apply(lambda x: int(x[11: 13]))
        data["miniute"] = data["time"].apply(lambda x: int(x[14: 16]))
        data["time"] = pd.to_datetime(data["time"])
        data = data.sort_values(by=["time"])
        data = data.set_index("time")

        ts = data["volume"]
        ts = ts.asfreq('5Min', method='pad')

        title = "tollgate id = " + str(key[0]) + ", direction = " + str(key[1])
        plt.figure(figsize=(40, 10))
        plt.plot(ts)
        plt.title(title)

        # create timeseiries report
        freq = ((24 * 60) // 5)
        decomposition = seasonal_decompose(ts, freq=freq)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        seasonal.name = 'seasonal'
        residual.name = 'residual'
        trend.name = 'trend'
        data = data[["tollgate_id", "direction", "hour", "miniute"]]
        data = pd.concat([data, seasonal], axis=1)

        data = data.drop_duplicates(['hour', 'miniute', 'seasonal'])
        ts_feature.append(data)

    df_ts = pd.concat(ts_feature, axis=0)

    df_ts.to_csv("timeseries.csv", index=False)

    print("./code/timeseries.csv")


def dataPrepare():
    # make new data.csv file
    # path_train = "./phase1_training/"

    df_train = pd.read_csv(path_train + "volume_training_phase1_table6.csv", parse_dates=['time'])
    # path_test = "./phase1_test/"
    df_test = pd.read_csv(path_test + "volume_test_phase1_table6.csv", parse_dates=['time'])

    df_test = df_test.rename(
        columns={'date_time': 'time', 'tollgate': 'tollgate_id', 'is_etc': 'has_etc', 'veh_type': 'vehicle_type',
                 'model': 'vehicle_model'})

    freq = "20min"
    # move time window 0 5 10 15 minute
    rng1 = pd.date_range("2016-09-19 00:00:00", "2016-10-18 00:00:00", freq=freq)
    rng2 = pd.date_range("2016-09-19 00:05:00", "2016-10-18 00:00:00", freq=freq)
    rng3 = pd.date_range("2016-09-19 00:10:00", "2016-10-18 00:00:00", freq=freq)
    rng4 = pd.date_range("2016-09-19 00:15:00", "2016-10-18 00:00:00", freq=freq)
    rng5 = pd.date_range("2016-10-18 00:00:00", "2016-10-25 00:00:00", freq=freq)

    df_train1 = run(df_train, rng1)
    df_train2 = run(df_train, rng2)
    df_train3 = run(df_train, rng3)
    df_train4 = run(df_train, rng4)
    df_test = run(df_test, rng5)

    # preprocessing time to test and train : split time_start -> time, hour, min
    df_filter(df_test).to_csv(path + "0_test_filter.csv", index=False)
    df_filter(df_train1).to_csv(path + "0_train_filter.csv", index=False)
    df_filter(df_train2).to_csv(path + "5_train_filter.csv", index=False)
    df_filter(df_train3).to_csv(path + "10_train_filter.csv", index=False)
    df_filter(df_train4).to_csv(path + "15_train_filter.csv", index=False)

##main##


def main():
    weather()
    print("../Pre/weather_feature.csv")
    exception()
    print("./code/execption_date.csv")

    # data
    dataPrepare()

    get_train()
    missing_weather('train')
    get_test()
    missing_weather('test')

    print("./code/train.csv, test.csv")

    create_timeseiries()

    print("Finish Preprocesssing")


if __name__ == '__main__':
    main()



