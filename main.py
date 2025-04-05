import pandas as pd
import xgboost as xgb
import random

bus_trips = pd.read_csv('./data/bus_running_times_654.csv')
bus_dwells = pd.read_csv('./data/bus_dwell_times_654.csv')

bus_trips = bus_trips.dropna()
bus_dwells= bus_dwells.dropna()

bus_trips["weekday"] = pd.to_datetime(bus_trips["date"]).dt.weekday
bus_trips["start_time"] = pd.to_datetime(bus_trips["start_time"], format="%H:%M:%S").dt.hour
bus_dwells["weekday"] = pd.to_datetime(bus_dwells["date"]).dt.weekday
bus_dwells["start_time"] = pd.to_datetime(bus_dwells["arrival_time"], format="%H:%M:%S").dt.hour

bus_trips_X = bus_trips[["direction", "segment", "start_time", "weekday"]]
bus_trips_Y = bus_trips["run_time_in_seconds"]
bus_dwells_X = bus_dwells[["direction", "bus_stop", "start_time", "weekday"]]
bus_dwells_Y = bus_dwells["dwell_time_in_seconds"]

# Define model with fixed hyperparameters
bus_trips_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.01,
    max_depth=7,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1
)

bus_dwells_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.01,
    max_depth=7,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1
)

bus_trips_model.fit(bus_trips_X, bus_trips_Y)
bus_dwells_model.fit(bus_dwells_X, bus_dwells_Y)

results = pd.DataFrame(columns=['expected', 'real'])

for _ in range(2000):
    random_trip_id = bus_trips['trip_id'].sample(1).iloc[0]
    direction = bus_trips[(bus_trips['trip_id'] == random_trip_id)]['direction'].iloc[0]

    if direction == 1:
        random_start = random.randint(1, 15)
        random_end = random.randint(random_start, 15)
    else:
        random_start = random.randint(21, 34)
        random_end = random.randint(random_start, 34)

    expected = 0
    real = 0

    for i in range(random_start, random_end+1):
        random_trip = bus_trips[
            (bus_trips['trip_id'] == random_trip_id) &
            (bus_trips['segment'] == i)]

        if random_trip.empty:
            average_value = bus_trips[(bus_trips['segment'] == i)]["run_time_in_seconds"].mean()
            expected += average_value
            real += average_value
            continue

        random_trip_x = random_trip[["direction", "segment", "start_time", "weekday"]]
        random_trip_y = bus_trips_model.predict(random_trip_x)

        expected += random_trip_y
        real += random_trip["run_time_in_seconds"].iloc[0]

    if direction == 1:
        random_start += 100
        random_end += 100
    else:
        random_start += 180
        random_end += 180

    for i in range(random_start, random_end):
        random_trip = bus_dwells[
            (bus_dwells['trip_id'] == random_trip_id) &
            (bus_dwells['bus_stop'] == i)]

        if random_trip.empty:
            average_value = bus_dwells[(bus_dwells['bus_stop'] == i)]["dwell_time_in_seconds"].mean()
            expected += average_value
            real += average_value
            continue

        random_trip_x = random_trip[["direction", "bus_stop", "start_time", "weekday"]]
        random_trip_y = bus_dwells_model.predict(random_trip_x)

        expected += random_trip_y
        real += random_trip["dwell_time_in_seconds"].iloc[0]

    results.loc[len(results)] = [expected, real]

results['absolute_error'] = abs(results['expected'] - results['real'])
mae = results['absolute_error'].mean()
print(f"Mean Absolute Error: {mae}")

# Print the first few rows
print(results.head(10))