


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import httpx as hx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

print("Waiting 10 seconds for OSRM service to initialize...", flush=True)
time.sleep(20)

#loading the dataset (also making sure the 8th column type does not cause any issues and selecting 5k rows since the dataset is heavy)
df = pd.read_csv("data/nyc_taxi_data_2014.csv", nrows=50000, dtype={8: str}, low_memory=False)


#creating a sample because the dataset is so heavy
sample = df.sample(n=1000, random_state=42)

#required columns
selected = ['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
col = sample[selected]
col = col.copy()

#required columns for OSRM backend
select = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
osm = sample[select]

#writing down the host and port (this host refers the osrm-backend container)
osrmhost = 'http://osrm'
osrmport = 5000

#function to get the real duration and distance with OSRM backend
def result(lonsource, latsource, londestination, latdestination, retries=3, delay=2):
    for attempt in range(1, retries + 1):
        requrl = f'{osrmhost}:{osrmport}/route/v1/driving/{lonsource},{latsource};{londestination},{latdestination}'
        try:
            response = hx.get(requrl, timeout=10)
            #parsing the response as JSON
            response_data = response.json()
            #making sure the routes exist in the response
            if 'routes' in response_data and len(response_data['routes']) > 0:
                bestrout = response_data['routes'][0]
                duration = bestrout['duration']
                distance = bestrout['distance']
                return duration, distance
            else:
                raise ValueError(f"No routes found in response: {response_data}")
        except Exception as e:
            print(f"Attempt {attempt} failed for coordinates ({lonsource}, {latsource} -> {londestination}, {latdestination}): {e}", flush=True)
            if attempt < retries:
                print(f"Retrying in {delay} seconds...", flush=True)
                time.sleep(delay)
    raise ValueError(f"All {retries} attempts failed for coordinates ({lonsource}, {latsource} -> {londestination}, {latdestination}).")



#initializing lists for real duration and distance
#valid_indices to get the valid results and use them
valid_indices = []
realduration = []
dist = []

#creating a for loop to use the function for all the entries, one value at a time
lat = osm['pickup_latitude']
lon = osm['pickup_longitude']
dlat = osm['dropoff_latitude']
dlon = osm['dropoff_longitude']

for idx, (l, j, p, z) in enumerate(zip(lat, lon, dlat, dlon)):
    try:
        duration, distance = result(lonsource=j, latsource=l, londestination=z, latdestination=p)
        realduration.append(duration)
        dist.append(distance)
        valid_indices.append(idx)
    except Exception as e:
        print(f"Error processing coordinates ({j}, {l} -> {z}, {p}): {e}")

#feature engineering
#filtering the DataFrame to include only rows with valid OSRM data
col = col.iloc[valid_indices].copy()

#turning some data to datetime to make them util
col['pickup_datetime'] = pd.to_datetime(col['pickup_datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
col['dropoff_datetime'] = pd.to_datetime(col['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

#using realduration as a feature
col[ 'real_duration'] = realduration

#calculating duration in minutes
col['duration'] = (col['dropoff_datetime'] - col['pickup_datetime']).dt.total_seconds() / 60
col['duration'] = col['duration'].round(2)

#calculating traffic delay
col['delay'] = col['duration'] - (col['real_duration'] / 60)  # Convert seconds to minutes for consistency

#determining day of week to use it as a feature
col['day_of_week'] = col['pickup_datetime'].dt.dayofweek

#using hours that the pickup happened to get rush hours
col['hour'] = col['pickup_datetime'].dt.hour

#calculating rush time
col['rush_time'] = col['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 16 <= x <= 18) else 0)

#using ideal distance calculated by OSRM as a feature p.s converting meters to kilometers
col['ideal_distance'] = np.array(dist) / 1000  

#using trip distance from the dataset
col['trip_distance'] = sample['trip_distance']

#removing outliers
col = col[(col['duration'] > 1) & (col['duration'] <= 50)]

#as the outliers persisted, i used isolation forrest algorythm 
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(col[['duration', 'real_duration', 'delay']])

#remove the outliers which were detected by iso forest
col = col[outliers == 1]


#feature selection and determining label
features = col[['real_duration', 'rush_time', 'day_of_week', 'trip_distance', 'delay', 'ideal_distance']]
label = col['duration']
x = features.values
y = label.values

#normalizing the data
norm = MinMaxScaler()
x = norm.fit_transform(x)

#parameters for Random Forest
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#initializing GridSearchCV
gridsearch = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=params, cv=5, n_jobs=-1, verbose=2)
gridsearch.fit(x, y)

#getting the best parameters with GridSearchCV
bestparams = gridsearch.best_params_
print(f'Best Parameters: {bestparams}')

#training the Random Forest model with best parameters
ranfor = RandomForestRegressor(**bestparams, random_state=42)
model = ranfor.fit(x, y)

#performing cross-validation
crossval = cross_val_score(ranfor, x, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -crossval
print(f'Cross-Validation MSE Scores: {cv_mse_scores}')
print(f'Average Cross-Validation MSE: {cv_mse_scores.mean()}')

#doing the prediction using cross-validation
y_pred_cv = cross_val_predict(ranfor, x, y, cv=5)

#ensuring 'output' directory exists
os.makedirs("output", exist_ok=True)

#evaluating the model with cross-validation predictions
mse = mean_squared_error(y, y_pred_cv)
r_squared = r2_score(y, y_pred_cv)
print(f'Cross-Validation MSE: {mse}', flush=True)
print(f'Cross-Validation R-squared: {r_squared}', flush=True)


#visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_cv, color='blue', alpha=0.5, label='Prediction vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Fit Line')
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.title('Prediction vs Actual values')
plt.legend()
plt.savefig("output/results_plot.png")
print("Script finished execution.", flush=True)
print("Plot saved to output/results_plot.png", flush=True)
plt.close()
