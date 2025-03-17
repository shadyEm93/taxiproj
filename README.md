
# OSRM-Powered Taxi duration prediction

This project processes NYC taxi trip data and predicts trip durations by calculating total duration and optimal routes using the Open Source Routing Machine (OSRM).

## Features
- Analyzes NYC taxi trip data.
- Calculates optimal routes using OSRM.
- Sums it with possible delays(by traffic or...) 
- Builds the model to predict duration for trips in NYC
- Fully dockerized setup for easy deployment.

## Prerequisites
- Create 2 folders in the project root called 'data' and 'output'
- Download OpenStreetMap extracts for NYC [here](https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf)
- Download the dataset from [kaggle](https://www.kaggle.com/datasets/kentonnlp/2014-new-york-city-taxi-trips)
- Docker and Docker Compose installed.


## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/shadyEm93/taxiproj.git
   cd taxiproj
2. Move those downloaded files (dataset and openstreet extracts) into 'data' folder before running anything else
   - The OSM file should be named `new-york-latest.osm`.
   - The dataset should be named `nyc_taxi_data_2014`
3. In the terminal use this code 'docker-compose build' to build the image (use vpn for this)
4. after that run this code in the terminal 'docker-compose up' and start the container (turn off the vpn to do this)
5. Open another terminal and run this command '  docker-compose logs -f taxi-app '
 
