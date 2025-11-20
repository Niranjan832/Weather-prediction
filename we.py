import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import folium
from geopy.geocoders import Nominatim
from openmeteo_requests import Client
import matplotlib.pyplot as plt


data = pd.read_csv(r"N:\proj\sih\GlobalWeatherRepository.csv")


data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by=['location', 'date'])


user_location = input("Enter location (City or Country): ").strip()


loc_data = data[data['location'].str.contains(user_location, case=False, na=False)]

if loc_data.empty:
    print("Location not found in dataset! Try a nearby major city.")
    exit()


loc_data = loc_data.set_index('date')
features = ['temperature', 'humidity', 'wind_speed', 'precipitation']

X = np.arange(len(loc_data)).reshape(-1, 1)
predictions = {}


for feature in features:
    y = loc_data[feature].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(loc_data), len(loc_data) + 6).reshape(-1, 1)
    preds = model.predict(future_days)
    predictions[feature] = preds


last_date = loc_data.index[-1]
future_dates = [last_date + timedelta(days=i + 1) for i in range(6)]
pred_df = pd.DataFrame(predictions, index=future_dates)
print("\n📅 Predicted Weather for Next 6 Days:")
print(pred_df)


pred_df.plot(figsize=(10, 6), title=f"Weather Forecast for {user_location}")
plt.xlabel("Date")
plt.ylabel("Values")
plt.show()


geolocator = Nominatim(user_agent="weather_predictor")
location = geolocator.geocode(user_location)

if location:
    m = folium.Map(location=[location.latitude, location.longitude], zoom_start=6)
    popup_text = f"<b>Predicted Weather for {user_location}</b><br>"
    for date, row in pred_df.iterrows():
        popup_text += f"{date.date()}: 🌡️ {row['temperature']:.2f}°C, 💧 {row['humidity']:.1f}%, 🌬️ {row['wind_speed']:.1f} km/h, ☔ {row['precipitation']:.2f}mm<br>"
    folium.Marker(
        [location.latitude, location.longitude],
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(m)
    m.save("weather_forecast_map.html")
    print("\n🗺️ Map saved as 'weather_forecast_map.html'")
else:
    print("❌ Unable to locate coordinates for that location.")


client = Client()
latitude, longitude = location.latitude, location.longitude
response = client.weather_api(latitude=latitude, longitude=longitude, daily=["temperature_2m_max", "temperature_2m_min"])
print("\n🔗 Live Open-Meteo Data Retrieved for Comparison.")
