import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import folium
from geopy.geocoders import Nominatim


CSV_PATH = r"N:\proj\sih\GlobalWeatherRepository.csv"   # <- update if needed
DATE_COL = "last_updated"  # date/time column in your sample
LOCATION_COL = "location_name"  # can fallback to 'country'
LAT_COL = "latitude"
LON_COL = "longitude"


POSSIBLE_FEATURES = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "precip_mm",
    "pressure_mb",
    "cloud",
    "feels_like_celsius",
    "visibility_km",
    "uv_index",
    "air_quality_PM2.5"
]

SEQ_LEN = 7           
PRED_STEPS = 6        # number of future frames to predict
EPOCHS = 50
BATCH_SIZE = 16
MODEL_SAVE = "best_lstm_weather.keras"  # Changed to .keras format
MAP_OUT = "weather_forecast_map.html"
PLOT_OUT = "lstm_weather_forecast.png"
# --------------------------------------------------


def load_and_prepare(file_path):
    df = pd.read_csv(file_path)

    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # parse date
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    else:
        # try alternative column names
        for alt in ["last_updated", "last_updated_epoch", "date", "datetime"]:
            if alt in df.columns:
                df[DATE_COL] = pd.to_datetime(df[alt], errors='coerce')
                break

    # drop rows without date
    df = df.dropna(subset=[DATE_COL])

    # sort by date
    df = df.sort_values(by=DATE_COL)
    return df


def select_features(df, possible_features):
    features = [f for f in possible_features if f in df.columns]
    if not features:
        raise ValueError("None of the expected numeric features were found in CSV. "
                         "Expected one of: " + ", ".join(possible_features))
    return features


def filter_location(df, user_input):
    # match location_name or country (case-insensitive substring match)
    mask = pd.Series(False, index=df.index)
    if LOCATION_COL in df.columns:
        mask = mask | df[LOCATION_COL].astype(str).str.contains(user_input, case=False, na=False)
    if "country" in df.columns:
        mask = mask | df["country"].astype(str).str.contains(user_input, case=False, na=False)
    filtered = df[mask].copy()
    return filtered


def make_sequences(data_arr, seq_len):
    X, y = [], []
    for i in range(len(data_arr) - seq_len):
        X.append(data_arr[i:i+seq_len])
        y.append(data_arr[i+seq_len])  # predict next time-step vector
    return np.array(X), np.array(y)


def build_model(seq_len, n_features):
    """Build LSTM model with proper Input layer"""
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation="tanh"),
        Dropout(0.2),
        Dense(n_features)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def iterative_forecast(model, last_window, steps):
    """
    last_window: np.array shape (seq_len, n_features) scaled
    returns list of predicted (unscaled) windows in scaled space
    """
    seq = last_window.copy()
    preds = []
    for _ in range(steps):
        inp = seq.reshape(1, seq.shape[0], seq.shape[1])
        p = model.predict(inp, verbose=0)[0]   # shape (n_features,)
        preds.append(p)
        # slide window forward
        seq = np.vstack([seq[1:], p])
    return np.array(preds)  # shape (steps, n_features)


def main():
    print("Loading CSV...")
    df = load_and_prepare(CSV_PATH)
    print(f"Total rows in CSV: {len(df):,}")

    # choose features present
    features = select_features(df, POSSIBLE_FEATURES)
    print("Using features:", features)

    user_location = input("Enter location (location_name or country substring): ").strip()
    loc_df = filter_location(df, user_location)

    if loc_df.empty:
        print("No matching rows found for that location. Try a different string.")
        return

    # take most recent contiguous data per date for that location
    # Group by day (if times present) to get daily aggregates (mean)
    loc_df = loc_df.set_index(DATE_COL)

    try:
        daily = loc_df[features].resample("D").mean().dropna()
    except Exception:
        daily = loc_df[features].copy().dropna()

    if len(daily) < SEQ_LEN + 1:
        print(f"Not enough data for this location. Need at least {SEQ_LEN+1} time points; found {len(daily)}.")
        return

    print(f"Using {len(daily)} daily samples for training.")

    # ----- scale data -----
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily.values)

    # prepare sequences
    X, y = make_sequences(scaled, SEQ_LEN)
    print("Sequence shapes:", X.shape, y.shape)

    # train/test split
    split_idx = int(len(X) * 0.9)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # ----- build & train model -----
    model = build_model(seq_len=SEQ_LEN, n_features=len(features))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    print("Training LSTM...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ----- iterative prediction -----
    last_window = scaled[-SEQ_LEN:]
    preds_scaled = iterative_forecast(model, last_window, PRED_STEPS)
    preds = scaler.inverse_transform(preds_scaled)

    # build DataFrame for predictions
    last_date = daily.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(PRED_STEPS)]
    pred_df = pd.DataFrame(preds, index=future_dates, columns=features)
    print("\nPredictions (next {} days):".format(PRED_STEPS))
    print(pred_df.round(3))

    # ----- Plot predictions -----
    plt.figure(figsize=(12, 6))
    if "temperature_celsius" in features:
        plt.plot(daily.index[-(len(pred_df)+30):], daily["temperature_celsius"].values[-(len(pred_df)+30):], label="History")
        plt.plot(pred_df.index, pred_df["temperature_celsius"], marker="o", linestyle="--", label="Forecast")
        plt.title(f"Temperature Forecast for '{user_location}'")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_OUT)
        print(f"Plot saved to {PLOT_OUT}")

    # ----- Map visualization (Folium) -----
    lat = lon = None
    if LAT_COL in df.columns and LON_COL in df.columns:
        try:
            last_row = loc_df.sort_index().iloc[-1]
            lat = float(last_row[LAT_COL]) if LAT_COL in last_row.index else None
            lon = float(last_row[LON_COL]) if LON_COL in last_row.index else None
        except Exception:
            lat = lon = None

    if lat is None or lon is None:
        try:
            geolocator = Nominatim(user_agent="lstm_weather_geocoder")
            g = geolocator.geocode(user_location, timeout=10)
            if g:
                lat, lon = g.latitude, g.longitude
        except Exception as e:
            print("Geocoding failed:", e)

    if lat and lon:
        m = folium.Map(location=[lat, lon], zoom_start=6)
        popup_html = f"<b>Forecast for {user_location}</b><br>"
        for d, row in pred_df.iterrows():
            t = row.get("temperature_celsius", None)
            h = row.get("humidity", None)
            p = row.get("precip_mm", None)
            popup_html += f"{d.date()}: "
            if t is not None: popup_html += f"🌡 {t:.1f}°C "
            if h is not None: popup_html += f"💧{h:.0f}% "
            if p is not None: popup_html += f"☔{p:.2f}mm"
            popup_html += "<br>"
        folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=350)).add_to(m)
        m.save(MAP_OUT)
        print(f"Map saved to {MAP_OUT}")
    else:
        print("Could not determine latitude/longitude for map visualization.")

    pred_csv = f"predicted_weather_next_{PRED_STEPS}_days.csv"
    pred_df.to_csv(pred_csv)
    print(f"Predictions saved to {pred_csv}")


if __name__ == "__main__":
    main()