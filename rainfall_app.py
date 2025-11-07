import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rainfall Event Analyzer", layout="wide")

st.title("🌧️ Rainfall Event Analyzer")
st.write("Upload your hourly rainfall dataset (CSV) with columns: `station_id`, `date`, `hour`, `rainfall`")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Basic check
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Ensure datetime format
    if 'date' in df.columns and 'hour' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00', errors='coerce')
        df = df.dropna(subset=['datetime'])
    else:
        st.error("CSV must contain 'date' and 'hour' columns!")
        st.stop()

    # Select station
    stations = df['station_id'].unique()
    station = st.selectbox("Select Station", stations)

    df_station = df[df['station_id'] == station].copy()
    df_station = df_station.sort_values('datetime')

    st.write(f"Selected Station: **{station}**")

    # Identify rainfall events
    threshold = st.slider("Rainfall threshold (mm)", 0.1, 10.0, 1.0, step=0.1)
    gap_hours = st.slider("Gap (hours) between events", 1, 12, 6)

    df_station['rain_event'] = (df_station['rainfall'] > threshold).astype(int)
    df_station['event_id'] = (df_station['rain_event'].diff().fillna(0) == 1).cumsum()
    df_station.loc[df_station['rain_event'] == 0, 'event_id'] = None

    # Compute event summary
    events = (
        df_station.dropna(subset=['event_id'])
        .groupby('event_id')
        .agg(
            start_time=('datetime', 'min'),
            end_time=('datetime', 'max'),
            total_rainfall=('rainfall', 'sum'),
            duration_hours=('datetime', lambda x: (x.max() - x.min()).total_seconds() / 3600 + 1)
        )
        .reset_index(drop=True)
    )

    st.subheader("Identified Rainfall Events")
    st.dataframe(events)

    # Plot time series
    st.subheader("Rainfall Time Series")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_station['datetime'], df_station['rainfall'], label='Rainfall (mm)')
    ax.set_xlabel("Time")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title(f"Hourly Rainfall - {station}")
    ax.legend()
    st.pyplot(fig)

    # Boxplot of events
    st.subheader("Event Rainfall Distribution")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(events['total_rainfall'])
    ax2.set_ylabel("Total Rainfall (mm)")
    ax2.set_title("Distribution of Rainfall per Event")
    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to begin.")
