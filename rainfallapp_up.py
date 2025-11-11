import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

# =========================
# STREAMLIT APP CONFIG
# =========================
st.set_page_config(page_title="Rainfall Analysis App", layout="wide")
st.title(" GHMC Rainfall Analysis Dashboard")

st.write("""
Upload hourly rainfall dataset (CSV) with columns:
'S.No', 'AWS_ID', 'Date_&_Time', 'District', 'Mandal', 'Location', 'Circle',
'Latitude', 'Longitude', 'Hourly__Rainfall_(mm)', 'Day_Cumulative__Rainfall_(mm)'.
""")

st.write("""
This app processes GHMC hourly rainfall data to generate:
- **Daily rainfall summaries**
- **Rainfall event detection**
- **Rainy days statistics**
- **Custom threshold-based and spatial queries**
""")

# =========================
# FILE UPLOAD SECTION
# =========================
uploaded_file = st.file_uploader(" Upload hourly rainfall CSV file", type=['csv'])

if uploaded_file is not None:
    st.success(" File uploaded successfully!")

    # =========================
    # READ & PREPROCESS DATA
    # =========================
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace(' ', '_')

    if 'Hourly__Rainfall_(mm)' in df.columns:
        df.rename(columns={'Hourly__Rainfall_(mm)': 'Hourly_Rain'}, inplace=True)
    if 'Day_Cumulative__Rainfall_(mm)' in df.columns:
        df.rename(columns={'Day_Cumulative__Rainfall_(mm)': 'Day_CumRain'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['Date_&_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['DateTime'])
    df['Date'] = df['DateTime'].dt.date
    df['Hour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'], errors='coerce').fillna(0)

    meta_cols = ['AWS_ID', 'District', 'Mandal', 'Location', 'Circle', 'Latitude', 'Longitude']

    # =========================
    # DAILY AGGREGATION
    # =========================
    daily = df.groupby(meta_cols + ['Date']).agg(
        Daily_Rainfall=('Hourly_Rain', 'sum'),
        Max_Hourly_Rain=('Hourly_Rain', 'max'),
        Min_Hourly_Rain=('Hourly_Rain', lambda x: x[x > 0].min() if any(x > 0) else 0),
        Hours_Rained=('Hourly_Rain', lambda x: (x > 0).sum())
    ).reset_index()

    daily['Daily_Intensity'] = daily.apply(
        lambda x: x['Daily_Rainfall'] / x['Hours_Rained'] if x['Hours_Rained'] > 0 else 0, axis=1)
    daily['RainFlag'] = (daily['Daily_Rainfall'] > 0).astype(int)

    # =========================
    # EVENT DETECTION
    # =========================
    df_sorted = df.sort_values(['AWS_ID', 'DateTime']).copy()
    df_sorted['RainFlag'] = (df_sorted['Hourly_Rain'] > 0).astype(int)
    df_sorted['EventStart'] = (df_sorted['RainFlag'].diff().fillna(0) == 1).astype(int)
    df_sorted['EventID'] = (df_sorted['EventStart'].cumsum() * df_sorted['RainFlag']).astype(int)

    events = df_sorted[df_sorted['EventID'] > 0].groupby(meta_cols + ['EventID']).agg(
        Start=('DateTime', 'min'),
        End=('DateTime', 'max'),
        Duration_hrs=('DateTime', lambda x: len(x)),
        Total_Rain=('Hourly_Rain', 'sum'),
        Max_Hourly=('Hourly_Rain', 'max')
    ).reset_index()
    events['Average_Intensity'] = events['Total_Rain'] / events['Duration_hrs']

    # =========================
    # RAINY DAYS STATISTICS
    # =========================
    rainy_days = daily[daily['Daily_Rainfall'] > 0].groupby(meta_cols).agg(
        Total_Rainy_Days=('Date', 'count'),
        Mean_Daily_Rain=('Daily_Rainfall', 'mean'),
        Max_Daily_Rain=('Daily_Rainfall', 'max'),
        Mean_Intensity=('Daily_Intensity', 'mean')
    ).reset_index()

    def longest_wet_spell(series):
        c = max_c = 0
        for val in series:
            if val == 1:
                c += 1
                max_c = max(max_c, c)
            else:
                c = 0
        return max_c

    wetspell = daily.groupby(meta_cols)['RainFlag'].apply(longest_wet_spell).reset_index(name='Longest_Wet_Spell_days')
    rainy_days = rainy_days.merge(wetspell, on=meta_cols, how='left')

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.subheader(" Analysis Results")

    # --- DAILY SUMMARY ---
    st.markdown("###  Daily Summary")
    nrows_daily = st.selectbox("Rows to display:", [10, 25, 50, "All"], key="daily_rows", index=0)
    nrows_daily = None if nrows_daily == "All" else int(nrows_daily)
    st.dataframe(daily.head(nrows_daily))

    # --- RAIN EVENTS ---
    st.markdown("###  Rain Events")
    nrows_events = st.selectbox("Rows to display:", [10, 25, 50, "All"], key="event_rows", index=0)
    nrows_events = None if nrows_events == "All" else int(nrows_events)
    st.dataframe(events.head(nrows_events))

    # --- RAINY DAYS STATS ---
    st.markdown("###  Rainy Days Statistics")
    nrows_rainy = st.selectbox("Rows to display:", [10, 25, 50, "All"], key="rainy_rows", index=0)
    nrows_rainy = None if nrows_rainy == "All" else int(nrows_rainy)
    st.dataframe(rainy_days.head(nrows_rainy))

    # =========================
    # CUSTOM QUERIES SECTION
    # =========================
    st.markdown("---")
    st.subheader("🔍 Custom Queries")

    with st.expander(" Threshold-based Filters", expanded=True):
        hr_thresh = st.number_input("Hourly rainfall threshold (mm)", value=10.0)
        daily_thresh = st.number_input("Daily rainfall threshold (mm)", value=50.0)
        max_rain_thresh = st.number_input("Max hourly rainfall threshold (mm)", value=40.0)
        long_event_thresh = st.number_input("Event duration threshold (hours)", value=5)

        filtered_hr = df[df["Hourly_Rain"] >= hr_thresh]
        st.write(f"**Hourly rainfall ≥ {hr_thresh} mm:** {len(filtered_hr)} records")
        st.dataframe(filtered_hr.head(10))

        high_rain_days = daily[daily["Daily_Rainfall"] >= daily_thresh]
        st.write(f"**Days with daily rainfall ≥ {daily_thresh} mm:** {len(high_rain_days)}")
        st.dataframe(high_rain_days.head(10))

        high_intensity_events = events[events["Max_Hourly"] >= max_rain_thresh]
        st.write(f"**Events with max hourly rainfall ≥ {max_rain_thresh} mm:** {len(high_intensity_events)}")
        st.dataframe(high_intensity_events.head(10))

        long_events = events[events["Duration_hrs"] >= long_event_thresh]
        st.write(f"**Events lasting ≥ {long_event_thresh} hours:** {len(long_events)}")
        st.dataframe(long_events.head(10))

    with st.expander(" Spatial / Temporal Filters"):
        station_sel = st.multiselect("Select Station(s):", sorted(df["AWS_ID"].unique()))
        mandal_sel = st.multiselect("Select Mandal(s):", sorted(df["Mandal"].unique()))
        month_sel = st.multiselect("Select Month(s):", sorted(df["Month"].unique()))

        filtered = df.copy()
        if station_sel:
            filtered = filtered[filtered["AWS_ID"].isin(station_sel)]
        if mandal_sel:
            filtered = filtered[filtered["Mandal"].isin(mandal_sel)]
        if month_sel:
            filtered = filtered[filtered["Month"].isin(month_sel)]

        st.write(f"Filtered records: {len(filtered)}")
        st.dataframe(filtered.head(10))

    with st.expander(" Additional Insights"):
        rain_counts = df[df['Hourly_Rain'] > 0].groupby('AWS_ID').size().reset_index(name='Rain_Hour_Count')
        st.write("**Total rainfall hours per station:**")
        st.dataframe(rain_counts.sort_values("Rain_Hour_Count", ascending=False).head(20))

        top_days = daily.sort_values("Daily_Rainfall", ascending=False).head(10)
        st.write("**Top 10 rainiest days across all stations:**")
        st.dataframe(top_days)

        longest_events = events.sort_values("Duration_hrs", ascending=False).head(10)
        st.write("**Top 10 longest rainfall events:**")
        st.dataframe(longest_events)

    # =========================
    # DOWNLOAD RESULTS
    # =========================
    st.markdown("---")
    st.subheader(" Download Results")
    col1, col2, col3 = st.columns(3)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    with col1:
        st.download_button(" Daily Summary", convert_df(daily), "daily_summary.csv", "text/csv")
    with col2:
        st.download_button(" Rain Events", convert_df(events), "rain_events.csv", "text/csv")
    with col3:
        st.download_button("Rainy Days Stats", convert_df(rainy_days), "rainy_days.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin the analysis.")
