import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

# =========================
# STREAMLIT APP CONFIG
# =========================
st.set_page_config(page_title="Rainfall Analysis App", layout="wide")
st.title("GHMC Rainfall Analysis Dashboard")
st.write("Upload hourly rainfall dataset (CSV) with columns: 'S.No', 'AWS_ID', 'Date_&_Time', 'District', 'Mandal', 'Location', 'Circle', 'Latitude', 'Longitude', 'Hourly__Rainfall_(mm)', 'Day_Cumulative__Rainfall_(mm)'")

st.write("""
Processes GHMC hourly rainfall data to generate:
- **Daily rainfall summaries**
- **Rainfall event detection**
- **Rainy days statistics**
""")

# =========================
# FILE UPLOAD SECTION
# =========================
uploaded_file = st.file_uploader("Upload hourly rainfall CSV file", type=['csv'])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Read data
    df = pd.read_csv(uploaded_file)

    # =========================
    # PREPROCESS DATA
    # =========================
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
        c = 0
        max_c = 0
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
    st.subheader("Analysis Completed Successfully!")
    st.write("Below are the generated rainfall summaries:")

    tab1, tab2, tab3 = st.tabs([
        "Daily Summary", "Rain Events", "Rainy Days Stats"
    ])

    with tab1:
        st.dataframe(daily.head(20))
    with tab2:
        st.dataframe(events.head(20))
    with tab3:
        st.dataframe(rainy_days.head(20))

    # =========================
    # DOWNLOAD RESULTS
    # =========================
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    st.markdown("### Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(" Daily Summary CSV", convert_df(daily), "daily_rainfall_summary.csv", "text/csv")
    with col2:
        st.download_button("Rain Events CSV", convert_df(events), "rain_events_summary.csv", "text/csv")
    with col3:
        st.download_button(" Rainy Days CSV", convert_df(rainy_days), "rainy_days_summary.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to begin the analysis.")
