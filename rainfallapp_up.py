import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="GHMC Rainfall Dashboard", layout="wide")
st.title(" GHMC Rainfall Analysis Dashboard")

st.markdown("""
This interactive tool allows you to analyze **hourly GHMC rainfall data**  
and generate rainfall summaries, rainfall event statistics, and threshold-based queries.
""")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(" Upload hourly rainfall CSV file", type=['csv'])

if uploaded_file is not None:
    st.success(" File uploaded successfully!")

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace(' ', '_')

    # Rename key columns
    df.rename(columns={
        'Hourly__Rainfall_(mm)': 'Hourly_Rain',
        'Day_Cumulative__Rainfall_(mm)': 'Day_CumRain'
    }, inplace=True, errors='ignore')

    # Parse datetime
    df['DateTime'] = pd.to_datetime(df['Date_&_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df.dropna(subset=['DateTime'], inplace=True)

    df['Date'] = df['DateTime'].dt.date
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'], errors='coerce').fillna(0)

    meta_cols = ['AWS_ID', 'District', 'Mandal', 'Location', 'Circle', 'Latitude', 'Longitude']

    # Daily Aggregation
    daily = df.groupby(meta_cols + ['Date']).agg(
        Daily_Rainfall=('Hourly_Rain', 'sum'),
        Max_Hourly_Rain=('Hourly_Rain', 'max'),
        Hours_Rained=('Hourly_Rain', lambda x: (x > 0).sum())
    ).reset_index()
    daily['Daily_Intensity'] = daily.apply(lambda x: x['Daily_Rainfall'] / x['Hours_Rained']
                                           if x['Hours_Rained'] > 0 else 0, axis=1)

    # Rain events
    df_sorted = df.sort_values(['AWS_ID', 'DateTime']).copy()
    df_sorted['RainFlag'] = (df_sorted['Hourly_Rain'] > 0).astype(int)
    df_sorted['EventStart'] = (df_sorted['RainFlag'].diff().fillna(0) == 1).astype(int)
    df_sorted['EventID'] = (df_sorted['EventStart'].cumsum() * df_sorted['RainFlag']).astype(int)

    events = df_sorted[df_sorted['EventID'] > 0].groupby(meta_cols + ['EventID']).agg(
        Start=('DateTime', 'min'),
        End=('DateTime', 'max'),
        Duration_hrs=('DateTime', 'count'),
        Total_Rain=('Hourly_Rain', 'sum'),
        Max_Hourly=('Hourly_Rain', 'max')
    ).reset_index()
    events['Average_Intensity'] = events['Total_Rain'] / events['Duration_hrs']

    # =========================
    # TABS LAYOUT
    # =========================
    tab1, tab2, tab3 = st.tabs([" Data Summary", " Custom Queries", " Visualization"])

    # =========================
    # TAB 1: DATA SUMMARY
    # =========================
    with tab1:
        st.header(" Rainfall Summary")

        st.markdown("Select type of summary to view:")

        summary_option = st.radio(
            "Choose summary type:",
            ["Daily Rainfall Summary", "Rain Events Summary"],
            horizontal=True
        )

        if summary_option == "Daily Rainfall Summary":
            daily_threshold = st.number_input("Enter daily rainfall threshold (mm):", value=50.0)
            filtered_daily = daily[daily['Daily_Rainfall'] >= daily_threshold]
            st.write(f"**Days with rainfall ≥ {daily_threshold} mm:** {len(filtered_daily)}")
            st.dataframe(filtered_daily)
        else:
            event_thresh = st.number_input("Enter event total rainfall threshold (mm):", value=30.0)
            filtered_events = events[events['Total_Rain'] >= event_thresh]
            st.write(f"**Events with total rainfall ≥ {event_thresh} mm:** {len(filtered_events)}")
            st.dataframe(filtered_events)

    # =========================
    # TAB 2: CUSTOM QUERIES
    # =========================
    with tab2:
        st.header(" Threshold-Based Queries")

        st.markdown("Enter rainfall thresholds to explore significant events:")
        col1, col2, col3 = st.columns(3)

        with col1:
            hr_thresh = st.number_input("Hourly rainfall (mm):", value=10.0)
        with col2:
            daily_thresh = st.number_input("Daily rainfall (mm):", value=50.0)
        with col3:
            duration_thresh = st.number_input("Event duration (hours):", value=5)

        st.markdown("---")

        if st.button("Run Queries"):
            filtered_hr = df[df["Hourly_Rain"] >= hr_thresh]
            high_daily = daily[daily["Daily_Rainfall"] >= daily_thresh]
            long_events = events[events["Duration_hrs"] >= duration_thresh]

            st.subheader("Results:")
            st.write(f" Hourly ≥ {hr_thresh} mm → {len(filtered_hr)} records")
            st.dataframe(filtered_hr.head(10))
            st.write(f" Daily ≥ {daily_thresh} mm → {len(high_daily)} days")
            st.dataframe(high_daily.head(10))
            st.write(f" Duration ≥ {duration_thresh} hrs → {len(long_events)} events")
            st.dataframe(long_events.head(10))

    # =========================
    # TAB 3: VISUALIZATION
    # =========================
    with tab3:
        st.header(" Rainfall Visualizations")

        st.markdown("Select plot type:")
        plot_option = st.selectbox(
            "Choose visualization:",
            ["Daily Rainfall Trend", "Event Intensity vs Duration", "Spatial Distribution"]
        )

        station_choice = st.selectbox("Select AWS station:", sorted(daily["AWS_ID"].unique()))

        if plot_option == "Daily Rainfall Trend":
            df_station = daily[daily["AWS_ID"] == station_choice]
            fig = px.line(df_station, x="Date", y="Daily_Rainfall",
                          title=f"Daily Rainfall Trend - {station_choice}",
                          labels={'Daily_Rainfall': 'Rainfall (mm)'})
            st.plotly_chart(fig, use_container_width=True)

        elif plot_option == "Event Intensity vs Duration":
            fig = px.scatter(events[events["AWS_ID"] == station_choice],
                             x="Duration_hrs", y="Average_Intensity",
                             color="Total_Rain", size="Total_Rain",
                             hover_data=["Start", "End"],
                             title=f"Event Duration vs Intensity - {station_choice}",
                             labels={"Duration_hrs": "Duration (hrs)", "Average_Intensity": "Avg Intensity (mm/hr)"})
            st.plotly_chart(fig, use_container_width=True)

        else:
            spatial_avg = daily.groupby(["AWS_ID", "Latitude", "Longitude"])["Daily_Rainfall"].mean().reset_index()
            fig_map = px.scatter_mapbox(
                spatial_avg, lat="Latitude", lon="Longitude",
                color="Daily_Rainfall", size="Daily_Rainfall",
                hover_name="AWS_ID", color_continuous_scale="Blues",
                mapbox_style="open-street-map", zoom=9,
                title="Average Daily Rainfall Across AWS Stations"
            )
            st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info("Please upload a CSV file to start the analysis.")
