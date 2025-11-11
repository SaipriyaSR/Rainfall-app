import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="GHMC Rainfall Dashboard", layout="wide")
st.title(" GHMC Rainfall Analysis Dashboard")

st.markdown("""
This interactive dashboard allows you to explore **hourly GHMC rainfall data**  
and generate rainfall summaries, rainfall event statistics, and threshold-based queries with insightful visualizations.
""")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(" Upload hourly rainfall CSV file", type=['csv'])

if uploaded_file is not None:
    st.success(" File uploaded successfully!")

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace(' ', '_')

    # Rename and preprocess
    df.rename(columns={
        'Hourly__Rainfall_(mm)': 'Hourly_Rain',
        'Day_Cumulative__Rainfall_(mm)': 'Day_CumRain'
    }, inplace=True, errors='ignore')

    df['DateTime'] = pd.to_datetime(df['Date_&_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df.dropna(subset=['DateTime'], inplace=True)
    df['Date'] = df['DateTime'].dt.date
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'], errors='coerce').fillna(0)

    meta_cols = ['AWS_ID', 'District', 'Mandal', 'Location', 'Circle', 'Latitude', 'Longitude']

    # =========================
    # AGGREGATED DATASETS
    # =========================
    daily = df.groupby(meta_cols + ['Year', 'Month','Date']).agg(
        Daily_Rainfall=('Hourly_Rain', 'sum'),
        Max_Hourly_Rain=('Hourly_Rain', 'max'),
        Hours_Rained=('Hourly_Rain', lambda x: (x > 0).sum())
    ).reset_index()
    daily['Daily_Intensity'] = daily.apply(lambda x: x['Daily_Rainfall'] / x['Hours_Rained']
                                           if x['Hours_Rained'] > 0 else 0, axis=1)

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
    # MAIN TABS
    # =========================
    tab1, tab2, tab3 = st.tabs([" Data Summary", "Custom Queries", " Visualization"])

    # =========================
    # TAB 1 - DATA SUMMARY
    # =========================
    with tab1:
        st.subheader(" Rainfall Summary")
        summary_option = st.radio("Select summary type:", ["Daily Rainfall Summary", "Rain Events Summary"], horizontal=True)

        col1, col2 = st.columns([1.3, 1])

        with col1:
            if summary_option == "Daily Rainfall Summary":
                daily_threshold = st.number_input("Enter daily rainfall threshold (mm):", value=50.0)
                if st.button("Show Daily Summary"):
                    filtered_daily = daily[daily['Daily_Rainfall'] >= daily_threshold]
                    st.success(f"Days with rainfall ≥ {daily_threshold} mm: {len(filtered_daily)}")
                    st.dataframe(filtered_daily)

            else:
                event_thresh = st.number_input("Enter event total rainfall threshold (mm):", value=30.0)
                if st.button("Show Event Summary"):
                    filtered_events = events[events['Total_Rain'] >= event_thresh]
                    st.success(f"Events with total rainfall ≥ {event_thresh} mm: {len(filtered_events)}")
                    st.dataframe(filtered_events)

        with col2:
            st.markdown("####  Quick Visualization")
            plot_type = st.selectbox("Select plot type:", ["Box", "Bar", "Line", "Spatial Map"])

            if summary_option == "Daily Rainfall Summary" and 'filtered_daily' in locals():
                if plot_type == "Box":
                    fig = px.box(filtered_daily, x="Month", y="Daily_Rainfall", color="AWS_ID",
                                 title="Monthly Rainfall Distribution")
                elif plot_type == "Bar":
                    fig = px.bar(filtered_daily, x="AWS_ID", y="Daily_Rainfall", color="Month",
                                 title="Rainfall by Station and Month")
                elif plot_type == "Line":
                    fig = px.line(filtered_daily, x="Date", y="Daily_Rainfall", color="AWS_ID",
                                  title="Daily Rainfall Trends")
                else:
                    spatial_avg = filtered_daily.groupby(["AWS_ID", "Latitude", "Longitude"])["Daily_Rainfall"].mean().reset_index()
                    fig = px.scatter_mapbox(spatial_avg, lat="Latitude", lon="Longitude", color="Daily_Rainfall",
                                            size="Daily_Rainfall", hover_name="AWS_ID", mapbox_style="open-street-map",
                                            color_continuous_scale="Blues", title="Spatial Distribution of Daily Rainfall")
                st.plotly_chart(fig, use_container_width=True)

            elif summary_option == "Rain Events Summary" and 'filtered_events' in locals():
                if plot_type == "Box":
                    fig = px.box(filtered_events, x="AWS_ID", y="Average_Intensity",
                                 title="Event Intensity Distribution Across Stations")
                elif plot_type == "Bar":
                    fig = px.bar(filtered_events, x="AWS_ID", y="Total_Rain",
                                 title="Total Event Rainfall by Station")
                elif plot_type == "Line":
                    fig = px.line(filtered_events, x="Start", y="Total_Rain", color="AWS_ID",
                                  title="Temporal Evolution of Events")
                else:
                    spatial_ev = filtered_events.groupby(["AWS_ID", "Latitude", "Longitude"])["Total_Rain"].mean().reset_index()
                    fig = px.scatter_mapbox(spatial_ev, lat="Latitude", lon="Longitude", color="Total_Rain",
                                            size="Total_Rain", hover_name="AWS_ID", mapbox_style="open-street-map",
                                            color_continuous_scale="Blues", title="Spatial Distribution of Rain Events")
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 2 - CUSTOM QUERIES
    # =========================
    with tab2:
        st.subheader(" Threshold-Based Queries")

        # Independent query sections
        with st.expander(" Hourly Rainfall Threshold Query"):
            hr_thresh = st.number_input("Enter hourly rainfall threshold (mm):", value=10.0, key="hourly_q")
            if st.button("Run Hourly Query"):
                filtered_hr = df[df["Hourly_Rain"] >= hr_thresh]
                st.write(f"Records ≥ {hr_thresh} mm/hour: {len(filtered_hr)}")
                st.dataframe(filtered_hr.head(10))
                fig = px.histogram(filtered_hr, x="Hourly_Rain", nbins=30, title="Distribution of Hourly Rainfall")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander(" Daily Rainfall Threshold Query"):
            daily_thresh = st.number_input("Enter daily rainfall threshold (mm):", value=50.0, key="daily_q")
            if st.button("Run Daily Query"):
                high_daily = daily[daily["Daily_Rainfall"] >= daily_thresh]
                st.write(f"Days ≥ {daily_thresh} mm/day: {len(high_daily)}")
                st.dataframe(high_daily.head(10))
                fig = px.box(high_daily, x="AWS_ID", y="Daily_Rainfall", color="AWS_ID",
                             title="Boxplot of Daily Rainfall Across Stations")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander(" Event Duration Query"):
            duration_thresh = st.number_input("Enter event duration threshold (hours):", value=5, key="event_q")
            if st.button("Run Event Duration Query"):
                long_events = events[events["Duration_hrs"] >= duration_thresh]
                st.write(f"Events ≥ {duration_thresh} hours: {len(long_events)}")
                st.dataframe(long_events.head(10))
                fig = px.scatter(long_events, x="Duration_hrs", y="Average_Intensity",
                                 color="AWS_ID", size="Total_Rain", hover_data=["Start", "End"],
                                 title="Duration vs Intensity of Events")
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 3 - VISUALIZATION PANEL
    # =========================
    with tab3:
        st.subheader(" Advanced Visualizations")

        vis_option = st.selectbox("Select visualization type:", [
            "Daily Rainfall Trend (Station-wise)",
            "Monthly Intensity Boxplot",
            "Event Duration vs Total Rain",
            "Spatial Distribution (Average Rainfall)"
        ])

        if vis_option == "Daily Rainfall Trend (Station-wise)":
            station_choice = st.selectbox("Select AWS station:", sorted(daily["AWS_ID"].unique()))
            df_station = daily[daily["AWS_ID"] == station_choice]
            fig = px.line(df_station, x="Date", y="Daily_Rainfall", title=f"Daily Rainfall - {station_choice}")
            st.plotly_chart(fig, use_container_width=True)

        elif vis_option == "Monthly Intensity Boxplot":
            fig = px.box(daily, x="Month", y="Daily_Intensity", color="AWS_ID",
                         title="Monthly Distribution of Daily Intensity")
            st.plotly_chart(fig, use_container_width=True)

        elif vis_option == "Event Duration vs Total Rain":
            fig = px.scatter(events, x="Duration_hrs", y="Total_Rain", color="AWS_ID",
                             size="Average_Intensity", hover_data=["Start", "End"],
                             title="Event Duration vs Total Rainfall")
            st.plotly_chart(fig, use_container_width=True)

        else:
            spatial_avg = daily.groupby(["AWS_ID", "Latitude", "Longitude"])["Daily_Rainfall"].mean().reset_index()
            fig = px.scatter_mapbox(spatial_avg, lat="Latitude", lon="Longitude",
                                    color="Daily_Rainfall", size="Daily_Rainfall",
                                    hover_name="AWS_ID", color_continuous_scale="Blues",
                                    mapbox_style="open-street-map", zoom=9,
                                    title="Spatial Distribution of Average Daily Rainfall")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info(" Please upload a CSV file to start the analysis.")
        # =========================
        # ADDITIONAL STATION ANALYSIS
        # =========================
        st.markdown("---")
        st.subheader(" Station-wise Rainfall Frequency and Intensity Analysis")

        st.write("""
        Explore how frequently rainfall occurs at each station — from sub-daily to monthly or seasonal scales — 
        along with rainfall intensity distribution and maximum intensity per station.
        """)

        station_select = st.selectbox("Select a Station for Detailed Analysis", sorted(df["AWS_ID"].unique()), key="station_analysis")

        df_station = df[df["AWS_ID"] == station_select]
        daily_station = daily[daily["AWS_ID"] == station_select]
        event_station = events[events["AWS_ID"] == station_select]

        #  How many times it rained in a day (hourly count >0)
        daily_rain_counts = df_station.groupby('Date')['Hourly_Rain'].apply(lambda x: (x > 0).sum()).reset_index(name='Hours_Rained')
        st.write("###  Number of hours it rained per day")
        st.dataframe(daily_rain_counts.head(10))
        fig = px.bar(daily_rain_counts, x='Date', y='Hours_Rained', title=f'Number of Rainy Hours per Day - {station_select}')
        st.plotly_chart(fig, use_container_width=True)

        #  How many days it rained in a month & season
        daily_station['RainDay'] = (daily_station['Daily_Rainfall'] > 0).astype(int)
        monthly_rain_days = daily_station.groupby('Month')['RainDay'].sum().reset_index(name='Rainy_Days')
        st.write("###  Number of Rainy Days per Month")
        st.dataframe(monthly_rain_days)
        fig = px.bar(monthly_rain_days, x='Month', y='Rainy_Days', title=f'Rainy Days per Month - {station_select}')
        st.plotly_chart(fig, use_container_width=True)

        # Define seasons manually
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Pre-Monsoon'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            else:
                return 'Post-Monsoon'

        daily_station['Season'] = daily_station['Month'].apply(get_season)
        seasonal_rain_days = daily_station.groupby('Season')['RainDay'].sum().reset_index(name='Rainy_Days')
        st.write("###  Rainy Days by Season")
        st.dataframe(seasonal_rain_days)
        fig = px.bar(seasonal_rain_days, x='Season', y='Rainy_Days', title=f'Rainy Days by Season - {station_select}')
        st.plotly_chart(fig, use_container_width=True)

        #  How many times it rained including sub-daily
        st.write("###  Sub-daily Rainfall Events")
        subdaily_rain_events = df_station[df_station["Hourly_Rain"] > 0].shape[0]
        st.success(f"Total hourly rain records (>0 mm/hr): {subdaily_rain_events}")

        #  Events with minimum rainfall intensity > 5 mm/hr
        st.write("###  Rainfall Events with Intensity > 5 mm/hr")
        intense_events = event_station[event_station['Average_Intensity'] > 5]
        st.write(f"Number of events: {len(intense_events)}")
        st.dataframe(intense_events.head(10))
        fig = px.histogram(intense_events, x="Average_Intensity", nbins=20, color="AWS_ID",
                           title=f"Distribution of High-Intensity Events (>5 mm/hr) - {station_select}")
        st.plotly_chart(fig, use_container_width=True)

        #  Maximum rainfall intensity & boxplots
        st.write("###  Maximum and Distribution of Rainfall Intensity")
        max_intensity = event_station['Average_Intensity'].max()
        st.success(f"Maximum rainfall intensity recorded: {max_intensity:.2f} mm/hr")

        fig = px.box(event_station, y="Average_Intensity", points="all", color_discrete_sequence=["#0072B2"],
                     title=f"Boxplot of Rainfall Intensities - {station_select}")
        st.plotly_chart(fig, use_container_width=True)

        monthly_intensity = event_station.copy()
        monthly_intensity['Month'] = monthly_intensity['Start'].dt.month
        fig = px.box(monthly_intensity, x="Month", y="Average_Intensity", color_discrete_sequence=["#009E73"],
                     title=f"Monthly Distribution of Event Intensities - {station_select}")
        st.plotly_chart(fig, use_container_width=True)
