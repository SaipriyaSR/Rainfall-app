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
Explore **hourly GHMC rainfall data** interactively.  
You can view **summary statistics**, run **threshold-based queries**, and visualize **spatial and temporal trends**.
""")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(" Upload hourly rainfall CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace(' ', '_')
    df.rename(columns={'Hourly__Rainfall_(mm)': 'Hourly_Rain',
                       'Day_Cumulative__Rainfall_(mm)': 'Day_CumRain'}, inplace=True, errors='ignore')

    df['DateTime'] = pd.to_datetime(df['Date_&_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df.dropna(subset=['DateTime'], inplace=True)
    df['Date'] = df['DateTime'].dt.date
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'], errors='coerce').fillna(0)

    meta_cols = ['AWS_ID', 'District', 'Mandal', 'Location', 'Circle', 'Latitude', 'Longitude']

    # =========================
    # DATA PROCESSING
    # =========================
    daily = df.groupby(meta_cols + ['Date']).agg(
        Daily_Rainfall=('Hourly_Rain', 'sum'),
        Max_Hourly_Rain=('Hourly_Rain', 'max'),
        Hours_Rained=('Hourly_Rain', lambda x: (x > 0).sum())
    ).reset_index()
    daily['Daily_Intensity'] = daily.apply(lambda x: x['Daily_Rainfall'] / x['Hours_Rained']
                                           if x['Hours_Rained'] > 0 else 0, axis=1)

    # Event segmentation
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
    # TAB STRUCTURE
    # =========================
    tab1, tab2, tab3 = st.tabs([" Data Summary", " Threshold Queries", " Visualization"])

    # ----------------------------------------------------------------------
    # TAB 1: DATA SUMMARY
    # ----------------------------------------------------------------------
    with tab1:
        st.header(" Rainfall Summary Statistics")

        with st.container():
            summary_option = st.radio("Select Summary Type:",
                                      ["Daily Rainfall Summary", "Rain Events Summary"],
                                      horizontal=True)

        if summary_option == "Daily Rainfall Summary":
            threshold = st.number_input("Enter daily rainfall threshold (mm):", value=50.0)
            if st.button("Show Daily Summary"):
                filtered_daily = daily[daily['Daily_Rainfall'] >= threshold]
                st.success(f"{len(filtered_daily)} days had rainfall ≥ {threshold} mm")
                st.dataframe(filtered_daily)

                # ---- Visualizations ----
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.box(filtered_daily, x='AWS_ID', y='Daily_Intensity',
                                 title="Boxplot of Daily Intensities by Station")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.scatter(filtered_daily, x='Date', y='Daily_Rainfall',
                                     color='AWS_ID', title="Temporal Distribution of Daily Rainfall")
                    st.plotly_chart(fig, use_container_width=True)

        else:
            event_thresh = st.number_input("Enter event rainfall threshold (mm):", value=30.0)
            if st.button("Show Event Summary"):
                filtered_events = events[events['Total_Rain'] >= event_thresh]
                st.success(f"{len(filtered_events)} rainfall events ≥ {event_thresh} mm")
                st.dataframe(filtered_events)

                # ---- Visualizations ----
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(filtered_events, x='Duration_hrs', y='Average_Intensity',
                                     color='AWS_ID', size='Total_Rain',
                                     title="Duration vs Intensity of Rainfall Events")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(filtered_events, x='Month', y='Average_Intensity',
                                 title="Monthly Distribution of Event Intensities")
                    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 2: THRESHOLD QUERIES (Independent Execution)
    # ----------------------------------------------------------------------
    with tab2:
        st.header(" Run Custom Threshold Queries")

        st.markdown("Run each threshold-based query independently and view results below:")

        # Hourly Threshold
        with st.expander("Hourly Rainfall Threshold"):
            hr_thresh = st.number_input("Hourly rainfall (mm):", value=10.0, key="hr_t")
            if st.button("Run Hourly Query"):
                res = df[df["Hourly_Rain"] >= hr_thresh]
                st.write(f"{len(res)} records with hourly rainfall ≥ {hr_thresh} mm")
                st.dataframe(res.head(10))
                fig = px.histogram(res, x="Hourly_Rain", nbins=30,
                                   title=f"Distribution of Hourly Rainfall ≥ {hr_thresh} mm")
                st.plotly_chart(fig, use_container_width=True)

        # Daily Threshold
        with st.expander("Daily Rainfall Threshold"):
            daily_thresh = st.number_input("Daily rainfall (mm):", value=50.0, key="daily_t")
            if st.button("Run Daily Query"):
                res = daily[daily["Daily_Rainfall"] >= daily_thresh]
                st.write(f"{len(res)} days with rainfall ≥ {daily_thresh} mm")
                st.dataframe(res.head(10))
                fig = px.box(res, x="AWS_ID", y="Daily_Rainfall",
                             title=f"Daily Rainfall Distribution ≥ {daily_thresh} mm")
                st.plotly_chart(fig, use_container_width=True)

        # Duration Threshold
        with st.expander("Rainfall Event Duration Threshold"):
            duration_thresh = st.number_input("Event duration (hours):", value=5, key="dur_t")
            if st.button("Run Duration Query"):
                res = events[events["Duration_hrs"] >= duration_thresh]
                st.write(f"{len(res)} events lasting ≥ {duration_thresh} hours")
                st.dataframe(res.head(10))
                fig = px.scatter(res, x="Duration_hrs", y="Average_Intensity",
                                 color="AWS_ID", size="Total_Rain",
                                 title="Event Duration vs Intensity")
                st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 3: VISUALIZATION
    # ----------------------------------------------------------------------
    with tab3:
        st.header(" Visualization and Spatial Insights")

        plot_option = st.selectbox("Select Visualization Type:",
                                   ["Spatial Distribution of Rainfall",
                                    "Station-wise Rainfall Trends",
                                    "Monthly Intensity Distribution"])

        if plot_option == "Spatial Distribution of Rainfall":
            spatial_avg = daily.groupby(["AWS_ID", "Latitude", "Longitude"])["Daily_Rainfall"].mean().reset_index()
            fig_map = px.scatter_mapbox(spatial_avg, lat="Latitude", lon="Longitude",
                                        color="Daily_Rainfall", size="Daily_Rainfall",
                                        hover_name="AWS_ID",
                                        mapbox_style="open-street-map",
                                        color_continuous_scale="Blues", zoom=9,
                                        title="Average Daily Rainfall Across AWS Stations")
            st.plotly_chart(fig_map, use_container_width=True)

        elif plot_option == "Station-wise Rainfall Trends":
            station_choice = st.selectbox("Select Station:", sorted(daily["AWS_ID"].unique()))
            df_station = daily[daily["AWS_ID"] == station_choice]
            fig = px.line(df_station, x="Date", y="Daily_Rainfall",
                          title=f"Daily Rainfall Trend - {station_choice}",
                          labels={'Daily_Rainfall': 'Rainfall (mm)'})
            st.plotly_chart(fig, use_container_width=True)

        else:
            monthly_intensity = daily.groupby(['Month', 'AWS_ID'])['Daily_Intensity'].mean().reset_index()
            fig = px.box(monthly_intensity, x="Month", y="Daily_Intensity", color="AWS_ID",
                         title="Monthly Rainfall Intensity Distribution Across Stations")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info(" Please upload a CSV file to start the analysis.")
