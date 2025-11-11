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
and generate rainfall summaries, rainfall event statistics, and threshold-based queries with plots.
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

        summary_option = st.radio(
            "Select summary type:",
            ["Daily Rainfall Summary", "Rain Events Summary"],
            horizontal=True
        )

        # Layout with 2 columns: input + plot side-by-side
        col1, col2 = st.columns([1.2, 1])

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
            plot_type = st.selectbox("Select plot type:", ["Bar", "Line", "Histogram"])

            if summary_option == "Daily Rainfall Summary" and 'filtered_daily' in locals():
                if plot_type == "Line":
                    fig = px.line(filtered_daily, x="Date", y="Daily_Rainfall", color="AWS_ID",
                                  title="Daily Rainfall Trend")
                elif plot_type == "Bar":
                    fig = px.bar(filtered_daily, x="AWS_ID", y="Daily_Rainfall",
                                 title="Daily Rainfall per Station")
                else:
                    fig = px.histogram(filtered_daily, x="Daily_Rainfall", nbins=20,
                                       title="Distribution of Daily Rainfall")
                st.plotly_chart(fig, use_container_width=True)

            elif summary_option == "Rain Events Summary" and 'filtered_events' in locals():
                if plot_type == "Line":
                    fig = px.line(filtered_events, x="Start", y="Total_Rain", color="AWS_ID",
                                  title="Event Rainfall Over Time")
                elif plot_type == "Bar":
                    fig = px.bar(filtered_events, x="AWS_ID", y="Total_Rain",
                                 title="Total Rain per Station")
                else:
                    fig = px.histogram(filtered_events, x="Total_Rain", nbins=20,
                                       title="Distribution of Event Rainfall")
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 2: CUSTOM QUERIES
    # =========================
    with tab2:
        st.header(" Threshold-Based Queries")

        col1, col2, col3 = st.columns(3)
        with col1:
            hr_thresh = st.number_input("Hourly rainfall (mm):", value=10.0)
        with col2:
            daily_thresh = st.number_input("Daily rainfall (mm):", value=50.0)
        with col3:
            duration_thresh = st.number_input("Event duration (hours):", value=5)

        if st.button("Run Queries"):
            filtered_hr = df[df["Hourly_Rain"] >= hr_thresh]
            high_daily = daily[daily["Daily_Rainfall"] >= daily_thresh]
            long_events = events[events["Duration_hrs"] >= duration_thresh]

            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                st.markdown("### Results")
                st.write(f" Hourly ≥ {hr_thresh} mm → {len(filtered_hr)} records")
                st.dataframe(filtered_hr.head(10))
                st.write(f" Daily ≥ {daily_thresh} mm → {len(high_daily)} days")
                st.dataframe(high_daily.head(10))
                st.write(f" Duration ≥ {duration_thresh} hrs → {len(long_events)} events")
                st.dataframe(long_events.head(10))

            with col_b:
                st.markdown("### Visualization")
                plot_sel = st.selectbox("Plot type:", ["Scatter", "Box", "Histogram"])
                if plot_sel == "Scatter":
                    fig = px.scatter(long_events, x="Duration_hrs", y="Average_Intensity",
                                     color="AWS_ID", size="Total_Rain",
                                     title="Duration vs Intensity of Events")
                elif plot_sel == "Box":
                    fig = px.box(high_daily, x="AWS_ID", y="Daily_Rainfall",
                                 title="Boxplot of Daily Rainfall by Station")
                else:
                    fig = px.histogram(filtered_hr, x="Hourly_Rain", nbins=30,
                                       title="Distribution of Hourly Rainfall")
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 3: VISUALIZATION
    # =========================
    with tab3:
        st.header(" Visualization Panel")

        st.markdown("Select visualization type:")
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
