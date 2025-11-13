import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="GHMC Rainfall Dashboard", layout="wide")
#st.title(" Rainfall Analysis Tool for GHMC")

# ---------- HEADER ----------
st.markdown("""
    <div style='background: linear-gradient(90deg, #002b5c 0%, #00509e 100%);
                padding: 18px; border-radius: 10px; text-align: center; color: white;'>
        <h1 style='margin-bottom: 5px;'> Rainfall Analysis Tool for GHMC</h1>
        <h5 style='margin-top: 0;'>Explore Hourly, Daily, and Event-Based Rainfall Patterns</h5>
    </div>
""", unsafe_allow_html=True)

# ---------- GLOBAL STYLES ----------
st.markdown("""
    <style>
    /* Increase sidebar width */
    [data-testid="stSidebar"] {
        min-width: 400px;  /* Default is ~250px, increase as needed */
        max-width: 400px;
    }


    /* Tabs container*/
    .stTabs [data-baseweb="tab-list"] {
        gap: 25px;
        background-color: #f4f6f8;
        padding: 10px;
        border-radius: 8px;
    }
    /* Individual Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 28px !important;  /* Increased font size */
        font-weight: 700 !important; /* Bolder font */
        color: #002b5c !important;
        padding: 15px 25px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease-in-out;
    }

    /* Hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e7ef !important;
        color: #004080 !important;
    }

    /* Active Tab */
    .stTabs [aria-selected="true"] {
        background-color: #002b5c !important;
        color: white !important;
        border-radius: 8px !important;
    }
    /* DataFrame scrollbar hidden */
    div[data-testid="stDataFrameResizable"] div[role="presentation"]::-webkit-scrollbar {
        display: none;
    }
    /* Enable vertical scrolling for long tab content */
    .stTabs [data-baseweb="tab-panel"] {
        max-height: 80vh;
        overflow-y: auto;
        padding-right: 10px;
    }

    /* Footer styling */
    .footer {
    text-align: center;
    padding: 10px;
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #f9f9f9;
    border-top: 1px solid #ddd;
    font-size: 14px;
    color: #444;
    }

    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
#st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/18/GHMC_logo.png/200px-GHMC_logo.png", width=120)
st.sidebar.header("About this App", width=200)
st.sidebar.info("""
This dashboard enables **interactive exploration** of rainfall data for GHMC stations.

You can:
- View hourly/daily rainfall summaries  
- Identify rainfall events  
- Run custom threshold queries  
- Analyze data station-wise  

**Upload a CSV file** to begin.
                
""")

uploaded_file = st.sidebar.file_uploader("📂 Upload hourly rainfall CSV file with columns: 'AWS_ID', 'Date_&_Time', 'Latitude', 'Longitude', 'Hourly__Rainfall_(mm)'", type=['csv'])

# =========================
# MAIN BODY
# =========================
if uploaded_file is not None:
    st.sidebar.success(" File uploaded successfully!")

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace(' ', '_')

    # ---------- Preview Section ----------
    with st.expander(" Data Preview and Station Map"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(" Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

        with col2:
            st.subheader(" AWS Station Locations")
            if {'Latitude', 'Longitude'}.issubset(df.columns):
                stations = df[['AWS_ID', 'Latitude', 'Longitude']].drop_duplicates()
                fig = px.scatter_mapbox(stations, lat="Latitude", lon="Longitude",
                                        hover_name="AWS_ID", zoom=9, mapbox_style="open-street-map",
                                        title="AWS Station Locations")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(" Latitude/Longitude columns not found in uploaded file.")

    # ---------- Preprocessing ----------
    df.rename(columns={'Hourly__Rainfall_(mm)': 'Hourly_Rain'}, inplace=True, errors='ignore')
    df['DateTime'] = pd.to_datetime(df['Date_&_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df.dropna(subset=['DateTime'], inplace=True)
    df['Date'] = df['DateTime'].dt.date
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'], errors='coerce').fillna(0)

    meta_cols = ['AWS_ID', 'District', 'Mandal', 'Location', 'Circle', 'Latitude', 'Longitude']

    # ---------- Aggregated Datasets ----------
    daily = df.groupby(meta_cols + ['Year', 'Month', 'Date']).agg(
        Daily_Rainfall=('Hourly_Rain', 'sum'),
        Max_Hourly_Rain=('Hourly_Rain', 'max'),
        Hours_Rained=('Hourly_Rain', lambda x: (x > 0).sum())
    ).reset_index()

    daily['Daily_Intensity'] = daily.apply(
        lambda x: x['Daily_Rainfall'] / x['Hours_Rained'] if x['Hours_Rained'] > 0 else 0, axis=1
    )

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

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs([
        " **Data Summary**",
        " **Custom Queries**",
        " **Visualization**",
        " **Station Analysis**"
    ])

    # =========================
    # TAB 1 - DATA SUMMARY
    # =========================
    with tab1:
        st.subheader(" Rainfall Summary")
        st.info("View daily or event-level rainfall summaries with customizable thresholds.")
        summary_option = st.radio("Select summary type:", ["Daily Rainfall Summary", "Rain Events Summary"], horizontal=True)

        col1, col2 = st.columns([1.3, 1])

    # ---- LEFT COLUMN: Data Filtering ----
    with col1:
        if summary_option == "Daily Rainfall Summary":
            daily_threshold = st.number_input("Enter daily rainfall threshold (mm):", value=0.0)

            # Store results persistently in session_state
            if st.button("Show Daily Summary"):
                st.session_state['filtered_daily'] = daily[daily['Daily_Rainfall'] >= daily_threshold]
                st.session_state['daily_threshold'] = daily_threshold

            # Display stored results if available
            if 'filtered_daily' in st.session_state:
                st.success(f"Days with rainfall ≥ {st.session_state['daily_threshold']} mm: "
                           f"{len(st.session_state['filtered_daily'])}")
                st.dataframe(st.session_state['filtered_daily'])

        else:
            event_thresh = st.number_input("Enter event total rainfall threshold (mm):", value=0.0)

            if st.button("Show Event Summary"):
                st.session_state['filtered_events'] = events[events['Total_Rain'] >= event_thresh]
                st.session_state['event_thresh'] = event_thresh

            if 'filtered_events' in st.session_state:
                st.success(f"Events with total rainfall ≥ {st.session_state['event_thresh']} mm: "
                           f"{len(st.session_state['filtered_events'])}")
                st.dataframe(st.session_state['filtered_events'])

    # ---- RIGHT COLUMN: Reactive Visualization ----
    with col2:
        st.markdown("#### Quick Visualization")
        plot_type = st.selectbox("Select plot type:", ["Box", "Bar", "Line", "Spatial Map"])

        # For Daily Rainfall Summary
        if summary_option == "Daily Rainfall Summary" and 'filtered_daily' in st.session_state:
            filtered_daily = st.session_state['filtered_daily']

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

        # For Rain Events Summary
        elif summary_option == "Rain Events Summary" and 'filtered_events' in st.session_state:
            filtered_events = st.session_state['filtered_events']

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

        # --- Hourly Rainfall Query ---
        with st.expander("Hourly Rainfall Threshold Query"):
            hr_thresh = st.number_input("Enter hourly rainfall threshold (mm):", value=10.0, key="hourly_q")
            if st.button("Run Hourly Query"):
                filtered_hr = df[df["Hourly_Rain"] >= hr_thresh]
                st.write(f"Records ≥ {hr_thresh} mm/hour: {len(filtered_hr)}")

                col1, col2 = st.columns([1.2, 1.8])
                with col1:
                    st.dataframe(filtered_hr, use_container_width=True)
                with col2:
                    fig = px.histogram(filtered_hr, x="Hourly_Rain", nbins=30, color="AWS_ID",
                                    title="Distribution of Hourly Rainfall ≥ Threshold")
                    st.plotly_chart(fig, use_container_width=True)

        # --- Daily Rainfall Query ---
        with st.expander("Daily Rainfall Threshold Query"):
            daily_thresh = st.number_input("Enter daily rainfall threshold (mm):", value=50.0, key="daily_q")
            if st.button("Run Daily Query"):
                high_daily = daily[daily["Daily_Rainfall"] >= daily_thresh]
                st.write(f"Days ≥ {daily_thresh} mm/day: {len(high_daily)}")

                col3, col4 = st.columns([1.2, 1.8])
                with col3:
                    st.dataframe(high_daily, use_container_width=True)
                with col4:
                    fig = px.box(high_daily, x="AWS_ID", y="Daily_Rainfall", color="AWS_ID",
                                title="Boxplot of Daily Rainfall Across Stations (≥ Threshold)")
                    st.plotly_chart(fig, use_container_width=True)

        # --- Event Duration Query ---
        with st.expander("Event Duration Query"):
            duration_thresh = st.number_input("Enter event duration threshold (hours):", value=5, key="event_q")
            if st.button("Run Event Duration Query"):
                long_events = events[events["Duration_hrs"] >= duration_thresh]
                st.write(f"Events ≥ {duration_thresh} hours: {len(long_events)}")

                col5, col6 = st.columns([1.2, 1.8])
                with col5:
                    st.dataframe(long_events, use_container_width=True)
                with col6:
                    fig = px.scatter(long_events, x="Duration_hrs", y="Average_Intensity",
                                    color="AWS_ID", size="Total_Rain", hover_data=["Start", "End"],
                                    title="Duration vs Intensity of Events (≥ Threshold)")
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

    # =========================
    # TAB 4 - STATION ANALYSIS
    # =========================
    with tab4:
        st.subheader("Station-wise Rainfall Frequency and Intensity Analysis")

        # Layout: selection + menu on left, results on right
        left_col, right_col = st.columns([0.3, 1.7])

        with left_col:
            station_select = st.selectbox("Select a Station", sorted(df["AWS_ID"].unique()))
            st.markdown("#### Station Metadata")
            meta_info = df[df["AWS_ID"] == station_select][['District', 'Mandal', 'Location', 'Circle']].drop_duplicates()
            st.dataframe(meta_info, hide_index=True, use_container_width=True)

            # --- Analysis menu ---
            st.markdown("##### Select Analysis Type")
            analysis_choice = st.radio(
                "Choose analysis to display:",
                [
                    "Rainy Hours per Day",
                    "Rainy Days per Month and Season",
                    "High-Intensity and Maximum Rainfall Events",
                    "Monthly Distribution of Event Intensities"
                ],
                index=0
            )

        with right_col:
            df_station = df[df["AWS_ID"] == station_select]
            daily_station = daily[daily["AWS_ID"] == station_select]
            event_station = events[events["AWS_ID"] == station_select]

            # ---------- 1️ Rainy Hours per Day ----------
            if analysis_choice == "Rainy Hours per Day":
                st.markdown("####  Number of Rainy Hours per Day")
                daily_rain_counts = df_station.groupby('Date')['Hourly_Rain'].apply(lambda x: (x > 0).sum()).reset_index(name='Hours_Rained')
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(daily_rain_counts, use_container_width=True)
                with col2:
                    fig = px.bar(daily_rain_counts, x='Date', y='Hours_Rained', title=f'Number of Rainy Hours per Day - {station_select}',color_discrete_sequence=["#A6B1B8"])
                    st.plotly_chart(fig, use_container_width=True)

            # ---------- 2 Rainy Days per Month and Season ----------
            elif analysis_choice == "Rainy Days per Month and Season":
                st.markdown("####  Number of Rainy Days per Month and Season")
                daily_station['RainDay'] = (daily_station['Daily_Rainfall'] > 0).astype(int)

                def get_season(month):
                    if month in [1, 2]:
                        return 'Winter'
                    elif month in [3, 4, 5]:
                        return 'Pre-Monsoon'
                    elif month in [6, 7, 8]:
                        return 'Monsoon'
                    else:
                        return 'Post-Monsoon'

                daily_station['Season'] = daily_station['Month'].apply(get_season)
                monthly_rain_days = daily_station.groupby('Month')['RainDay'].sum().reset_index(name='Rainy_Days')
                seasonal_rain_days = daily_station.groupby('Season')['RainDay'].sum().reset_index(name='Rainy_Days')

                col3, col4 = st.columns([1, 2])
                with col3:
                    st.write("**Monthly Rainy Days**")
                    st.dataframe(monthly_rain_days, hide_index=True, use_container_width=True)
                    st.write("**Seasonal Rainy Days**")
                    st.dataframe(seasonal_rain_days, hide_index=True, use_container_width=True)
                with col4:
                    fig = px.bar(monthly_rain_days, x='Month', y='Rainy_Days', title=f'Rainy Days per Month - {station_select}',color_discrete_sequence=["#A6B1B8"])
                    st.plotly_chart(fig, use_container_width=True)

            # ---------- 3️ High-Intensity and Maximum Rainfall Events ----------
            elif analysis_choice == "High-Intensity and Maximum Rainfall Events":
                st.markdown("####  High-Intensity and Maximum Rainfall Events")
                intense_events = event_station[event_station['Average_Intensity'] > 5]

                col5, col6 = st.columns([1, 2])
                with col5:
                    st.metric("Max Rainfall Intensity (mm/hr)", f"{event_station['Average_Intensity'].max():.2f}")
                    st.metric("Total Events (>5 mm/hr)", f"{len(intense_events)}")
                    st.dataframe(intense_events[['Start', 'End', 'Duration_hrs', 'Total_Rain', 'Average_Intensity']], use_container_width=True)
                with col6:
                    fig = px.histogram(intense_events, x="Average_Intensity", nbins=20,
                                    color_discrete_sequence=["#959799"],
                                    title=f"Distribution of High-Intensity Events (>5 mm/hr) - {station_select}")
                    st.plotly_chart(fig, use_container_width=True)

            # ---------- 4️ Monthly Distribution of Event Intensities ----------
            elif analysis_choice == "Monthly Distribution of Event Intensities":
                st.markdown("####  Monthly Distribution of Event Intensities")
                monthly_intensity = event_station.copy()
                monthly_intensity['Month'] = monthly_intensity['Start'].dt.month
                fig = px.box(monthly_intensity, x="Month", y="Average_Intensity",
                            title=f"Monthly Distribution of Event Intensities - {station_select}",
                            color_discrete_sequence=["#A6B1B8"])
                st.plotly_chart(fig, use_container_width=True)


else:
    st.info(" Please upload a CSV file to start the analysis.")

# ---------- FOOTER ----------
st.markdown("""
    <div class="footer">
        Developed by <b>RAFT Research Group, Department of Civil Engineering, IIT Hyderabad</b>
    </div>
""", unsafe_allow_html=True)

