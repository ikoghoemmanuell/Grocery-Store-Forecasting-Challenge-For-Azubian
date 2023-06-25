import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from bokeh.plotting import figure
from bokeh.models import HoverTool
import joblib
import os

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model from the pickle file
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

# Load the scaler from the pickle file
scaler_path = os.path.join(current_dir, 'encoder.pkl')
scaler = joblib.load(scaler_path)


# Define the getDateFeatures() function
def getDateFeatures(date):
    df = pd.DataFrame({'date': [date]})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week

    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)

    # Extract the 'year' and 'weekofyear' components from the 'date' column
    df['year_weekofyear'] = df['date'].dt.year * 100 + df['date'].dt.weekofyear

    # create new columns to represent the cyclic nature of a year
    df['dayofyear'] = df['date'].dt.dayofyear
    df["sin(dayofyear)"] = np.sin(df["dayofyear"])
    df["cos(dayofyear)"] = np.cos(df["dayofyear"])

    df["is_weekend"] = np.where(df['dayofweek'] > 4, 1, 0)

    # Define the criteria for each season
    seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}

    # Create the 'season' column based on the 'date' column
    df = df.set_index('date')
    prediction_inputs = df.to_dict(orient='records')[0]
    return prediction_inputs

# Set Page Configurations
st.set_page_config(page_title="ETA Prediction App", page_icon="fas fa-chart-line", layout="wide", initial_sidebar_state="auto")

# Loading GIF
gif_url = "https://raw.githubusercontent.com/Gilbert-B/Forecasting-Sales/main/app/salesgif.gif"

# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox("Select an option", menu)


# Home section
if choice == 'Home':
    st.image(gif_url, use_column_width=True)
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)
    
    # Set Page Title
    st.title('SEER- A Sales Forecasting APP')
    st.markdown('Enter the required information to forecast sales:')

    
    # Input form
    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("Date")

        onpromotion = st.number_input("How many products are on promotion?", min_value=0, step=1)

        
        categories = ['Category_0', 'Category_1', 'Category_2','Category_3', 'Category_4', 'Category_5'
                      ,'Category_6', 'Category_7', 'Category_8','Category_9', 'Category_9', 'Category_11'
                      ,'Category_12', 'Category_13', 'Category_14','Category_15', 'Category_16','Category_17'
                      ,'Category_18', 'Category_19', 'Category_20','Category_21', 'Category_22', 'Category_23'
                      ,'Category_24', 'Category_25', 'Category_26','Category_27', 'Category_28', 'Category_29'
                      ,'Category_30', 'Category_31', 'Category_32']  
       
        selected_category = st.selectbox("Category", categories)


    with col2:
        Stores = ['Store_1', 'Store_2', 'Store_3','Store_4', 'Store_5', 'Store_6', 'Store_7','Store_8', 'Store_9', 'Store_10',
                  'Store_11', 'Store_12', 'Store_13','Store_14', 'Store_15', 'Store_16', 'Store_17','Store_18', 'Store_19', 'Store_20',
                  'Store_21', 'Store_22', 'Store_23','Store_24', 'Store_25', 'Store_26', 'Store_27','Store_28', 'Store_29', 'Store_30',
                  'Store_31', 'Store_32', 'Store_33','Store_34', 'Store_35', 'Store_36', 'Store_37','Store_38', 'Store_39', 'Store_40',
                  'Store_41', 'Store_42', 'Store_43','Store_44', 'Store_45', 'Store_46', 'Store_47','Store_48', 'Store_49', 'Store_50',
                  'Store_51', 'Store_52', 'Store_53','Store_54']
        selected_store = st.selectbox("Store", Stores)


        cities = ['city_0','city_1', 'city_2', 'city_3', 'city_4', 'city_5', 'city_6', 'city_7', 'city_8', 'city_9',
                  'city_10', 'city_11', 'city_12', 'city_13', 'city_14', 'city_15', 'city_16', 'city_17',
                  'city_18', 'city_19', 'city_20', 'city_21']
        selected_city = st.selectbox("City", cities)


        clusters = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6',
                    'cluster_7', 'cluster_8', 'cluster_9', 'cluster_10', 'cluster_11', 'cluster_12', 'cluster_13',
                    'cluster_14', 'cluster_15', 'cluster_16']
        selected_cluster = st.selectbox("Cluster", clusters)

    # Call getDateFeatures() function on sales_data (replace sales_data with your DataFrame)
    sales_data = pd.DataFrame({
        'date': [date],
        'store': [selected_store],
        'category': [selected_category],
        'onpromotion': [onpromotion],
        'cities' :[selected_city],
        'clusters':[selected_cluster]
        })

    date = getDateFeatures(sales_data['date'].values[0])

    # Make predictions for the next 8 weeks
    prediction_inputs = []  # Initialize the list for prediction inputs
   
    for week in range(1, 9):
        offset = pd.DateOffset(weeks=week)
        datetime_object = pd.to_datetime(date['date'].values[0]) + offset
        prediction_inputs.append([
        date['date'].values[0],
        datetime_object,
        selected_store,
        selected_category,
        onpromotion,
        selected_city,
        selected_cluster
    ])
    def prediction_inputs(date):
        prediction_inputs = pd.DataFrame({
        'date': date,
    })    
    prediction_inputs['year'] = getDateFeatures(prediction_inputs['date'].values[0])['year']
    prediction_inputs['month'] = getDateFeatures(prediction_inputs['date'].values[0])['month']
    prediction_inputs['dayofmonth'] = getDateFeatures(prediction_inputs['date'].values[0])['dayofmonth']
    prediction_inputs['dayofweek'] = getDateFeatures(prediction_inputs['date'].values[0])['dayofweek']
    prediction_inputs['weekofyear'] = getDateFeatures(prediction_inputs['date'].values[0])['weekofyear']
    prediction_inputs['quarter'] = getDateFeatures(prediction_inputs['date'].values[0])['quarter']
    prediction_inputs['is_month_start'] = getDateFeatures(prediction_inputs['date'].values[0])['is_month_start']
    prediction_inputs['is_month_end'] = getDateFeatures(prediction_inputs['date'].values[0])['is_month_end']
    prediction_inputs['is_quarter_start'] = getDateFeatures(prediction_inputs['date'].values[0])['is_quarter_start']
    prediction_inputs['is_quarter_end'] = getDateFeatures(prediction_inputs['date'].values[0])['is_quarter_end']
    prediction_inputs['is_year_start'] = getDateFeatures(prediction_inputs['date'].values[0])['is_year_start']
    prediction_inputs['is_year_end'] = getDateFeatures(prediction_inputs['date'].values[0])['is_year_end']

    # Scale the prediction inputs
    prediction_inputs_scaled = scaler.transform(prediction_inputs)


    if st.button('Predict'):
        # Make the prediction using the loaded machine learning model
        predicted_sales = model.predict(prediction_inputs_scaled)

    

# Scale the prediction inputs
prediction_inputs_scaled = scaler.transform(prediction_inputs)

if st.button('Predict'):
    # Make the prediction using the loaded machine learning model
    predicted_sales = model.predict(prediction_inputs_scaled)


    # Display the forecast results
    st.subheader("Sales Forecast for the Next 8 Weeks:")
    for week, sales in enumerate(predicted_sales, start=1):
        st.write(f"Week {week}: {sales:.2f} units")

    # Update the line chart
    chart_data = pd.DataFrame({'Week': range(1, 9), 'Sales': predicted_sales})
    p = figure(plot_width=600, plot_height=400, title="Sales Forecast",
               x_axis_label="Week", y_axis_label="Sales")

    p.line(chart_data['Week'], chart_data['Sales'], line_width=2)
    p.circle(chart_data['Week'], chart_data['Sales'], fill_color="white", size=6)
    p.add_tools(HoverTool(tooltips=[("Week", "@x"), ("Sales", "@y")]))
    st.bokeh_chart(p)

# About section
elif choice == 'About':
    # Load the banner image
    banner_image_url = "https://raw.githubusercontent.com/Gilbert-B/Forecasting-Sales/0d7b869515bysBoi5XxNGa3hayALLn9BK1VQqD69Dc/app/seer.png"
    banner_image = Image.open(requests.get(banner_image_url, stream=True).raw)

    # Display the banner image
    st.image(banner_image, use_column_width=True)
    st.markdown('''
            <p style='font-size: 20px; font-style: italic;font-style: bold;'>
            SEER is a powerful tool designed to assist businesses in making accurate 
            and data-driven sales predictions. By leveraging advanced algorithms and 
            machine learning techniques, our app provides businesses with valuable insights 
            into future sales trends. With just a few input parameters, such as distance and 
            average speed, our app generates reliable sales forecasts, enabling businesses
            to optimize their inventory management, production planning, and resource allocation. 
            The user-friendly interface and intuitive design make it easy for users to navigate 
            and obtain actionable predictions. With our Sales Forecasting App, 
            businesses can make informed decisions, mitigate risks, 
            and maximize their revenue potential in an ever-changing market landscape.
            </p>
            ''', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This Sales Forecasting App is developed using Streamlit and Python.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>It demonstrates how machine learning can be used to predict sales for the next 8 weeks based on historical data.</p>", unsafe_allow_html=True)
