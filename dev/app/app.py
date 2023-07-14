import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from bokeh.plotting import figure
from bokeh.models import HoverTool
import joblib
import os
from date_features import getDateFeatures


# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model from the pickle file
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

# Load the scaler from the pickle file
encoder_path = os.path.join(current_dir, 'encoder.pkl')
encoder = joblib.load(encoder_path)


# Set Page Configurations
st.set_page_config(page_title="Sales Prediction App", page_icon="fas fa-chart-line", layout="wide", initial_sidebar_state="auto")

# Loading GIF
gif_url = "https://raw.githubusercontent.com/Gilbert-B/Forecasting-Sales/main/app/salesgif.gif"

# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox("Select an option", menu)

def predict(sales_data):
    if sales_data.empty:
        raise ValueError("No sales data provided.")
    
    # Perform the necessary data processing steps
    sales_data = getDateFeatures(sales_data).set_index('date')
    numeric_columns = ['onpromotion', 'year', 'month', 'dayofmonth', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'year_weekofyear', 'sin(dayofyear)', 'cos(dayofyear)']
    categoric_columns = ['store_id', 'category_id', 'city', 'store_type', 'cluster', 'holiday_type', 'is_holiday', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_weekend', 'season']
    
    num = sales_data[numeric_columns]
    encoded_cat = encoder.transform(sales_data[categoric_columns])
    sales_data = pd.concat([num, encoded_cat], axis=1)

    # Make predictions using the pre-trained model
    predicted_sales = model.predict(sales_data)

    return predicted_sales

# Home section
if choice == 'Home':
    # Set Page Title
    st.title('SEER- A Sales Forecasting APP')
    
    st.image(gif_url, use_column_width=True)
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)
    
    st.markdown('Enter the required information to forecast sales:')

    
    # Input form
    col1, col2 = st.columns(2)

    Store_type = ["Supermarket", "Department Store", "Convenience Store", "Pharmacy", "Clothing Store", "Electronics Store", "Furniture Store", "Hardware Store", "Bookstore", "Jewelry Store", "Toy Store", "Pet Store", "Sporting Goods Store", "Shoe Store", "Dollar Store", "Liquor Store", "Beauty Supply Store", "Home Improvement Store", "Stationery Store", "Gift Shop", "Bakery", "Butcher Shop", "Fish Market", "Vegetable Market", "Farmers Market", "Coffee Shop", "Café", "Restaurant", "Fast Food Restaurant", "Pizza Place", "Burger Joint", "Ice Cream Shop", "Food Truck", "Bar", "Pub", "Nightclub", "Gas Station", "Car Dealership", "Auto Repair Shop", "Car Wash", "Bank", "ATM", "Post Office", "Laundry", "Hair Salon", "Nail Salon", "Spa", "Gym", "Yoga Studio", "Movie Theater", "Bowling Alley", "Arcade", "Museum", "Art Gallery"]
    Stores1 = ['Store_' + str(i) for i in range(0, 5)]
    cities = ["Lagos", "Abuja", "Kano", "Ibadan", "Kaduna", "Port Harcourt", "Benin City", "Maiduguri", "Zaria", "Aba", "Jos", "Ilorin", "Oyo", "Enugu", "Abeokuta", "Onitsha", "Warri", "Sokoto", "Calabar", "Katsina", "Akure", "Bauchi"]
    clusters = ["Fashion", "Electronics", "Supermarket", "Home Improvement", "Department Store","Pharmacy", "Furniture", "Sports Goods", "Jewelry", "Cosmetics", "Automotive","Bookstore", "Toy Store", "Pet Store", "Convenience Store", "Hardware Store","Outdoor Recreation"]
    categories = ["Apparel", "Beauty", "Books", "Electronics", "Furniture", "Grocery", "Health", "Home", "Jewelry", "Kitchen", "Music", "Office", "Outdoors", "Pets", "Shoes", "Sports", "Toys", "Automotive", "Baby", "Computers", "Garden", "Movies", "Tools", "Watches", "Appliances", "Cameras", "Fitness", "Industrial", "Luggage", "Software", "Video Games", "Cell Phones", "Home Improvement"]

    with col1:
        start_date = st.date_input("Start Date")
        # Convert the date to datetime format
        start_date = pd.to_datetime(start_date)
        end_date = st.date_input("End Date")
        # Convert the date to datetime format
        end_date = pd.to_datetime(end_date)
        onpromotion = st.number_input("How many products are on promotion?", min_value=0, step=1)
        selected_category = st.selectbox("Product_Category", categories)


    with col2:
        selected_store = st.selectbox("Store_type", Store_type)
        selected_store1 = st.selectbox("Store_id", Stores1)
        selected_city = st.selectbox("City", cities)
        selected_cluster = st.selectbox("Cluster", clusters)

    predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'Store', 'Category', 'On Promotion', 'City', 'Cluster', 'Predicted Sales'])


    if st.button('Predict'):
        if start_date > end_date:
            st.error("Start date should be earlier than the end date.")
        else:
            with st.spinner('Predicting sales...'):
                sales_data = pd.DataFrame({
                    'date': pd.date_range(start=start_date, end=end_date),
                    'store_id': [selected_store] * len(pd.date_range(start=start_date, end=end_date)),
                    'category_id': [selected_category] * len(pd.date_range(start=start_date, end=end_date)),
                    'onpromotion': [onpromotion] * len(pd.date_range(start=start_date, end=end_date)),
                    'city': [selected_city] * len(pd.date_range(start=start_date, end=end_date)),
                    'store_type': [selected_store1] * len(pd.date_range(start=start_date, end=end_date)),
                    'cluster': [selected_cluster] * len(pd.date_range(start=start_date, end=end_date))
                })
                try:
                    sales = predict(sales_data)
                    formatted_sales = round(sales[0], 2)
                    predicted_data = predicted_data.append({
                         'Start Date': start_date,
                         'End Date': end_date,
                         'Store': selected_store,
                         'Category': selected_category,
                         'On Promotion': onpromotion,
                         'City': selected_city,
                         'Cluster': selected_cluster,
                         'Predicted Sales': formatted_sales}, ignore_index=True)

                    st.success(f"Total sales for the period is: #{formatted_sales}")
                except ValueError as e:
                    st.error(str(e))
    if st.button('Clear Data'):
        predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'Store', 'Category', 'On Promotion', 'City', 'Cluster', 'Predicted Sales'])
        st.success("Data cleared successfully.")



# About section
elif choice == 'About':
    # Load the banner image
    banner_image_url = "https://guardian.ng/wp-content/uploads/2017/03/Sales-targets.jpg"
 
    # Display the banner image
    st.image(
        banner_image_url, 
        use_column_width=True, 
        width=400)
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