import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt



# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model from the pickle file
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

# Load the scaler from the pickle file
scaler_path = os.path.join(current_dir, 'encoder.pkl')
scaler = joblib.load(scaler_path)


# Set Page Configurations
st.set_page_config(page_title="Sales Forecasting App", page_icon="fas fa-chart-line", layout="wide", initial_sidebar_state="auto")

# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox("Select an option", menu)

# Home section
if choice == 'Home':
    # Set Page Title
    st.title('Sales Forecasting App')
    st.markdown('Enter the required information to forecast sales:')

    # Input form
    date = st.date_input("Date")
    onpromotion = st.number_input("How many products are on promotion?", min_value=0, step=1)
    
    categories = ['Category_0', 'Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5', 'Category_6',
                  'Category_7', 'Category_8', 'Category_9', 'Category_10', 'Category_11', 'Category_12', 'Category_13',
                  'Category_14', 'Category_15', 'Category_16', 'Category_17', 'Category_18', 'Category_19', 'Category_20',
                  'Category_21', 'Category_22', 'Category_23', 'Category_24', 'Category_25', 'Category_26', 'Category_27',
                  'Category_28', 'Category_29', 'Category_30', 'Category_31', 'Category_32']
    selected_category = st.selectbox("Category", categories)

    Stores = ['Store_1', 'Store_2', 'Store_3', 'Store_4', 'Store_5', 'Store_6', 'Store_7', 'Store_8', 'Store_9', 'Store_10',
              'Store_11', 'Store_12', 'Store_13', 'Store_14', 'Store_15', 'Store_16', 'Store_17', 'Store_18', 'Store_19',
              'Store_20', 'Store_21', 'Store_22', 'Store_23', 'Store_24', 'Store_25', 'Store_26', 'Store_27', 'Store_28',
              'Store_29', 'Store_30', 'Store_31', 'Store_32', 'Store_33', 'Store_34', 'Store_35', 'Store_36', 'Store_37',
              'Store_38', 'Store_39', 'Store_40', 'Store_41', 'Store_42', 'Store_43', 'Store_44', 'Store_45', 'Store_46',
              'Store_47', 'Store_48', 'Store_49', 'Store_50', 'Store_51', 'Store_52', 'Store_53', 'Store_54']
    selected_store = st.selectbox("Store", Stores)

    cities = ['city_0', 'city_1', 'city_2', 'city_3', 'city_4', 'city_5', 'city_6', 'city_7', 'city_8', 'city_9', 'city_10',
              'city_11', 'city_12', 'city_13', 'city_14', 'city_15', 'city_16', 'city_17', 'city_18', 'city_19', 'city_20',
              'city_21']
    selected_city = st.selectbox("City", cities)

    clusters = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7',
                'cluster_8', 'cluster_9', 'cluster_10', 'cluster_11', 'cluster_12', 'cluster_13', 'cluster_14', 'cluster_15',
                'cluster_16']
    selected_cluster = st.selectbox("Cluster", clusters)

    # Make predictions for the next 8 weeks
    prediction_inputs = []  # Initialize the list for prediction inputs
    for week in range(1, 9):
        prediction_inputs.append([
            date,
            selected_store,
            selected_category,
            onpromotion,
            selected_city,
            selected_cluster
        ])

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
        fig, ax = plt.subplots()
        ax.plot(chart_data['Week'], chart_data['Sales'], marker='o')
        ax.set_xlabel("Week")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Forecast")
        st.pyplot(fig)

# About section
elif choice == 'About':
    st.title("About")
    st.markdown('''
        This Sales Forecasting App demonstrates how machine learning can be used to predict sales 
        for the next 8 weeks based on historical data. By providing the necessary input parameters, 
        such as date, on-promotion products, category, store, city, and cluster, the app generates 
        a forecast of sales for the upcoming weeks. The app utilizes a trained machine learning model 
        that has been trained on historical sales data and is capable of making accurate predictions 
        based on the provided inputs. The predictions are displayed in both numerical and graphical forms 
        to provide a clear understanding of the sales forecast trends.
    ''')
