# Focast Grocery Sales Using Machine Learning 
![grocery-sales-challenge-readme-azubian-](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/assets/102419217/621cc943-14ca-4148-bed5-85afd96b6c96)
 

Azubian, a company with grocery stores across Africa, has been facing a decline in sales lately, primarily due to losing customers due to understocking popular goods. When customers visit their stores and find essential items like rice unavailable, they may be disappointed and shop elsewhere. Another challenge they've encountered is limited storage space for their stock.

To address these issues and improve their business outlook, Azubian seeks the expertise of a skilled data analyst like yourself. They have approached you to analyze their sales data and provide valuable recommendations for the future. Your analysis will focus on identifying the products most likely to run out of stock and predict the quantities required, enabling them to proactively manage their inventory and meet customer demand effectively. 

## What is time series analysis?

This is a time series regression problem. Now let me break that down. First of all, what is time series analysis?

Time series analysis involves time, meaning that you analyze a dataset with a date column. 

A time series state is a data set collected at successive points in time, while a cross-sectional dataset is collected at a fixed time for a dataset where time doesn't matter. 
 
For example, if your smartwatch tracks your sleep, it should be able to tell you how much sleep you got throughout the month. The graph below shows my sleep over a fixed period. The y-axis shows my sleep levels, while the x-axis shows the day of the month. So when time is involved, the dataset becomes a Time Series dataset.


![my sleep chart](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689435953689_image.png)


Watch [this video](https://youtu.be/FsroWpkUuYI) to know more about this.


## Importance of a Time Series Data Project

Why is this important? 

This is because most businesses out there are affected by time. Time usually affects a product's sales and a business's revenue. For example, the sales of clothes are high during the Christmas period. Another example is agriculture; one can use time series analysis to determine the best time of the year to plant their crops.

How do you know whether a dataset is a time series dataset? Simple; if there is a time variable in it (usually the index), then yes, it is a time series dataset, else it is not. 


## Aim of the project

In this project, we want to predict sales of products sold in grocery stores; owned by Azubian. We want to know which products sell better and at what period. We also want to predict future sales to avoid overstocking or understocking. 

For example, we want to know how much chicken to buy for the coming 8 weeks; or how much tomato will be sold in the next 8 weeks. 

For perishable goods like so, if we overstock, it will end up getting spoilt, and that’s a loss. We want to provide tools to satisfy customers by always having what they want to buy when they want it. This could give our grocery stores a competitive advantage in today’s market.

At the end of this project, we would have built a machine learning model to forecast product sales pay week over the next 8 weeks. We want to get this information for each store in each location.

Finally, we’ll deploy this model as a web application using Streamlit.

Clone the repo and get started from  [here.](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian)

# Dataset description

The dataset provided for this project contains for 4 years and is broken into 6 files. Here’s a brief overview of the dataset.

**train.csv**
This is the file that will be used to train the model. It contains the target column which is sales. It also has a column called “onpromotion.” This column will tells us how many products were being promoted (advertised) at the time.


![Preview of train.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689421639944_image.png)


**holidays.csv**
This file contains information about holidays.

![preview of holidays.csv](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689421953863_image.png)


**test.csv**
This file resembles train.csv file but without the target column. We will use this file to evaluate the performance of the trained model.

![Preview of test.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689422068460_image.png)


**stores.csv**
Contains information about each store such as their city, type and cluster.

![Preview of stores.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689422175669_image.png)


**dates.csv**
Contains information about each day of each year in our dataset, and the associated date features.

![Preview of dates.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689422552387_image.png)



## Issues with the data

Here are the major issues observed from previewing the dataset.

![Issues with the data and how to solve them](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689423176844_image.png)




# Analysis and cleaning of the dataset.

In this stage, let’s start preparing the dataset for analysis.

## Observations

Notably, I won't outline all of them here for the sake of this article. Here are my key observations.


- The date column in dates.csv file ranges from 365 - 1684, which covers all the dates for the train.csv and test.csv files.
- There are no null values present.
- There are no missing values.
- There is no holiday column in any other csv file apart from the holidays.csv file.


## Data cleaning

Let me highlight the major Steps I took to prepare in the dataset.


1. To get a proper date column, I combined the year, month and day columns of the dates.csv file to create a full date column. This column will be used as the date index for the other files.


    def get_datetime(df):
      # Create a new column combining the year, month, and day of the month in the desired format (yyyy-mm-dd)
      df['date_extracted'] = (
          dates['year'].astype(int).add(2000).astype(str) + '-' +
          dates['month'].astype(str).str.zfill(2) + '-' +
          dates['dayofmonth'].astype(str).str.zfill(2)
      )
    
    get_datetime(dates)



2. Using the holidays.csv file, I created a column called “holiday_type” using this logic. If the date is in holidays.csv file, then it is a holiday. Else it is a work day.


    def get_is_holiday_column(df):
      df['holiday_type'] = df['holiday_type'].fillna('Workday')
    
      # create column to show if its a holiday or not (non-holidays are zeros)
      df['is_holiday'] = df['holiday_type'].apply(
          lambda x: False if x=='Workday'
          else True)



3. Set the date column we just created as index.


    def set_index(df):
      df.drop('date', inplace=True, axis=1)
      df.set_index('date_extracted', inplace=True)
    set_index(train_merged2)
    set_index(test_merged2)



4. After checking for invalid dates, the only invalid date is “2003-02-29”, so when converting to datetime, we will first set invalid dates to NaT. Then fill them with “2003-02-29” which is february 29 for a leap year.


    train_merged2['date_extracted'] = pd.to_datetime(train_merged2['date_extracted'], errors='coerce')
    test_merged2['date_extracted'] = pd.to_datetime(test_merged2['date_extracted'])
    train_merged2['date_extracted'].fillna('2003-02-29')



# Exploratory data analysis

In this section, we are going to ask some questions and come up with a hypothesis.


## Hypothesis

***Null Hypothesis***: holidays have a big effect on sales, hence the sales data is seasonal.
***Alternative Hypothesis***: holidays don't affect sales, hence sales data is stationary.

## Hypothesis Validation

Let’s check if our null hypothesis is true or not.

![Bar chart of sales vs holiday type](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689428352904_image.png)

![Box plot of sales during holidays vs non-holidays](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689428466937_image.png)


The hypothesis H1, which states that holidays affect sales and the sales data is seasonal, is more likely to be true. Alternative Hypothesis REJECTED!


## Questions & answers  
1. **How do sales vary by promotion status?**


![](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689429178014_image.png)


From the above plot, we can see that promotions greatly impact overall sales. Sales are higher when a product is on promotion.


2. **Is there a relationship between sales and** **transactions?**
![Scatterplot of sales vs transactions](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689429374239_image.png)


Transactions do not affect sales. The above plot shows that sales has little correlation with transactions or no correlation at all.


3. **How does sales vary during holidays compared to non-holidays?**
![Bar chart and box plot of sales during holidays vs non-holidays](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689429618307_image.png)


The chart on the left shows that holiday sales are higher than sales on non-holidays. What does this mean? The total sales during holidays alone is higher than the total sales during all normal days combined. The plot on the right shows the outliers on holidays are much higher than those on non-holidays. This is expected since the craziest sales numbers are usually seen during holidays.


4. **What is the trend of sales over time?** 
![plot of sales over time](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689430307584_image.png)


Over time, sales have been increasing, we can see an overall increase in sales from the above chart.


5. **How much does promotion affect sales of different product categories?**
![](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689433239456_InkedgG9Q9kfT_LI.jpg)


We can see from the above that some product categories are less affected by promotions than others, for example we can see that ‘category_0’ on the left has higher sales with more promotion than ‘category_7’ on the right.


6. **Is our sales data seasonal?**
![monthly sales over time](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689431531919_image.png)

> What is the difference between seasonality and trend? Seasonality and trend are two different concepts in time series analysis. Seasonality refers to the repeating patterns in a time series that occur over a fixed period of time, such as daily, weekly, or monthly. Trend refers to the long-term direction of a time series, such as increasing or decreasing.
> 
> For example, the number of ice cream sales might be seasonal, with higher sales in the summer and lower sales in the winter. The number of people using a social media platform might have a trend, with increasing usage over time.


Checking for seasonality by just looking at a chart can be tricky. So let’s use a statistical method to check it. We’ll be using the KPSS test.


    # Assuming your time series data is stored in the variable 'sales_data'
    sales_data = train['target']


    # Perform KPSS test
    kpss_result = kpss(sales_data)
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_critical_values = kpss_result[3]


![](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689432517211_image.png)


Sales data is stationary if p-value > 0.05. Sales data is stationary since 0.01 < 0.05. This means that our sales is non-seasonal and there is no repeating pattern over time.



# Recommendations 

The goal of data analysis is to give actionable recommendations to stakeholders. The following are some data driven decisions the company should take to boost sales.


> **Utilize promotions**: Promotions have a significant impact on sales. Consider implementing strategic promotional campaigns for products to boost sales. Identify products that have higher sales when on promotion and focus on promoting them more frequently.
> 
> **Focus on sales drivers**: While transactions do not seem to directly affect sales, it’s important to identify other factors that drive sales. Analyze customer behavior, product attributes, pricing, and other variables to understand what influences sales the most. Use this information to optimize your sales strategies.
> 
> **Capitalize on holidays**: Holidays drive higher sales compared to non-holidays. Plan and execute targeted marketing and promotional campaigns during holiday seasons to take advantage of increased consumer spending. Ensure sufficient inventory and resources to accommodate the surge in demand during these periods.
> 
> **Monitor and adapt to sales trends**: Sales have been consistently increasing over time. Continuously monitor sales data and identify emerging trends. Adjust your inventory, marketing, and sales strategies accordingly to maximize the opportunities presented by the growing sales trend.
> 
> **Tailor promotions to product categories**: Different product categories respond differently to promotions. Analyze the impact of promotions on various product categories and allocate promotional efforts accordingly. Focus more on categories that show a higher increase in sales with promotions to maximize their potential.
> 
> **Leverage non-seasonality**: Since the sales data is non-seasonal, explore other factors driving customer demand and purchasing behavior. Consider conducting customer surveys, analyzing market trends, and studying competitors to identify opportunities for growth outside of seasonal patterns.


Now that you know what the dataset is all about and I’ve shared some insights with you, it’s time to start machine learning modeling. Hurray!

# Machine learning Modelling

For a Time series regression problem, we can train two types of models. We have the traditional time series models like the AR model. Then we have the machine learning models you are used to (like linear regression). Both have their advantages and disadvantages. Let Me Explain.

Traditional time Series models use data from the past to predict the future using correlation. They rely on the correlation between past values and future values. In this case, they would rely on the fact that last week's sales might be correlated or have a relationship with this week's sales for example.

The diagram below shows us the partial autocorrelation of the sales column alone. Being that this is a time series project, the three lines highlighted show that for each sales value, the past 3 sales values correlate with it. This is autocorrelation. It is on this basis that the traditional time series models work. Interesting right? 🤯 I know.


![Plot of partial autocorrelation](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689436910282_Inked_-ggIk3B_LI.jpg)


Using machine learning models is a feature driven approach. these models can take several features as input parameters. For this approach, it is good to select the features that are most important for your training. 

The table below shows a brief explanation of the difference between the two.


![machine learning vs traditional time series models for time series modelling](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689437265730_image.png)


After training with the two approaches, this is my result.


![traditional time series models](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689437759039_image.png)



![machine learning models](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689437748585_image.png)


To build the app, I picked the best model from the second approach, which is KNN model with a RMSE score of 123.11. I picked a model from the second approach because it was trained with more features that the user can play with.

# Model deployment 

This is the simplest part of the project. Now that the data analysis is done, and we have trained our model, we can now deploy the best performing model as a user friendly web application.

I exported the best model and encoder from the notebook using pickle and built a web application with Streamlit. Streamlit is a python framework that makes it easy to deploy web applications quickly. Let me Explain step-by-step all the code that I used to build the app. [Try the app.](https://huggingface.co/spaces/ikoghoemmanuell/SEER-A_sales_forecasting_app)


![screenshot of the app](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/raw/main/seer-sales-prediction-for-azubian.gif)



## Importing necessary requirements


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

We first of all import the libraries that we need, like streamlit, pandas, numpy and the likes. We also imported a function from another python file called *date_features.py*. This file contains a function called *getDateFeatures* which will take a dataframe as input and return the date features like the year month, day of month, week of year, day of week, season and so on.


    import numpy as np
    
    # Define the getDateFeatures() function
    def getDateFeatures(df):
        df['holiday_type'] = 'Workday'
        df['is_holiday'] = False
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dayofmonth'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekofyear'] = df['date'].dt.weekofyear
    
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        # Extract the 'year' and 'weekofyear' components from the 'date' column
        df['year_weekofyear'] = df['date'].dt.year * 100 + df['date'].dt.weekofyear
    
        # create new coolumns to represent the cyclic nature of a year
        df['dayofyear'] = df['date'].dt.dayofyear
        df["sin(dayofyear)"] = np.sin(df["dayofyear"])
        df["cos(dayofyear)"] = np.cos(df["dayofyear"])
    
        df["is_weekend"] = np.where(df['dayofweek'] > 4, 1, 0)
    
        # Define the criteria for each season
        seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}
    
        # Create the 'season' column based on the 'date' column
        df['season'] = df['month'].map({month: season for season, months in seasons.items() for month in months})
        
    
        return df

The file containing this function must be located in the same directory as is the *app.py* file. 


## Loading machine learning assets

Now we load the files for machine learning like the model and encoder which should also be in the same directory as the *app.py* file. 


    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the model from the pickle file
    model_path = os.path.join(current_dir, 'model.pkl')
    model = joblib.load(model_path)
    
    # Load the scaler from the pickle file
    encoder_path = os.path.join(current_dir, 'encoder.pkl')
    encoder = joblib.load(encoder_path)



# Building the interface
## Basic configurations

First we must set page configurations for the app like the title, the layout, the initial side bar state and the icon for the page.


    # Set Page Configurations
    st.set_page_config(page_title="Sales Prediction App", page_icon="fas fa-chart-line", layout="wide", initial_sidebar_state="auto")

Now we set up the sidebar of the app. The user should be able to choose between two pages within the app. The home page and the about page.


![code for the sidebar](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689440921302_image.png)



## Home section

If you are in the home section you will first of all be introduced to the app and then you will be required to fill an input form and provide some parameters for sales prediction. 


![Intro part of the home page](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689441116579_image.png)


Column one should have the start date, end date, onpromotion and product category.

While column two should have information about the store; like the store type and store ID as well as city and cluster.


![input form on the home page](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689441304475_image.png)


Now all this information will be saved as a dataframe.


    predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'Store', 'Category', 'On Promotion', 'City', 'Cluster', 'Predicted Sales'])


Now we are almost done with the homepage. all that is left now is two buttons. the "predict" button and the "clear data" button.

The predict button uses a function called predict. This function takes the dataframe as input, performs the encoding, and finally uses the model to predict sales. If the sales data is empty, it will raise a value error saying " No sales data provided." Else, it will return the predicted sales as output.


![predict function](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689441600975_image.png)


**Predict button**
If you click the predict button, it will check if the start date comes before the end date, if it comes after, it will raise an error saying "Start date should be earlier than the end date." Else, the predicted sales will be displayed as " total sales for the period is: #salesValue"


    
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
                        


**Clear data button**
If you click the "clear data" button, it will reset the dataframe to become empty, then it returns a message saying "Data cleared successfully."


                        
        if st.button('Clear Data'):
            predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'Store', 'Category', 'On Promotion', 'City', 'Cluster', 'Predicted Sales'])
            st.success("Data cleared successfully.")



## About section

The about section tells you what the app is all about. You have a  banner image for the app, as well as some text explaining what the app does.


![about section code](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689442553146_image.png)



## Hosting the app

To make your app accessible on the internet, you can host it using a platform called Hugging Face. They provide a [tutorial](https://huggingface.co/docs/hub/spaces-overview) to guide you through the process. It's a great resource to help you learn how to do it step by step.


# Conclusion 

We have conducted exploratory data analysis of store data and built a machine-learning model that can predict future sales using Streamlit. We have also deployed the best-performing model as a user-friendly web application. 

The code for this project is available [here.](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian)

Kindly give this article a like or a clap if you like it, or drop a comment. 


## Resources
- [What is time series data?](https://youtu.be/FsroWpkUuYI)
- [What is time series analysis?](https://youtu.be/GE3JOFwTWVM)
- [Get started with Streamlit](https://docs.streamlit.io/library/get-started/create-an-app)
- [A step by step tutorial playlist](https://www.youtube.com/playlist?list=PLa6CNrvKM5QU7AjAS90zCMIwi9RTFNIIW)
- [Huggingface Spaces tutorial](https://huggingface.co/docs/hub/spaces-overview)
- [Showcase your model demos with 🤗 Spaces](https://www.youtube.com/watch?v=vbaKOa4UXoM)

