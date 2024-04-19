import streamlit as st
import joblib
import pandas as pd
import folium
import plotly.express as px

data = pd.read_csv('all_ontario_data.csv')

#page 1
def dashboard():
    st.image('NorQuest.png', use_column_width=True)
    st.subheader('Unlocking Housing Market Dynamics: Data-Driven Solutions for Pricing Analysis and Urban Planning')
    Inspiration='''
    The Ontario Rental Market Data Analysis Using Machine Learning
    '''
    st.write(Inspiration)
    st.subheader('What We Do')
    what_we_do='''
    This project aims to delve into the complexities of the rental market in Ontario, Canada, by employing advanced data analytics and machine learning methodologies. Our goal is to uncover pivotal insights related to rental prices, trends, and the various factors that influence the rental market dynamics.
    '''
    st.write(what_we_do)

#page 2
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Price Distribution
    fig = px.histogram(data, x='Price', nbins=20, title='Distribution of Rental Prices')
    st.plotly_chart(fig)
    
    
    type_counts = data['Type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']

# Calculate percentage
    type_counts['Percentage'] = type_counts['Count'] / type_counts['Count'].sum() * 100

# Format percentage with two decimal places and add to names
    type_counts['Percentage'] = type_counts.apply(lambda row: f"{row['Type']} ({row['Percentage']:.2f}%)", axis=1)

# Create the pie chart
    fig = px.pie(type_counts, values='Count', names='Percentage', title='Distribution of Property Types')

# Display the chart using Streamlit
    st.plotly_chart(fig)

# Assuming 'data' is your DataFrame containing the 'Size' and 'Bathrooms' columns
    fig = px.bar(data, x='Size', y='Bathrooms', title='Price Distribution by Bathrooms')
    st.plotly_chart(fig)

#page 3
def machine_learning_modeling():
    st.title("Kijiji Rental Price Prediction")
    st.write("Enter the details of the property to predict its rental price:")

    # Input fields for user to enter data
    property_type = st.selectbox("Type of Property", ['Apartment', 'House', 'Condo', 'Townhouse'])
    bedrooms = st.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 3, 1)
    size = st.slider("Size (sqft)", 300, 5000, 1000)
    unique_locations = data['CSDNAME'].unique()
    location = st.selectbox("Location", unique_locations)

    if st.button("Predict"):
        # Load the trained model including preprocessing
        model = joblib.load('gradient_boost_regressor_model.pkl')

        # Assuming the model_with_preprocessing is a pipeline that ends with your estimator
        # Prepare input data as a DataFrame to match the training data structure
        input_df = pd.DataFrame({
            'Type': [property_type],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Size': [size],
            'CSDNAME': [location]
        })

        # Make prediction
        prediction = model.predict(input_df)

        # Display the prediction
        st.success(f"Predicted Rental Price: ${prediction[0]:,.2f}")

    #page 4

# Page 4: Community Mapping
def community_mapping():
    st.title("Small Communities Map: Population <10000")
    geodata = pd.read_csv("output_file.csv")


    # Create the map using Plotly Express
    fig = px.scatter_mapbox(geodata,
                            lat='Latitude',
                            lon='Longitude',
                            color='Population',  # Color points by population, or choose another column
                            size='Price',  # Size points by price, or choose another column
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10,
                            hover_name='Type',  # Display property type when hovering over points
                            hover_data={'Price': True, 'Population': True, 'Bathrooms': True, 'Bedrooms': True, 'Size': True, 'Latitude': False, 'Longitude': False},
                            title='Small Communities Map')

    fig.update_layout(mapbox_style="open-street-map")  # Use OpenStreetMap style
    st.plotly_chart(fig)

#page 6
def Assistant():
    st.title('Assistant Chatbot')
    # Assistant
    st.write('Here is our Assistant:')
    chatbot_url = "https://hf.co/chat/assistant/660f337f9959e57107bd0917"
    st.markdown(f'<iframe src="{chatbot_url}" width="100%" height="400" style="border:none;"></iframe>', unsafe_allow_html=True)

    #main
def main():
    st.sidebar.title('Kijiji Community App')
    app_page = st.sidebar.selectbox('Select Page', ['Dashboard', 'Exploratory Data Analysis', 'Machine Learning Modeling', 'Community Mapping','Assistant'])

    if app_page == 'Dashboard':
        dashboard()
    elif app_page == 'Exploratory Data Analysis':
        exploratory_data_analysis()
    elif app_page == 'Machine Learning Modeling':
        machine_learning_modeling()
    elif app_page == 'Community Mapping':
        community_mapping()
    #elif app_page == 'Looker':
       # Looker()
    elif app_page == 'Assistant':
        Assistant()

if __name__ == '__main__':
    main()
