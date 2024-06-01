import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Page title
st.set_page_config(page_title='Rental Price Prediction', page_icon='🎫')
st.title('Rental Price Prediction')
st.info('This is a rental price prediction mainly based in KL & Selangor area. Fill in the information and check the predicted price below.')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv').copy()  # Make a copy of the DataFrame to avoid mutation

# Function to map 'Yes' or 'No' to 1 or 0
def map_yes_no_to_binary(value):
    return 1 if value == 'Yes' else 0

# Define region bins
region_bins = {
    'Kuala Lumpur': 0,
    'Selangor': 1
}

# Define furnished bins
furnished_bins = {
    'Not Furnished': 0,
    'Partially Furnished': 1,
    'Fully Furnished': 2
}

# Define location bins
location_bins = {
    'Alam Impian': 5,
    'Ampang': 3,
    'Ampang Hilir': 6,
    'Ara Damansara': 6,
    'Balakong': 2,
    'Bandar Botanic': 2,
    'Bandar Bukit Raja': 3,
    'Bandar Bukit Tinggi': 6,
    'Bandar Damai Perdana': 4,
    'Bandar Kinrara': 3,
    'Bandar Mahkota Cheras': 4,
    'Bandar Menjalara': 4,
    'Bandar Saujana Putra': 3,
    'Bandar Sri Damansara': 1,
    'Bandar Sungai Long': 1,
    'Bandar Sunway': 4,
    'Bandar Tasik Selatan': 1,
    'Bandar Utama': 4,
    'Bangi': 2,
    'Bangsar': 6,
    'Bangsar South': 5,
    'Banting': 1,
    'Batu Caves': 2,
    'Brickfields': 6,
    'Bukit Beruntung': 1,
    'Bukit Bintang': 6,
    'Bukit Jalil': 3,
    'Bukit Jelutong': 5,
    'Bukit Subang': 1,
    'Cheras': 3,
    'City Centre': 6,
    'Cyberjaya': 3,
    'Damansara': 6,
    'Damansara Damai': 2,
    'Damansara Heights': 4,
    'Damansara Perdana': 3,
    'Dengkil': 2,
    'Desa Pandan': 5,
    'Desa ParkCity': 6,
    'Desa Petaling': 2,
    'Glenmarie': 5,
    'Gombak': 4,
    'Hulu Langat': 1,
    'I-City': 4,
    'Jalan Ipoh': 5,
    'Jalan Kuching': 5,
    'Jalan Sultan Ismail': 6,
    'Jenjarom': 2,
    'Jinjang': 1,
    'KL City': 6,
    'KLCC': 6,
    'Kajang': 2,
    'Kapar': 1,
    'Kelana Jaya': 5,
    'Kepong': 3,
    'Keramat': 5,
    'Klang': 3,
    'Kota Damansara': 5,
    'Kota Kemuning': 6,
    'Kuala Langat': 3,
    'Kuala Selangor': 1,
    'Kuchai Lama': 4,
    'Mid Valley City': 6,
    'Mont Kiara': 6,
    'Mutiara Damansara': 2,
    'OUG': 3,
    'Old Klang Road': 5,
    'Others': 3,
    'Pandan Indah': 2,
    'Pandan Jaya': 2,
    'Pandan Perdana': 4,
    'Pantai': 4,
    'Petaling Jaya': 3,
    'Port Klang': 4,
    'Puchong': 2,
    'Puchong South': 2,
    'Pudu': 5,
    'Pulau Indah (Pulau Lumut)': 2,
    'Puncak Alam': 1,
    'Puncak Jalil': 1,
    'Putra Heights': 2,
    'Rawang': 1,
    'Salak Selatan': 1,
    'Salak Tinggi': 1,
    'Saujana Utama': 2,
    'Segambut': 5,
    'Selayang': 3,
    'Semenyih': 1,
    'Sentul': 4,
    'Sepang': 2,
    'Seputeh': 5,
    'Serdang': 1,
    'Serendah': 1,
    'Seri Kembangan': 2,
    'Setapak': 4,
    'Setia Alam': 4,
    'Setiawangsa': 5,
    'Shah Alam': 3,
    'Solaris Dutamas': 5,
    'Sri Damansara': 6,
    'Sri Hartamas': 5,
    'Sri Petaling': 4,
    'Subang Bestari': 2,
    'Subang Jaya': 3,
    'Sungai Besi': 4,
    'Sungai Buloh': 3,
    'Sungai Penchala': 1,
    'Taman Desa': 4,
    'Taman Melawati': 6,
    'Taman Tun Dr Ismail': 6,
    'Telok Panglima Garang': 5,
    'Titiwangsa': 6,
    'USJ': 3,
    'Ulu Klang': 6,
    'Wangsa Maju': 4
}

data = load_data()
label_encoder = LabelEncoder()
label_encoder.fit(data['property_type'])

# Preprocess data and train XGBoost model
def preprocess_and_train_model():
    
    label_encoder = LabelEncoder()
    # Preprocess data
    data['property_type'] = label_encoder.fit_transform(data['property_type'])
    data['region'] = label_encoder.fit_transform(data['region'])

    # Split data into features and target
    X = data.drop(['monthly_rent', 'location'], axis=1)
    y = data['monthly_rent']

    # Train XGBoost model
    model = XGBRegressor()
    model.fit(X, y)

    return model

# Train the model
model = preprocess_and_train_model()

# Rental Price prediction form
with st.form('predict'):
    location = st.selectbox('Location', list(location_bins.keys()))
    property_type = st.selectbox('Property Type',['Apartment', 'Condominium', 'Duplex' ,'Flat', 'Service Residence', 'Studio', 'Townhouse Condo', 'Others'])
    rooms = st.selectbox('Rooms Number',['1', '2','3','4','5','6','7','8','9','10'])
    size = st.number_input('Size (sqft)')
    furnished = st.selectbox('Furnished',list(furnished_bins.keys()))
    region = st.selectbox('Region', list(region_bins.keys()))
    gymnasium = st.radio("Gymnasium", ("Yes", "No"))
    air_cond = st.radio("Air-cond", ("Yes", "No"))
    washing_machine = st.radio("Washing Machine", ("Yes", "No"))
    swimming_pool = st.radio("Swimming Pool", ("Yes", "No"))
    submit = st.form_submit_button('Predict')

if submit:
    property_type = int(label_encoder.transform([property_type])[0])
    rooms = int(rooms)
    gymnasium = map_yes_no_to_binary(gymnasium)
    air_cond = map_yes_no_to_binary(air_cond)
    washing_machine = map_yes_no_to_binary(washing_machine)
    swimming_pool = map_yes_no_to_binary(swimming_pool)
    location_bin = location_bins[location]
    furnished = furnished_bins[furnished]
    region = region_bins[region]
    input_data = [[property_type, rooms, size, furnished, region, gymnasium, air_cond, washing_machine, swimming_pool, location_bin]]
    prediction = model.predict(input_data)
    st.write("Predicted Rental Price:", prediction[0])