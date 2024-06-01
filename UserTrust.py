import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# Set default background image for the sidebar
sidebar_bg_image = "https://img.freepik.com/free-photo/azure-pigment-diffusing-water_23-2147798220.jpg?t=st=1716592208~exp=1716595808~hmac=09e94f3edfbd8e3d3e38238abd10731ac27ca9548f6a9498d67fa4befa68f837&w=740"
sidebar_bg_img = f"""
<style>
[data-testid="stSidebar"] {{
background-image: url("{sidebar_bg_image}");
background-size: cover;
}}
</style>
"""
st.markdown(sidebar_bg_img, unsafe_allow_html=True)

# Set default background image for the main content area
default_bg_image = "https://img.freepik.com/free-photo/azure-pigment-diffusing-water_23-2147798220.jpg?t=st=1716592208~exp=1716595808~hmac=09e94f3edfbd8e3d3e38238abd10731ac27ca9548f6a9498d67fa4befa68f837&w=740"
page_bg_img = f"""
<style>
[data-testid="stApp"] {{
background-image: url("{default_bg_image}");
background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("UserTrust Prediction App")
st.markdown("---")  # horizontal line for spacing

# Load the dataset
#dataset_path = r'C:\Users\TATEND2024\Downloads\project\Consolidated_data.csv'

#dataset = pd.read_csv(dataset_path)

# Load the trained model
#model_path = r'C:\Users\TATEND2024\Downloads\project\UserTPmodel.pkl'
#random_forest_model = joblib.load(model_path)
# Load the dataset
# Load the dataset

##############################################################################################################
import json
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import pandas as pd
import joblib

# Path to your service account key file
SERVICE_ACCOUNT_FILE = 'C:\Users\TATEND2024\Downloads\project\avian-buffer-425114-f2-d272ecfab250'

# Load the credentials from the file
with open(SERVICE_ACCOUNT_FILE) as source:
    info = json.load(source)

# Authenticate using the service account
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/drive']
)

gauth = GoogleAuth()
gauth.credentials = credentials

drive = GoogleDrive(gauth)



# Load the dataset from Google Drive
st.write("Loading dataset from Google Drive...")
dataset_file_id = '1J3Lrp74uIjb0YVpYaiJh70t9xcYDuGX-'
dataset_file = drive.CreateFile({'id': dataset_file_id})
dataset_file.GetContentFile('Consolidated_data.csv')
dataset = pd.read_csv('Consolidated_data.csv')

st.write("Dataset loaded successfully.")
st.dataframe(dataset.head())

# Load the trained model from Google Drive
st.write("Loading model from Google Drive...")
model_file_id = '1_319OL-IjaPIPj88td840i0Sucm8diux'
model_file = drive.CreateFile({'id': model_file_id})
model_file.GetContentFile('UserTPmodel.pkl')
random_forest_model = joblib.load('UserTPmodel.pkl')




###############################################################################################################



features = ['TRUSTEE', 'OBJECT_ID', 'CONTENT_ID', 'SUBJECT_ID', 'RATING', 'POSITIVE_RATINGS_RECEIVED', 'NEGATIVE_RATINGS_RECEIVED']
feature_suggestions = {}

# Generate feature suggestions from the respective data columns of the dataset
for feature in features:
    feature_suggestions[feature] = dataset[feature].unique().tolist()

# Rearrange features for display
features_reordered = [features[0]] + features[1:4] + features[4:7]

# Example prediction on user-entered data
new_data = {}

col1, col2, col3 = st.columns(3)

# Create sidebar menu
st.sidebar.title("Menu")

# Dropdown for suggestive input
enable_suggestions = st.sidebar.checkbox("Enable Suggestive Input")

# Real-time learning
enable_real_time_learning = st.sidebar.checkbox("Enable Real-Time Learning")

# Appearance settings
enable_appearance_settings = st.sidebar.checkbox("Appearance Settings")

# Add a button to compare trustees
#compare_button = st.sidebar.button("Compare Trustees")

#if compare_button:
#    num_trustees = st.sidebar.number_input("Select the number of trustees to compare", min_value=2, step=1)
 #   trustee_ids = st.sidebar.multiselect(
 #       "Select the trustees to compare",
 #       dataset["TRUSTEE"].unique(),
 #       max_selections=int(num_trustees)
#    )

 #   submit_button = st.sidebar.button("Apply")

#    if submit_button:
 #       if len(trustee_ids) >= 2:
 #           most_trusted_trustee = dataset.loc[dataset["TRUSTEE"].isin(trustee_ids)].groupby("TRUSTEE")["POSITIVE_RATINGS_RECEIVED"].sum().idxmax()
 #           st.write(f"The most trusted trustee ID is: {most_trusted_trustee}")
 #       else:
 #           st.write("Please select at least two trustees to compare.")


if enable_appearance_settings:
    # Background image options
    background_options = {
        "Option 1": "https://img.freepik.com/premium-vector/technology-future-polygon-geometric-point-touch-hand_34679-529.jpg?w=740",
        "Option 2": "https://img.freepik.com/free-vector/light-effects-background_2065-27.jpg?t=st=1716552985~exp=1716556585~hmac=090c353ae6a755e9001c21e2e7241aff781d38e3e63b561de1f5ea6cb787056c&w=740",
        "Option 3": "https://img.freepik.com/free-photo/abstract-textured-backgound_1258-30516.jpg?t=st=1716556064~exp=1716559664~hmac=8cd8a82d57b973a5b9e06ea05b5f1134ebeef35cd02e83b97eabd20b484c9fb2&w=740",
        "Option 4": "https://www.twi-global.com/image-library/hero/reportsaccs-istock-663622960.jpg",
    }

    selected_background = st.sidebar.selectbox("Select Background Image", list(background_options.keys()))

    
    background_brightness = st.sidebar.slider("Background Brightness", -2.0, 2.0, -2.0, step=0.1)
    
    page_bg_img = f"""
    <style>
    [data-testid="stSidebar"], [data-testid="stApp"] {{
    background-image: linear-gradient(rgba(255, 255, 255, {background_brightness}), rgba(255, 255, 255, {background_brightness})), url("{background_options[selected_background]}");
    background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def retrain_model(dataset):
    """
    Retrains the model using the provided dataset.
    """
    # Select the relevant columns for prediction
    selected_features = ['TRUSTEE', 'OBJECT_ID', 'CONTENT_ID', 'SUBJECT_ID', 'RATING', 'POSITIVE_RATINGS_RECEIVED', 'NEGATIVE_RATINGS_RECEIVED']
    features = dataset[selected_features]
    target = dataset['LABEL']

    # Splitting the data into training and testing sets
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(train_features, train_target)

    return rf_model

def save_model(model, model_path):
    """
    Saves the retrained model to the specified path.
    """
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

if enable_real_time_learning:
    # Add a button to incorporate new data
    if st.sidebar.button("Incorporate New Data"):
        # Prompt the user to select the desired file
        new_data_file = st.file_uploader("Select the new data file (CSV)", type="csv", accept_multiple_files=False)
        if new_data_file is not None:
            new_data = pd.read_csv(new_data_file)
            # Save the original dataset
            original_dataset = dataset.copy()

            # Append the new data to the existing dataset
            dataset = pd.concat([dataset, new_data], ignore_index=True)

            # Retrain the model with the updated dataset
            model = retrain_model(dataset)

            # Save the updated model
            save_model(model, 'UserTPmodel.pkl')
            st.sidebar.write("New data incorporated and model retrained.")
        else:
            st.sidebar.write("No new data file selected.")

    # Add a button to view the list of uploaded files
    #if st.sidebar.button("View New Uploaded Data Files"):
        # Get a list of all the CSV files that have been uploaded to the app
    #    uploaded_files = [f for f in os.listdir(r'C:\Users\TATEND2024\Downloads\project') if f.endswith('.csv') and os.path.isfile(os.path.join(r'C:\Users\TATEND2024\Downloads\project', f))]
        
        # Display the list of uploaded files
      #  st.sidebar.write("List of uploaded data files:")
       # for file in uploaded_files:
       #     st.sidebar.write(f"- {file}")
    if st.sidebar.button("Rollback New Data"):
       if 'original_dataset' in locals():
        rollback_option = st.sidebar.radio("Rollback to:", ["Recent Upload", "Original"])
        if rollback_option == "Recent Upload":
            # Restore the recent dataset
            dataset = original_dataset.copy()
            # Retrain the model with the recent dataset
            model = retrain_model(dataset)
            # Save the updated model
            save_model(model, 'UserTPmodel.pkl')
            st.sidebar.write("New data rolled back and model retrained.")
        elif rollback_option == "Original":
            # Restore the original dataset
            dataset = original_dataset.copy()
            # Retrain the model with the original dataset
            model = retrain_model(dataset)
            # Save the updated model
            save_model(model, 'UserTPmodel.pkl')
            st.sidebar.write("Rolled back to original data and model retrained.")
    #else:
        st.sidebar.write("No new data to rollback.")

# About section
st.sidebar.title("About")
st.sidebar.write("This app predicts UserTrust based on input features.")

for i, feature in enumerate(features_reordered):
    if enable_suggestions:
        value = col1.selectbox(f"Enter {feature}: ", feature_suggestions.get(feature), key=f"input_{i}")
    else:
        value = col1.text_input(f"Enter {feature}: ", key=f"input_{i}")

    if value == '':
        value = None
    elif isinstance(value, str):
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
    elif isinstance(value, float) and value.is_integer():
        value = int(value)

    new_data[feature] = [value]

    if (i + 1) % 3 == 0:
        col1, col2, col3 = col2, col3, st.columns(3)

submit_button = st.button("Submit")

if submit_button:
    if not any(new_data.values()):
        st.write("No input has been provided.")
    elif any(feature not in new_data or new_data[feature][0] is None for feature in features):
        st.write("Please provide input for all features.")
    else:
        # Perform any necessary preprocessing on the new data
        # ...
        # Make the prediction using the trained model
        prediction = random_forest_model.predict(pd.DataFrame(new_data))
        st.write("UserTrust Prediction:", prediction[0])
