import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords

import numpy as np
from PIL import Image
import pickle

from streamlit_option_menu import option_menu


def predict_salary(loaded_model, input_features):
    loaded_model_salary = pickle.load(open('linear_regression.pkl', 'rb'))

    try:
        # Convert input features to numeric values
        input_features = [float(feature) for feature in input_features]
    except ValueError:
        return "Please enter valid numeric values for all features."

    input_features = np.array(input_features).reshape(1, -1)

    # Predict the salary using the loaded model
    predicted_salary = loaded_model_salary.predict(input_features)[0]

    return predicted_salary


def diabetes_prediction(input_data):
    loaded_model_diabetes = pickle.load(open('trained_model.sav', 'rb'))

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model_diabetes.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


# Load the trained k-NN model and scaler
#model_knn = joblib.load('admission_probability_model.joblib')
#scaler = joblib.load('scaler.joblib')

# Load the trained model for image compression
model_image_compression = joblib.load('image.joblib')

# Load the trained Decision Tree model
model_decision_tree = joblib.load('decisiontree.joblib')

# Set the NLTK data path
nltk.data.path.append('nltk_data')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the trained model for email detection
model_email_detection = joblib.load('email_detection_model.joblib')


# Function for ML Model Page
 
def ml_model_page():
    st.title("ML Model Page")
    selected_tab = st.selectbox("Select Model", ["Loan Approval Prediction", "Email Spam Detection", "Image Compression", "College Admission Probability", "Diabetes Prediction", "Salary Prediction"])

    if selected_tab == "Loan Approval Prediction":
        st.write("You are on the Loan Approval Prediction page.")
        # Add specific content for Loan Approval Prediction
        #loan approval model
        
        
        # Streamlit UI
        st.title("Loan Approval Prediction")

        # Input form for user to enter data
        gender = st.selectbox("Select Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
        loan_amount_term = st.number_input("Loan Amount Term (Months)", min_value=0, step=1)
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        # Create a DataFrame with the input data
        new_input = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area],
        }

        # Convert categorical variables to numeric
        le = LabelEncoder()
        for column in new_input.keys():
            if isinstance(new_input[column][0], str):  # Check if the value is a string
                new_input[column] = le.fit_transform(new_input[column])

        # Create a DataFrame with the new input data
        new_input_df = pd.DataFrame(new_input)

        # Make predictions using the trained model
        predicted_loan_status = model.predict(new_input_df)

        # Display the predicted loan status
        if st.button("Predict Loan Approval Status"):
            if predicted_loan_status[0] == 1:
                st.success("Congratulations! Your loan is approved.")
            else:
                st.error("Sorry, your loan is not approved.")


    elif selected_tab == "Email Spam Detection":
        st.write("You are on the Email Spam Detection page.")
        # Add specific content for Email Spam Detection
        #email spam prediction model
        
        # Streamlit UI
        st.title("Email Spam Detection")

        # Input for user to enter an email
        user_input = st.text_area("Enter the email text:")

        # Button to trigger prediction
        if st.button("Check Spam"):
            # Make prediction using the loaded model
            prediction = model.predict([user_input])

            # Display result
            if prediction[0] == 1:
                st.error("This email is classified as spam.")
            else:
                st.success("This email is not classified as spam.")


    elif selected_tab == "Image Compression":
        st.write("You are on the Image Compression page.")
        # Add specific content for Image Compression
        #image compression model
        

        # Title
        st.title('Image Compression')

        # User input for image upload
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            # Load and preprocess the image
            image = Image.open(uploaded_file)
            image = np.array(image)
            pixels = image.reshape(-1, 3)

            # Make predictions on the input data
            compressed_pixels = model.cluster_centers_[model.labels_]
            compressed_image = compressed_pixels.reshape(image.shape)

            # Display the original and compressed images
            st.image(image, caption='Original Image', use_column_width=True)
            st.image(compressed_image, caption='Compressed Image', use_column_width=True)


    elif selected_tab == "College Admission Probability":
        st.write("You are on the College Admission Probability page.")
        # Add specific content for College Admission Probability
        #college admission probability model
       
        # Streamlit UI
        st.title("College Admission Probability Prediction")

        # Input for user to enter data
        all_india_rank = st.number_input("Enter All India Rank:", min_value=1, max_value=1000, step=1)
        nirf_ranking = st.number_input("Enter NIRF Ranking:", min_value=1, max_value=30, step=1)
        placement_percentage = st.number_input("Enter Placement Percentage:", min_value=10, max_value=100, step=1)
        median_placement_package = st.number_input("Enter Median Placement Package (in lakhs):", min_value=10, max_value=200, step=1)
        distance_from_college = st.number_input("Enter Distance from College (in km):", min_value=10, max_value=2000, step=10)
        government_funded = st.checkbox("Is the College Government Funded?")
        teachers_qualification = st.number_input("Enter Teachers Qualification (1 to 10):", min_value=1, max_value=10, step=1)
        college_fee = st.number_input("Enter College Fee (in lakhs):", min_value=1, max_value=10, step=1)
        living_facilities = st.number_input("Enter Living Facilities (1 to 10):", min_value=1, max_value=10, step=1)
        girls_boys_ratio_percentage = st.number_input("Enter Girls/Boys Ratio Percentage (1 to 100):", min_value=1, max_value=100, step=1)

       
        # Button to trigger prediction
        if st.button("Percentage Probability to Join the College"):
            # Make predictions using the trained model
            st.write("it will predict soon")
            

    elif selected_tab == "Diabetes Prediction":
        st.write("You are on the Diabetes Prediction page.")
        # Add specific content for Diabetes Prediction
        #diabetes prediction model
        st.title('Diabetes prediction using SVM')

        # getting the input data from user
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose level')
        BloodPressure = st.text_input('Blood pressure value')
        SkinThickness = st.text_input('Skin thickness value')
        Insulin = st.text_input('Insulin level')
        BMI = st.text_input('BMI value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age value')
        
        # code for prediction
        diagnosis = ''
        
        # creating a button for prediction
        if st.button('Diabetes test result'):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                              DiabetesPedigreeFunction, Age])

        
        st.success(diagnosis)

        
        

    elif selected_tab == "Salary Prediction":
        st.write("You are on the Salary Prediction page.")
        # Add specific content for Salary Prediction
        #salary prediction model
        
        loaded_model_salary = pickle.load(open('linear_regression.pkl', 'rb'))

            
        st.title('Salary CTC prediction using multiple linear regression')

        # getting the input data from the user
        grade = st.text_input('Grade (1-10)')
        softskill = st.text_input('Soft skill (1-10)')
        problemsolvingskill = st.text_input('Problem-solving skill (1-10)')
        meditationandyoga = st.text_input('Meditation and Yoga (1-10)')
        discipline = st.text_input('Discipline level (1-10)')
        strongcommandinoneskill = st.text_input('Strong command in one skill (1-10)')

        # code for prediction
        diagnosis = ''

        # creating a button for prediction
        if st.button('Predict Salary CTC in Lacs'):
            new_features = [grade, softskill, problemsolvingskill, meditationandyoga, discipline, strongcommandinoneskill]
            diagnosis = predict_salary(loaded_model_salary, new_features)

        st.success(f'Predicted Salary CTC: {float(diagnosis):.2f} Lacs' if isinstance(diagnosis, (int, float)) else diagnosis)

        

# Function for Feedback Page
def feedback_page():
    st.title("Feedback Page")
    st.write("You can contact me here:")
    st.write("Email - my12345@gmail.com")
    st.write("For face to face meeting - Dalmia hostel, BHU")

    st.write("Or you can submit your feedback here:")
    feedback = st.text_area("Feedback")
    if st.button("Submit Feedback"):
        # Add logic to save feedback
        st.success("Thank you for your valuable time!")

# Function for FLTI Page
def flti_page():
    st.title("FLTI")
    st.write("From Learning to Implementation")
    
# Add authentication logic for the User page
def user_page():
    st.title("User Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Add authentication logic here
        st.success("Login successful!")
        # Add content for authenticated user



# Main function
def main():
    st.set_page_config(page_title="Streamlit Web App", page_icon="🌐", layout="wide")

    # Navigation bar
    navigation_bar = st.sidebar.radio("Navigation", ["FLTI", "ML Model", "User", "Feedback"])

    # Create a horizontal layout
    col1, col2 = st.columns(2)

    with col1:
        if navigation_bar == "FLTI":
            flti_page()
        elif navigation_bar == "ML Model":
            ml_model_page()
        elif navigation_bar == "User":
            user_page()
        elif navigation_bar == "Feedback":
            feedback_page()


if __name__ == "__main__":
    main()
