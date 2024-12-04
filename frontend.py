import streamlit as st
import pandas as pd
import joblib
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Cohere

# Load the trained model and scaler
model = joblib.load('gradient_boosting_model_optimized.joblib')
scaler = joblib.load('scaler.joblib')

# Define the required input features
required_features = [
    'Dehydration', 'Medicine Overdose', 'Acidious', 'Cold', 'Cough',
    'Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA',
    'Respiratory Rate', 'Oxygen Saturation', 'PH', 'Causes Respiratory Imbalance'
]

# Function to make predictions
def predict_health_condition(input_data):
    # Check if the input data has all required columns
    if not all(feature in input_data.columns for feature in required_features):
        missing_features = list(set(required_features) - set(input_data.columns))
        raise ValueError(f"The uploaded file is missing the following required columns: {', '.join(missing_features)}")
    
    # Filter only the required features
    input_data = input_data[required_features]
    
    # Scale the data
    scaled_data = scaler.transform(input_data)
    
    # Predict using the trained model
    predictions = model.predict(scaled_data)
    return predictions

# Streamlit UI
st.title("Health Condition Prediction System")
st.write("Upload a CSV file containing patient health details to predict their health condition and get medical advice.")

# Input for Cohere API Key
api_key = st.text_input("Enter your Cohere API Key:", type="password")
if not api_key:
    st.warning("Please provide your Cohere API key to proceed.")
    st.stop()

# Initialize the Cohere LLM
llm = Cohere(cohere_api_key=api_key, temperature=0.7, model="command-xlarge")

# Define LangChain prompt template
prompt_template = """
Patient health details:
Dehydration: {Dehydration}
Medicine Overdose: {Medicine Overdose}
Acidious: {Acidious}
Cold: {Cold}
Cough: {Cough}
Temperature: {Temperature}
Heart Rate: {Heart Rate}
Pulse: {Pulse}
BPSYS: {BPSYS}
BPDIA: {BPDIA}
Respiratory Rate: {Respiratory Rate}
Oxygen Saturation: {Oxygen Saturation}
PH: {PH}
Causes Respiratory Imbalance: {Causes Respiratory Imbalance}

Prediction: {Health_Condition}

Based on the above information, provide detailed medical advice for the patient's betterment.
"""
prompt = PromptTemplate(input_variables=[
    "Dehydration", "Medicine Overdose", "Acidious", "Cold", "Cough", 
    "Temperature", "Heart Rate", "Pulse", "BPSYS", "BPDIA", 
    "Respiratory Rate", "Oxygen Saturation", "PH", "Causes Respiratory Imbalance",
    "Health_Condition"], template=prompt_template)

chain = LLMChain(llm=llm, prompt=prompt)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        input_data = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.subheader("Uploaded Data")
        st.write(input_data)
        
        # Make predictions
        predictions = predict_health_condition(input_data)
        
        # Add predictions to the data
        input_data['Health Condition'] = ['Health condition is good' if pred == 0 else 'Health condition is bad' for pred in predictions]
        
        # Display the results and generate medical advice
        st.subheader("Prediction Results and Medical Advice")
        for i, row in input_data.iterrows():
            st.write(f"**Patient {i+1}: {row['Health Condition']}**")
            if row['Health Condition'] == 'Bad':
                # Generate medical advice using LangChain
                advice = chain.run({
                    "Dehydration": row['Dehydration'],
                    "Medicine Overdose": row['Medicine Overdose'],
                    "Acidious": row['Acidious'],
                    "Cold": row['Cold'],
                    "Cough": row['Cough'],
                    "Temperature": row['Temperature'],
                    "Heart Rate": row['Heart Rate'],
                    "Pulse": row['Pulse'],
                    "BPSYS": row['BPSYS'],
                    "BPDIA": row['BPDIA'],
                    "Respiratory Rate": row['Respiratory Rate'],
                    "Oxygen Saturation": row['Oxygen Saturation'],
                    "PH": row['PH'],
                    "Causes Respiratory Imbalance": row['Causes Respiratory Imbalance'],
                    "Health_Condition": row['Health Condition']
                })
                st.write(f"**Medical Advice:** {advice}")
            else:
                st.write("**Medical Advice:** The patient is in good health. Continue maintaining a healthy lifestyle.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
