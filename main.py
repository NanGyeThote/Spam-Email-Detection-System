import streamlit as st
import joblib
import pandas as pd
from io import StringIO
import json
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# ================== Helper Functions ==================

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie animation
lottie_mail = load_lottiefile("lottie/mail.json")

# ================== Page Configuration ==================

# Set up the page configuration
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered",  # Options: 'centered', 'wide'
    initial_sidebar_state="expanded"
)

# Render the Lottie animation
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratios as needed
with col2:
    st_lottie(
        lottie_mail,  # Assuming lottie_mail is your animation data
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=250,
        width=250,
        key="mail_animation"
    )

# Initialize session state for model
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

# Function to load the real or mock model
def load_model_and_vectorizer(model_name):
    try:
        if model_name == "Model 1":
            model = joblib.load("model/spam_modelf.joblib")
            tfidf = joblib.load("model/tfidf_vectorizerf.joblib")
        else:
            # Mock models for Model 2 and Model 3
            model = lambda x: [1] if "spam" in x[0].lower() else [0]  # Mock spam detection
            tfidf = lambda x: np.array(x)  # Mock vectorizer (does nothing)

        st.session_state.current_model = model
        st.session_state.tfidf_vectorizer = tfidf
        st.success(f"{model_name} loaded successfully!")
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")

# ================== Sidebar and Navigation ==================

# Horizontal Menu for navigation
selected2 = option_menu(None, ["Home", "Upload", "About", 'Settings'], 
    icons=['house', 'cloud-upload', "info-circle", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

# ================== Home Page ==================

# Main content area
if selected2 == "Home":
    # Header section
    st.title("📧 Spam Detection System")

    # Model selection dropdown
    model_option = st.selectbox("Select Model", ("Model 1", "Model 2", "Model 3"))

    # Load the selected model dynamically
    if st.button("Load Model", key="load_model_button"):
        load_model_and_vectorizer(model_option)

    # Main content area for text input
    st.markdown("### 🔍 Enter a Message to Detect")

    # Initialize session state for message and result
    if "message" not in st.session_state:
        st.session_state.message = ""
    if "result" not in st.session_state:
        st.session_state.result = None

    # Text input for message
    message = st.text_area(
        "Paste your email or message here:",
        placeholder="Type your message...",
        value=st.session_state.message,
        key="message_input"
    )

    # Button to classify the message
    if st.button("Classify", key="classify_button"):
        if message.strip():
            if st.session_state.current_model and st.session_state.tfidf_vectorizer:
                try:
                    # Mock and real model behavior
                    if model_option == "Model 1":
                        input_tfidf = st.session_state.tfidf_vectorizer.transform([message])
                        prediction = st.session_state.current_model.predict(input_tfidf)[0]
                    else:
                        # Use mock model for Model 2 and Model 3
                        input_mock = [message]  # Mock input for the model
                        prediction = st.session_state.current_model(input_mock)[0]

                    # Interpret result
                    result = "Spam" if prediction == 1 else "Not Spam"
                    st.session_state.result = result

                    # Clear the message in session state (this clears the input field)
                    st.session_state.message = ""  # This will reset the text area input
                except Exception as e:
                    st.error(f"Error during classification: {e}")
            else:
                st.warning("⚠️ Please load a model before classification.")
        else:
            st.warning("⚠️ Please enter a message to classify.")

    # Display the result if available
    if st.session_state.result is not None:
        result = st.session_state.result
        st.markdown(f"### 🛑 Classification Result: **{result}**")
        if result == "Spam":
            st.error("⚠️ This message is classified as **Spam**.")
        else:
            st.success("✅ This message is classified as **Not Spam**.")

# ================== Upload ==================

elif selected2 == "Upload":
    st.markdown("### 📂 Batch Spam Detection")
    
    # Allow the user to select the model they want to use
    model_option = st.selectbox("Select Model", ("Model 1", "Model 2", "Model 3"))
    
    # Load the selected model dynamically when the button is pressed
    if st.button("Load Model"):
        try:
            if model_option == "Model 1":
                model = joblib.load("model/spam_modelf.joblib")
                tfidf = joblib.load("model/tfidf_vectorizerf.joblib")
            elif model_option == "Model 2":
                model = joblib.load("model/spam_model2.joblib")
                tfidf = joblib.load("model/tfidf_vectorizer2.joblib")
            else:
                model = joblib.load("model/spam_model3.joblib")
                tfidf = joblib.load("model/tfidf_vectorizer3.joblib")

            st.session_state.current_model = model
            st.session_state.tfidf_vectorizer = tfidf
            st.success(f"{model_option} loaded successfully!")

        except Exception as e:
            st.error(f"Error loading {model_option}: {e}")

    # Check if model is loaded in session state
    if 'current_model' not in st.session_state or 'tfidf_vectorizer' not in st.session_state:
        st.warning("⚠️ Please load a model before uploading a file.")
    
    else:
        uploaded_file = st.file_uploader("Upload a file (CSV or TXT)", type=["csv", "txt"])

        if uploaded_file:
            st.info(f"Uploaded file: {uploaded_file.name}")
            
            try:
                model = st.session_state.current_model
                tfidf = st.session_state.tfidf_vectorizer

                # Check if the uploaded file is a CSV
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    if df.empty:
                        st.warning("⚠️ The CSV file is empty or not properly formatted.")
                    else:
                        if df.columns[0] != 'Message':
                            df.columns = ['Message']  # Assuming the first column is the message column

                        st.write("**Processing CSV file...**")
                        predictions = model.predict(tfidf.transform(df['Message']))  
                        spam_count = predictions.sum()
                        total_messages = len(df)
                        spam_percentage = (spam_count / total_messages) * 100
                        spam_threshold = 50  # You can adjust this threshold

                        if spam_percentage > spam_threshold:
                            st.markdown(f"### 🛑 The file is classified as **Spam**. ({spam_percentage:.2f}% spam messages)")
                        else:
                            st.markdown(f"### ✅ The file is classified as **Not Spam**. ({spam_percentage:.2f}% spam messages)")
                        
                        # Display summary statistics
                        st.write("**Summary:**")
                        st.write(f"- Total Messages: {total_messages}")
                        st.write(f"- Spam Messages: {spam_count}")

                elif uploaded_file.name.endswith('.txt'):
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    messages = stringio.readlines()
                    df = pd.DataFrame(messages, columns=["Message"])
                    
                    if df.empty:
                        st.warning("⚠️ The TXT file is empty.")
                    else:
                        st.write("**Processing TXT file...**")
                        predictions = model.predict(tfidf.transform(df['Message']))  
                        spam_count = predictions.sum()
                        total_messages = len(df)
                        spam_percentage = (spam_count / total_messages) * 100
                        spam_threshold = 3  # You can adjust this threshold

                        if spam_percentage > spam_threshold:
                            st.markdown(f"### 🛑 The file is classified as **Spam**. ({spam_percentage:.2f}% spam messages)")
                        else:
                            st.markdown(f"### ✅ The file is classified as **Not Spam**. ({spam_percentage:.2f}% spam messages)")
                        
                        # Display summary statistics
                        st.write("**Summary:**")
                        st.write(f"- Total Messages: {total_messages}")
                        st.write(f"- Spam Messages: {spam_count}")

                else:
                    st.error("⚠️ Only CSV and TXT files are supported.")

            except Exception as e:
                st.error(f"Error reading or processing the file: {e}")

# ================== About Page ==================

elif selected2 == "About":
    st.markdown("### ℹ️ About the System")
    st.write("""
        This **Spam Detection System** is designed to identify and classify messages or emails as either **Spam** or **Not Spam**. 
        Leveraging advanced machine learning models, it ensures reliable and accurate detection, helping users manage their communications more effectively.
    """)

    # Key Features
    st.subheader("🔑 Key Features")
    st.write("""
    - **Real-time Spam Detection**: Classify individual messages instantly.
    - **Batch Processing**: Upload files (CSV or TXT) containing multiple messages for bulk classification.
    - **Custom Model Selection**: Choose between different models to evaluate and compare their performance.
    - **Interactive UI**: User-friendly interface for seamless interaction.
    """)

    # System Workflow
    st.subheader("⚙️ System Workflow")
    st.write("""
    1. **Model Selection**: Select a model to evaluate messages.
    2. **Real-time Detection**: Input a message to classify.
    3. **Batch Detection**: Upload CSV or TXT files for bulk classification.
    """)

    # Technical Details
    st.subheader("🛠️ Technical Details")
    st.write("""
    - **Model**: Machine learning model for spam detection.
    - **Vectorizer**: TF-IDF for text preprocessing.
    - **Framework**: Built using **Streamlit** for a responsive experience.
    """)

    # Credits
    st.subheader("👨‍💻 Developer Information")
    st.write("""
    Developed by **Shin Thant Phyo**, a Computer Science student specializing in AI, Web Development, and Software Engineering. 
    For inquiries, feel free to reach out!
    """)

# ================== Settings Page ==================

elif selected2 == "Settings":
    st.markdown("### ⚙️ App Information and Preferences")
    st.write("View app details or reset your preferences.")
    st.subheader("ℹ️ Application Details")
    st.write("""
        - **Version**: 1.0.0  
        - **Developer**: Shin Thant Phyo  
        - **Last Updated**: January 2025
    """)
    if st.button("Reset to Default Settings"):
        st.session_state.clear()
        st.success("All settings have been reset to default!")

# Footer section
st.markdown("---")
st.caption("🔒 Your data is processed securely. | Made with ❤️ using Streamlit")
