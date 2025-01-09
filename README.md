# Spam Detection AI Streamlit Application

**Prepared by**: Shin Thant Phyo  
**Institution**: -  
**Date**: January 2025

## Project Overview

This project is a **Spam Detection AI** application built with **Streamlit**. It is designed to classify text messages as either **spam** or **not spam (ham)**. The system leverages machine learning models to predict the category of the input message, using algorithms such as **Random Forest**, **Logistic Regression** and **Support Vector Machine (SVM)**.

The web application allows users to:
- Input messages directly to be classified.
- Upload `.txt` files to analyze their content.
- View real-time classification results, including spam prediction with a clear, user-friendly interface.

## Features

- **Text Input**: Users can enter a text message, and the system will classify it as spam or ham.
- **File Upload**: Users can upload `.txt` files, and the application will classify the message inside the file as spam or ham.
- **Real-Time Prediction**: The application processes the input in real-time and displays the result instantly.

### Application Screenshot:
![Application Screenshot](images/img3.png) <!-- Replace with actual screenshot path -->

## Dataset

The **SMS Spam Collection Dataset** was used for training the machine learning models.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Description**: A collection of 5,574 labeled SMS messages, with categories "ham" (non-spam) and "spam."
- **Spam Messages**: 747
- **Ham Messages**: 4,827

This dataset is highly suitable for spam detection because it contains a diverse set of messages in English.

### Dataset Statistics:
```bash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Features and labels
X = data['message']
y = data['label']

# Text to numeric vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
```

## Libraries Used

The following libraries were used to implement this project:

- **Streamlit**: For creating the interactive web interface.
- **scikit-learn**: For building the machine learning models (Random Forest, SVM).
- **pandas**: For data manipulation and processing.
- **numpy**: For numerical operations.
- **pickle**: For serializing and deserializing the trained model.

## How to Run the Project

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/spam-detection-streamlit.git
    cd spam-detection-streamlit
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

4. Open your browser and navigate to:  
   [http://localhost:8501](http://localhost:8501)

5. **Use the features**:
    - **Text Input Box**: Type or paste a message to be classified as spam or ham.
    - **File Upload Button**: Upload a `.txt` file to classify its contents.

### Additional Screenshot:
1. **Home**
![Model Prediction](images/img1.png)  <!-- Replace with actual model prediction image -->
2. **Batch**
![Model Prediction](images/img2.png) 


## How It Works

The system uses the **SMS Spam Collection Dataset** to train a machine learning model to classify SMS messages as spam or ham. The machine learning pipeline consists of the following steps:

1. **Text Preprocessing**: 
    - Tokenization: Breaking down the message into individual words.
    - Vectorization: Converting text into numerical form using `TfidfVectorizer`.

2. **Model Training**: 
    - The model is trained using algorithms like **Random Forest**,**Logistic Regression** or **Support Vector Machine (SVM)**.

3. **Prediction**: 
    - Once trained, the model can classify unseen messages as spam or ham.

### Model Architecture Diagram:
1. **Random Forest**
![Model Architecture](images/rfr.jpg)  <!-- Replace with actual model architecture diagram -->

2. **Logistic Regression**
![Model Architecture](images/lrg.png)  <!-- Replace with actual model architecture diagram -->

3. **SVM**
![Model Architecture](images/svm.jpg)  <!-- Replace with actual model architecture diagram -->

## Evaluation Metrics

The following metrics were used to evaluate the model performance:

- **Accuracy**: The percentage of correctly classified messages.
- **Precision**: The percentage of spam messages that were correctly identified.
- **Recall**: The percentage of actual spam messages that were correctly detected.
- **F1-Score**: The harmonic mean of precision and recall.

### Model Evaluation Graph:
![Model Evaluation Graph](images/eva.png)  <!-- Replace with actual evaluation graph -->

## Example Usage

### Text Input Example

**Input**: "Congratulations, you've won a free gift card!"  
**Output**: Spam

### File Upload Example

**Input**: Upload a `.txt` file containing the message: "This is an urgent message to claim your reward!"  
**Output**: Spam

## Contributing

Feel free to fork this repository and contribute! Open issues and submit pull requests for any bug fixes, enhancements, or features you'd like to add.

## Challenges and Solutions

### Challenges:
- **Handling Imbalanced Datasets**: Spam messages typically outnumber non-spam messages, leading to potential bias in classification.
- **Model Selection**: Choosing the right model that ensures both high recall (minimizing false negatives) and precision (minimizing false positives).

### Solutions:
- **Oversampling**: Applied techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Model Comparison**: Experimented with multiple models (Logistic Regression, Naive Bayes, SVM) to find the optimal balance of accuracy, speed, and precision.

## Future Improvements:
- **Enhanced Models**: Integrate advanced models, such as transformers, for better performance on complex datasets.
- **Multilingual Support**: Extend the system to handle multiple languages, expanding its use beyond English.
- **User Interface Enhancements**: Add support for dynamic light/dark modes to improve the user experience.

---

## Appendices

### A. Dataset Description
- **Dataset Name**: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- **Description**: This dataset contains a collection of SMS messages labeled as either "ham" (non-spam) or "spam." It consists of 5,574 messages in English, making it suitable for binary text classification tasks.
  - Total Messages: 5,574
  - Spam Messages: 747
  - Ham Messages: 4,827

### B. Libraries Used
1. **scikit-learn**:
   - **Purpose**: For machine learning model training, evaluation, and metrics calculation.
   - **Functions used**:
     - RandomForestClassifier, TfidfVectorizer, train_test_split, etc.
   - Documentation: [scikit-learn Documentation](https://scikit-learn.org/stable/)
   
2. **Streamlit**:
   - **Purpose**: To develop the web interface for the spam detection system.
   - **Features used**:
     - Displaying input forms, rendering model predictions, and visualizing results.
   - Documentation: [Streamlit Documentation](https://docs.streamlit.io/)

### C. Additional Resources
- **Other datasets explored (not used in the project)**:
  1. [Spam Text Message Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset): Contains 5,572 SMS messages formatted for easier Kaggle integration.
  2. [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset): A collection of emails for spam classification and other NLP tasks.

---

## References

### Dataset:
- **UCI Machine Learning Repository** (SMS Spam Collection Dataset) - [Link](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

### Libraries:
1. **scikit-learn Documentation** - [Link](https://scikit-learn.org/stable/)
2. **Streamlit Documentation** - [Link](https://docs.streamlit.io/)

### Research and Articles:
1. Almeida, T. A., Hidalgo, J. M., & Yama kami, A. (2011). "Contributions to the study of SMS spam filtering: New collection and results."
2. Zhang, L., Zhu, J., & Yao, T. (2004). "An evaluation of statistical spam filtering techniques."
3. **Kaggle Competitions and Notebooks**:
   - Explore Kaggle notebooks on spam detection for insights on workflows and alternative approaches.

---

## AI Ethics

### Ethical Considerations:
As artificial intelligence (AI) technologies such as spam detection systems become more prevalent, it is essential to address ethical concerns to ensure the responsible and fair application of AI. Some of the key ethical issues in spam detection are:

- **Bias and Fairness**: AI models can inherit biases present in the data they are trained on. This project has taken steps to address imbalances in the dataset by using oversampling techniques, but further work can be done to ensure fairness in classification across different types of messages.
  
- **Privacy and Data Security**: Spam detection systems analyze potentially sensitive text data. It is crucial to implement robust data privacy and security measures to protect users' information.

- **Transparency and Accountability**: Users should be able to understand how AI systems classify messages as spam or not. This project aims to provide clarity on model behavior through detailed results and explanations.

- **Continuous Monitoring and Improvement**: Spam tactics evolve over time, and AI models should be continuously updated to adapt to new patterns. The future improvement of this project includes regular updates to the model to handle evolving spam tactics.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.
- The **Streamlit** library for easy web development.
- **scikit-learn** for providing machine learning models.
