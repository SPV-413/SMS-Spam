# SMS Spam Detection
This project focuses on predicting SMS Spam using machine learning model trained on the **spam**. The dataset contains **label, message, & some unwanted columns**. SMS spam, unsolicited and often malicious text messages, overwhelms users, leading to annoyance, potential fraud (phishing), and security risks. The objective is to develop a machine learning model that accurately classifies incoming SMS messages as either **spam or ham** (legitimate), thereby filtering out unwanted messages and protecting users.
## Dataset Overview
- **Source**: spam
- **Rows**: 5572
- **Columns**: 5 (including the target variable)
- **Target Variable**:
  - label:
     - `ham` - No spam message present
     - `spam` - Spam message present
- **Features**:
  - message
  - Unnamed: 2
  - Unnamed: 3
  - Unnamed: 4
    
## Project Workflow
### 1. **Data Preprocessing**
- Checking missing values
- Removing unwanted columns
- Removing duplicates
- Tokenization
- Lowercasing
- Stopword Removal
- Lemmatization
- Removing Punctuation & Special Characters
- Word Embedding (Vectorization):
  Applied & verified individually
  - Bag of Words (BoW) – Counts word occurrences
  - TF-IDF (Term Frequency-Inverse Document Frequency) – Measures word importance
  - Word2Vec – to create dense word vectors
- Encoding target feature(`OneHotEncoder`)
- Splitting dataset into training & testing sets (`train_test_split`)
  
### 2. **Machine Learning Model**
The project implements machine learning model for classification:
- **BernoulliNB(Naive-Bayes)**

### 3. **Model Evaluation**
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**

## Installation & Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.10 and above
- Jupyter Notebook
- Required libraries (`pandas`, `numpy`, `scikit-learn`, `nltk`)

### Running the Notebook
1. Clone the repository:
   git clone https://github.com/SPV-413/SMS-Spam-Detection.git
2. Navigate to the project folder:
   cd SMS-Spam-Detection
3. Open the Jupyter Notebook:
   jupyter notebook SMS Spam.ipynb & Spam.ipynb

## Results
- Using both **Bag of Words (BoW) and TF-IDF** for feature extraction resulted in the same highest accuracy score when combined with the **Bernoulli Naive Bayes** model. This suggests that both methods effectively captured the relevant information in the dataset, highlighting their robustness in certain contexts. However, Word2Vec, which generates dense word vectors, did not improve the accuracy in this specific scenario. Thereby, filtering out unwanted messages and protecting users.

## Contact
For any inquiries, reach out via GitHub or email.
