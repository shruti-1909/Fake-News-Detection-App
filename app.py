import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import re
import string

# Load the data
data_fake = pd.read_csv('Dataset/Fake.csv')
data_true = pd.read_csv('Dataset/True.csv')

data_fake['class'] = 0
data_true['class'] = 1

# Merge and shuffle data
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

# Preprocess the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize the text
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train the models
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

# Define the output label function
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Define the manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return {
        "Logistic Regression": output_label(pred_LR[0]),
        "Decision Tree": output_label(pred_DT[0]),
        "Gradient Boosting": output_label(pred_GB[0]),
        "Random Forest": output_label(pred_RF[0])
    }

# Streamlit app
st.title("Fake News Detection")

news = st.text_area("Enter the news article text:")

if st.button("Predict"):
    if news:
        predictions = manual_testing(news)
        st.write("Predictions:")
        for model, result in predictions.items():
            st.write(f"{model}: {result}")
    else:
        st.write("Please enter the news article text.")
