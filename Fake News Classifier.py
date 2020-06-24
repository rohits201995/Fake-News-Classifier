# importing the Dataset
import numpy as np
import pandas as pd
df=pd.read_csv(r'C:\Users\rsroh\Downloads\fake-news\train.csv')
y=df['label']
x=df.dropna()
x=df.drop('label',axis=1)
message=df.copy()
message.reset_index(inplace=True)
message.head(10)

#Data cleaning and preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#nltk.download('stopwords')
corpus = []
for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', str(message['text'][i]))
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
 
# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()




# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))

y_pred=classifier.predict(X_test)





















