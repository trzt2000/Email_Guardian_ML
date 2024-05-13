# %%
import numpy as np
import pandas as pd

# %%
df=pd.read_csv('spam.csv', encoding= 'latin-1')

# %%
df

# %%
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# %%
df

# %%
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.head()

# %%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y='target'

# %%
df['target']=le.fit_transform(df[y])

# %%
df.head()

# %%
df.drop_duplicates(inplace=True)

# %%
df.duplicated().sum()

# %%
df.shape

# %%
#0-ham,1-spam
import seaborn as sns
sns.countplot(x='target',data=df)

# %%
import nltk
nltk.download('punkt')

# %%
df['num_chars']=df['text'].apply(len)

# %%
df

# %%
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

# %%
df.sample(5)

# %%
df['num_sents']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# %%
df

# %%
df.describe()

# %%
df.corr()

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
print(sns.histplot(x='num_chars',hue='target',data=df))


# %%
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# %%
transform_text('Call Germany for only 1 pence per minute! Call from a fixed line via access number 0844 861 85 85. No prepayment. Direct access')

# %%
df['text'].apply(transform_text)

# %%
df['new_text']=df['text'].apply(transform_text)

# %%
df.head()

# %%
from wordcloud import WordCloud

# %%
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
wc_spam=wc.generate(df[df['target']==1]['new_text'].str.cat(sep=' '))

# %%
plt.figure(figsize=(10,12))
plt.imshow(wc_spam)

# %%
wc=WordCloud(width=500,height=500,min_font_size=10)
ham=wc.generate(df[df['target']==0]['new_text'].str.cat(sep=''))
plt.figure(figsize=(10,12))
plt.imshow(ham)


# %%
df[df['target']==0]['new_text'].tolist()
    

# %%
spam=[]
for msg in df[df['target']==0]['new_text'].tolist():
    for word in msg.split():
        spam.append(word)
        

# %%
from collections import Counter
import pandas as pd
pd.DataFrame(Counter(spam).most_common(30))

# %%
ham=[]
for msg in df[df['target']==1]['new_text'].tolist():
    for word in msg.split():
        ham.append(word)

# %%
from collections import Counter
print("Ham:",Counter(ham))


# %%
pd.DataFrame(Counter(ham).most_common(30))


# %%
df

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# %%
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(df['new_text']).toarray()
cv= CountVectorizer()
x1=cv.fit_transform(df['new_text']).toarray()

# %%
print("tfidf:",tfidf.idf_)
print("x1:",x1)


# %%
y=df['target'].values


# %%
y

# %%
from sklearn.model_selection import train_test_split

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y,test_size=0.2,random_state=0)

# %%
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

# %%
gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

# %%
gnb.fit(x_train,y_train)

# %%
y_pred4=gnb.predict(x_test)

# %%
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix

# %%
accuracy_score(y_test,y_pred4)

# %%
precision_score(y_test,y_pred4)

# %%
confusion_matrix(y_test,y_pred4)

# %%
"""
# countvectorizer of gnb
"""

# %%
gnb.fit(x_train1,y_train1)

# %%
y_pred1=gnb.predict(x_test1)

# %%
accuracy_score(y_test1,y_pred1)

# %%
precision_score(y_test1,y_pred1)

# %%
confusion_matrix(y_test1,y_pred1)

# %%
mnb.fit(x_train,y_train)
mnb.fit(x_train1,y_train1)

# %%
y_pred2=mnb.predict(x_test)
y_pred22=mnb.predict(x_test1)

# %%
print("tf:",accuracy_score(y_test,y_pred2))
print("cv:",accuracy_score(y_test1,y_pred22))

# %%
print("p_tf:",precision_score(y_test,y_pred2))
print("p_cv:",precision_score(y_test1,y_pred22))

# %%
print("tf:",confusion_matrix(y_test,y_pred2))
print("cv:",confusion_matrix(y_test1,y_pred22))

# %%
bnb.fit(x_train,y_train)
bnb.fit(x_train1,y_train1)

# %%
y_pred3=bnb.predict(x_test)
y_pred33=bnb.predict(x_test1)

# %%
print("tf:",accuracy_score(y_test,y_pred3))
print("cv:",accuracy_score(y_test1,y_pred33))

# %%
print("tf:",precision_score(y_test,y_pred3))
print("cv:",precision_score(y_test1,y_pred33))

# %%
print("tf:",confusion_matrix(y_test,y_pred3))
print("cv:",confusion_matrix(y_test1,y_pred33))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# %%
svc=SVC(kernel='sigmoid',gamma=1.0)
mnb=MultinomialNB()
ETC=ExtraTreesClassifier(n_estimators=50,random_state=2)
rf=RandomForestClassifier(n_estimators=50,random_state=2)
lg=LogisticRegression(solver='liblinear',penalty='l1')

# %%
def classifier(clf,x_train,x_test,y_train,y_test):
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    return accuracy,precision

# %%
classifier(svc,x_train,x_test,y_train,y_test)

# %%
classifier(mnb,x_train,x_test,y_train,y_test)

# %%
classifier(ETC,x_train,x_test,y_train,y_test)

# %%
classifier(rf,x_train,x_test,y_train,y_test)

# %%
"""
# We will export the "Random Forest" classifier 
"""

# %%
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(rf,open('model.pkl','wb'))