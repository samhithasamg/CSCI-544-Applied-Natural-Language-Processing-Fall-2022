#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")


# ## Read Data

# In[2]:


raw_data=pd.read_csv("amazon_reviews_us_Jewelry_v1_00.tsv",sep="\t",on_bad_lines='skip')


# ## Keep Reviews and Ratings

# In[3]:


data=raw_data[['star_rating','review_body']]


# In[4]:


data=data.dropna()
data = data.reset_index(drop=True)
data['star_rating']=data['star_rating'].astype(int)


#  ## We select 20000 reviews randomly from each rating class.
# 
# 

# In[5]:


df=data.groupby('star_rating').sample(n=20000)


# # Data Cleaning
# 
# 

# In[6]:


cnt_before=(df['review_body'].str.len()).mean()
#print("Average review_body character count before  Datcleaning:"+ str(str_counts.mean()))


# In[7]:


df['review_body']=df['review_body'].str.lower()  #convert to lower case

#remove contractions
df['review_body']=df['review_body'].str.replace("\'re"," are")
df['review_body']=df['review_body'].str.replace("br"," ")
df['review_body']=df['review_body'].str.replace("\n't"," not")
df['review_body']=df['review_body'].str.replace("\'s'"," is")
df['review_body']=df['review_body'].str.replace("i'm"," i am")
df['review_body']=df['review_body'].str.replace("\'ve'"," have")
df['review_body']=df['review_body'].str.replace("din't","did not")

df['review_body']=df['review_body'].str.replace('http\S+|www.\S+', '', case=False)
df['review_body']=df['review_body'].str.replace('[^a-zA-Z0-9 ]', '')
df['review_body']=df['review_body'].str.replace('/ +/', ' ')#convert multispace to space
df['review_body']=df['review_body'].str.replace('/^ /', '') # remove spaces in the start of the string
df['review_body']=df['review_body'].str.replace('/ $/', '') # remove unnecesary space at the end of the string


# In[8]:


cnt_after=(df['review_body'].str.len()).mean()
print(str(cnt_before)+","+str(cnt_after))
#print("Average review_body character count after Data cleaning:"+ str(str_counts.mean()))


# # Pre-processing

# In[ ]:


cnt_before=(df['review_body'].str.len()).mean()
#print("Average review_body character count before preprocessing:"+ str(str_counts.mean()))


# ## remove the stop words 

# In[ ]:


from nltk.corpus import stopwords
 
stop_words=stopwords.words('english')
df['review_body']=df['review_body'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop_words)]))


# ## perform lemmatization  

# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()
df['review_body']=df['review_body'].apply(lambda x:' '.join([lemmatizer.lemmatize(word) for word in str(x).split()]))


# In[ ]:


cnt_after=(df['review_body'].str.len()).mean()
#print("Average review_body character count after preprocessing:"+ str(str_counts.mean()))
print(str(cnt_before)+","+str(cnt_after))


# # TF-IDF Feature Extraction

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer()
X=df['review_body']
X=vectorizer.fit_transform(X)
Y=df['star_rating']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=30)


# # Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron
clf_perceptron = Perceptron()
clf_perceptron.fit(X_train, Y_train)


# In[ ]:


y_test_perceptron=clf_perceptron.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_perceptron,output_dict=True)


# In[ ]:


print(str(report['1']['precision'])+","+str(report['1']['recall'])+","+str(report['1']['f1-score']))
print(str(report['2']['precision'])+","+str(report['2']['recall'])+","+str(report['2']['f1-score']))
print(str(report['3']['precision'])+","+str(report['3']['recall'])+","+str(report['3']['f1-score']))
print(str(report['4']['precision'])+","+str(report['4']['recall'])+","+str(report['4']['f1-score']))
print(str(report['5']['precision'])+","+str(report['5']['recall'])+","+str(report['5']['f1-score']))
print(str(report['weighted avg']['precision'])+","+str(report['weighted avg']['recall'])+","+str(report['weighted avg']['f1-score']))


# # SVM

# In[ ]:


from sklearn.svm import LinearSVC
clf_SVM = LinearSVC()
clf_SVM.fit(X_train, Y_train)
y_test_SVM=clf_SVM.predict(X_test)


# In[ ]:


report_SVM=classification_report(Y_test, y_test_SVM,output_dict=True)


# In[ ]:


print(str(report_SVM['1']['precision'])+","+str(report_SVM['1']['recall'])+","+str(report_SVM['1']['f1-score']))
print(str(report_SVM['2']['precision'])+","+str(report_SVM['2']['recall'])+","+str(report_SVM['2']['f1-score']))
print(str(report_SVM['3']['precision'])+","+str(report_SVM['3']['recall'])+","+str(report_SVM['3']['f1-score']))
print(str(report_SVM['4']['precision'])+","+str(report_SVM['4']['recall'])+","+str(report_SVM['4']['f1-score']))
print(str(report_SVM['5']['precision'])+","+str(report_SVM['5']['recall'])+","+str(report_SVM['5']['f1-score']))
print(str(report_SVM['weighted avg']['precision'])+","+str(report_SVM['weighted avg']['recall'])+","+str(report_SVM['weighted avg']['f1-score']))


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(max_iter=1000,solver='lbfgs')
clf_logistic.fit(X_train,Y_train)
y_test_logistic=clf_logistic.predict(X_test)


# In[ ]:


report_logistic=classification_report(Y_test, y_test_logistic,output_dict=True)


# In[ ]:


print(str(report_logistic['1']['precision'])+","+str(report_logistic['1']['recall'])+","+str(report_logistic['1']['f1-score']))
print(str(report_logistic['2']['precision'])+","+str(report_logistic['2']['recall'])+","+str(report_logistic['2']['f1-score']))
print(str(report_logistic['3']['precision'])+","+str(report_logistic['3']['recall'])+","+str(report_logistic['3']['f1-score']))
print(str(report_logistic['4']['precision'])+","+str(report_logistic['4']['recall'])+","+str(report_logistic['4']['f1-score']))
print(str(report_logistic['5']['precision'])+","+str(report_logistic['5']['recall'])+","+str(report_logistic['5']['f1-score']))
print(str(report_logistic['weighted avg']['precision'])+","+str(report_logistic['weighted avg']['recall'])+","+str(report_logistic['weighted avg']['f1-score']))


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB()
clf_NB.fit(X_train,Y_train)
y_test_NB= clf_NB.predict(X_test)


# In[ ]:


report_NB=classification_report(Y_test, y_test_NB,output_dict=True)


# In[ ]:


print(str(report_NB['1']['precision'])+","+str(report_NB['1']['recall'])+","+str(report_NB['1']['f1-score']))
print(str(report_NB['2']['precision'])+","+str(report_NB['2']['recall'])+","+str(report_NB['2']['f1-score']))
print(str(report_NB['3']['precision'])+","+str(report_NB['3']['recall'])+","+str(report_NB['3']['f1-score']))
print(str(report_NB['4']['precision'])+","+str(report_NB['4']['recall'])+","+str(report_NB['4']['f1-score']))
print(str(report_NB['5']['precision'])+","+str(report_NB['5']['recall'])+","+str(report_NB['5']['f1-score']))
print(str(report_NB['weighted avg']['precision'])+","+str(report_NB['weighted avg']['recall'])+","+str(report_NB['weighted avg']['f1-score']))

