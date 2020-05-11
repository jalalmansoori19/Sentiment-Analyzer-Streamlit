#!/usr/bin/env python
# coding: utf-8

# In[168]:


import streamlit as st
import joblib
import numpy as np
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 


# In[169]:


#Loading Our Pretrained Model 
model= open("Models/Logistic_Reg.pkl", "rb")
log_reg=joblib.load(model)

#Loading Features Bag of words Count Vectorizer
bow_countVec=open("Features/countVectorizer.pkl", "rb")
bow_countVec=joblib.load(bow_countVec)


# In[170]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 


# In[171]:


#Preprocessing function for entered tweet/text
def preprocess(input_txt):
    
    data = input_txt
    df = pd.DataFrame([x.split(';') for x in data.split('\n')])
    df.columns=['Tweet']
    
    #Step 1 : Removing twitter handles 
    df['Tweet'] = np.vectorize(remove_pattern)(df['Tweet'], "@[\w]*")
    
    #Step 2: Removing Punctuations, Numbers, and Special Characters
   
    df['Tweet'] = df['Tweet'].str.replace("[^a-zA-Z#]", " ")
    
    #Step 3 : Removing Short Words having length 3 or less
    df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
    #Tokenization Step
    tokenized_tweet = df['Tweet'].apply(lambda x: x.split())
    
    #Stemming Step exp “ing”, “ly”, “es”, “s” etc) 

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    df['Tweet'] = tokenized_tweet
    
    return df['Tweet']


# In[172]:


#Converting text/tweet into numeric features
def bag_of_words(text, feature):
    
    return feature.transform(text)
    


# In[173]:


st.title("Twitter Sentiment Analysis")
user_input = st.text_input("Enter Tweet")

if st.button("Predict") and user_input != "":
    #PreProcessing
    t=preprocess(user_input)
    
    #Converting into numeric features
    t=bag_of_words(t, bow_countVec)
    
    #Predicting
    prediction = log_reg.predict_proba(t)
    prediction=float(prediction[:,1]*100)
    sentiment='Positive' if prediction >=0.3 else 'Negative'
    
    st.write('\n','\n')
    
    #DISPLAYING MODEL OUTPUT AND PREDICTION PROBABILITY 
    #st.write('Prediction is:**',sentiment,' **with **',prediction,'** confidence')
    st.success('Prediction is : ** {} ** with  {:.3f} % Confidence'.format(sentiment,prediction))
    st.write('\n','\n')


# In[ ]:
#Displaying Images
st.subheader("Visualization and Insights of Twitter Dataset")
#from PIL import Image
#image = Image.open('Pictures/common_words.png')
#st.image(image, caption='Common Words in Twitter Dataset', use_column_width=True)

pics={"Common Words in Twitter Dataset" : "Pictures/common_words.png",
        "Common Words in Positive Tweets" : "Pictures/common_words_Post_tweet.png",
        "Common Words in Negative Tweets ": "Pictures/common_words_Neg_tweet.png",
        "Hashtags Mentioned in Positive Tweets": "Pictures/hash_post.png",
        "Hashtags Mentioned in Negative Tweets" : "Pictures/hash_Neg.png"
}


pic = st.selectbox("Insights choices", list(pics.keys()), 0)
st.image(pics[pic], use_column_width=True, caption=None)



