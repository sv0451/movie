#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd


# In[8]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[9]:


movies=movies.merge(credits, on='title')


# In[10]:


movies.head(1)


# In[11]:


#genre
#id
#keywords
#title
#overview
#cast
#crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.head()


# In[13]:


movies.isnull().sum()


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.duplicated().sum()


# In[16]:


movies.iloc[0].genres


# In[17]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','Scifi']


# In[18]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


movies['genres']=movies['genres'].apply(convert)


# In[20]:


movies.head()


# In[21]:


movies['keywords']=movies['keywords'].apply(convert)


# In[22]:


movies.head()


# In[23]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
               L.append(i['name'])
               counter+=1
        else:
            break
    return L
            
     
    


# In[24]:


movies['cast']=movies['cast'].apply(convert3)


# In[25]:


movies.head()


# In[26]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
             L.append(i['name'])
             break      
    return L


# In[27]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[28]:


movies.head()


# In[29]:


movies['overview'][0]


# In[30]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[31]:


movies.head()


# In[32]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[33]:


movies.head()


# In[34]:


movies['tag']=movies['overview']+movies['genres']+movies['cast']+movies['crew']


# In[35]:


movies.head()


# In[36]:


new_df=movies[['movie_id','title','tag']]


# In[37]:


new_df


# In[38]:


new_df['tag']=new_df['tag'].apply(lambda x:" ".join(x))


# In[39]:


new_df.head()


# In[40]:


new_df['tag']=new_df['tag'].apply(lambda x:x.lower())


# In[41]:


new_df.head()


# In[97]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[98]:


vectors=cv.fit_transform(new_df['tag']).toarray()


# In[99]:


vectors


# In[100]:


cv.get_feature_names()


# In[46]:


pip install nltk


# In[47]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[94]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[96]:


new_df['tag']=new_df['tag'].apply(stem)


# In[101]:


from sklearn.metrics.pairwise import cosine_similarity


# In[103]:


similarity=cosine_similarity(vectors)


# In[81]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[104]:


#new_df[new_df['title']=='Batman Begins'].index[0]
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[106]:


recommend('Avatar')


# In[107]:


import pickle


# In[113]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[111]:


new_df['title'].values


# In[112]:


new_df.to_dict()


# In[115]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




