
# coding: utf-8

# In[12]:

import pandas as pd
import numpy as np
from scipy import stats
import json
from pandas.io.json import json_normalize
import datetime
import pymongo
from pymongo import MongoClient
from collections import OrderedDict, defaultdict


# In[14]:

conn = 'mongodb://system:system@ds127936.mlab.com:27936/heroku_njlc7ffz'
client = pymongo.MongoClient(conn)

db = client.heroku_njlc7ffz
tedtalks = db.tedtalks

df = pd.DataFrame(list(tedtalks.find()))
df.head()


# In[ ]:




# In[15]:

df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d/%m/%Y'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d/%m/%Y'))


# In[ ]:




# In[17]:

pop_talks = df[['title', 'main_speaker', 'views','comments', 'film_date']].sort_values('views', ascending=False)[:50]
pop_talks


# In[ ]:




# In[19]:

comments_talks = df[['title', 'main_speaker', 'views','comments', 'film_date']].sort_values('views', ascending=False)[:20]
comments_talks


# In[ ]:




# In[20]:

import ast
df.iloc[1]['ratings']
df['ratings'] = df['ratings'].apply(lambda x: ast.literal_eval(x))


# In[21]:

df['Funny'] = df['ratings'].apply(lambda x: x[0]['count'])
df['Beautiful'] = df['ratings'].apply(lambda x: x[1]['count'])
df['Ingenious'] = df['ratings'].apply(lambda x: x[2]['count'])
df['Courageous'] = df['ratings'].apply(lambda x: x[3]['count'])
df['Longwinded'] = df['ratings'].apply(lambda x: x[4]['count'])
df['Confusing'] = df['ratings'].apply(lambda x: x[5]['count'])
df['Informative'] = df['ratings'].apply(lambda x: x[6]['count'])
df['Fascinating'] = df['ratings'].apply(lambda x: x[7]['count'])
df['Unconvincing'] = df['ratings'].apply(lambda x: x[8]['count'])
df['Persuasive'] = df['ratings'].apply(lambda x: x[9]['count'])
df['Jawdrop'] = df['ratings'].apply(lambda x: x[10]['count'])
df['Unconvincing'] = df['ratings'].apply(lambda x: x[8]['count'])
df['Persuasive'] = df['ratings'].apply(lambda x: x[9]['count'])
df['Jawdrop'] = df['ratings'].apply(lambda x: x[10]['count'])
df['Ok'] = df['ratings'].apply(lambda x: x[11]['count'])
df['Obnoxious'] = df['ratings'].apply(lambda x: x[12]['count'])
df['Inspiring'] = df['ratings'].apply(lambda x: x[13]['count'])

df.head()


# In[22]:

normalized_rating=df[['name','main_speaker','views','comments','Funny','Beautiful','Ingenious','Courageous','Longwinded','Confusing','Informative',
   'Fascinating','Unconvincing','Persuasive','Jawdrop','Ok','Obnoxious','Inspiring']]
normalized_rating


# In[24]:

normalized_rating.to_csv('assets/data/normalized_rating.csv', date_format='%Y-%m-%d %H:%M:%S')


# In[ ]:




# In[ ]:



