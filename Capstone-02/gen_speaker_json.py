
# coding: utf-8

# In[99]:

import pandas as pd
import numpy as np
from scipy import stats
import json
from pandas.io.json import json_normalize
import datetime
import ast
import pymongo
from pymongo import MongoClient
from collections import OrderedDict, defaultdict


# In[100]:
i=10
conn = 'mongodb://system:system@ds127936.mlab.com:27936/heroku_njlc7ffz'
client = pymongo.MongoClient(conn)

db = client.heroku_njlc7ffz
tedtalks = db.tedtalks_1

df = pd.DataFrame(list(tedtalks.find()))
df.head()


# In[111]:

# import ast
# df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
# df['tags'].head()


# In[112]:

s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'theme'


# In[113]:


theme_df = df.drop('tags', axis=1).join(s)
theme_df.head()


# In[114]:

pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()
pop_themes.columns = ['theme', 'talks']
pop_themes.head(10)


# In[115]:

pop_list=pop_themes['theme'].head(4).tolist()
pop_list


# In[200]:

# pop_theme_talks = theme_df.loc[(theme_df['theme'].isin(pop_themes.head(10)['theme'])) & (theme_df['theme'] != 'TEDx')]
top_tags_df=theme_df.loc[theme_df['theme'].isin(pop_list)]
top_tags_df


# In[201]:

top_technology_df=theme_df.loc[theme_df['theme']=="technology"].sort_values('views', ascending=False)[:i]
top_10_technology=top_technology_df[['main_speaker','speaker_occupation','url','image_url']].reset_index(drop=True)
top_technology=json.dumps(top_10_technology.to_dict('records'))


# In[202]:

top_science_df=theme_df.loc[theme_df['theme']=="science"].sort_values('views', ascending=False)[:i]
top_10_science=top_science_df[['main_speaker','speaker_occupation','url','image_url']].reset_index(drop=True)
top_science=json.dumps(top_10_science.to_dict('records'), indent=4)   


# In[203]:

top_global_df=theme_df.loc[theme_df['theme']=="global issues"].sort_values('views', ascending=False)[:i]
top_10_global=top_global_df[['main_speaker','speaker_occupation','url','image_url']].reset_index(drop=True)
top_global=json.dumps(top_10_global.to_dict('records'))


# In[216]:

top_culture_df=theme_df.loc[theme_df['theme']=="culture"].sort_values('views', ascending=False)[:i]
top_10_culture=top_culture_df[['main_speaker','speaker_occupation','url','image_url']].reset_index(drop=True)
top_culture=json.dumps(top_10_culture.to_dict('records'))


# In[215]:
#outfile="speaker"+ i +".json"
import sys
sys.stdout = open("speakers.json", "w+")
print("{\"main_speaker\": \"TedTalk\", \"speaker_occupation\": \"A trendy talk\", \"url\": \"https://www.ted.com/\",\"image_url\": \"https://cdn0.iconfinder.com/data/icons/circle-icons/512/ted.png\",")
print("\"children\": [")
print("{\"main_speaker\": \"Technology\",\"speaker_occupation\": \"Technology Speakers\",\"url\": \"https://www.ted.com/topics/technology\",")
print("\"children\": "+ top_technology +"},") 
print("{\"main_speaker\": \"Science\",\"speaker_occupation\": \"Science Speakers\",\"url\": \"https://www.ted.com/topics/science\",")
print("\"children\": "+ top_science +"},")
print("{\"main_speaker\": \"Global Issues\",\"speaker_occupation\": \"Global Issues Speakers\",\"url\": \"https://www.ted.com/topics/global+issues\",")
print("\"children\": "+ top_global +"},")
print("{\"main_speaker\": \"Culture\",\"speaker_occupation\": \"Culture Speakers\",\"url\": \"https://www.ted.com/topics/culture\",")
print("\"children\": "+ top_culture +"}]}")

