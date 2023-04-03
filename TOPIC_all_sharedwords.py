#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import re

#!pip install gensim
#!pip install jeiba
#Importing of Genism --Gensim is a free open-source Python library used to represent documents as semantic vectors, as efficiently (computer-wise) and painlessly (human-wise) as possible.
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora,models

#Importing some plotting tools to aid in visualisation
#!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()  # don't skip this

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[3]:


from gensim import corpora, models
# !pip install ckiptagger
# !pip install tensorflow
# !pip install gdown
# !pip install -U ckiptagger[tfgpu,gdown] #------somme trouble with tensorflow
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER   
import os


# In[4]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ws = WS("./ckip/data", disable_cuda=False)
#pos = POS("./ckip/data",disable_cuda=False)
#ner = NER("./ckip/data",disable_cuda=False)


# #### load law data

# In[5]:


import pandas as pd
df_laws = pd.read_csv('./laws/道路交通管理處罰條例33-56.csv')
df_laws


# #### load behavior data

# In[6]:


data_list = []

files = ['702300-嘉雲-107_109(原).csv',         '402200-基宜-107_109(原).csv',         '501800-竹苗-107_109(原).csv',         '802400-屏澎-107_109(原).csv',         '602100-南投-107_109(原).csv',         '602300-彰化-107_109(原).csv',         '402100-花東-107_109(原).csv']

df_behaviors = pd.DataFrame()
useless_columns = ['申請鑑定者','肇事日期','肇事時間',                   '公文編號','公文文號名稱','來文之發文日期',                   '收案日期','國賠案件','道路工程案',                   '現勘後請警方補正警繪圖','函請道路主管機關改善案',                   '警員','現場處理單位','事件編號','院檢裁定',                   '採納情形','判決裁定日期','司法機關','鑑定事項',                   '當事人-性別','當事人-出生日期','當事人-年齡',                   '當事人-汽車駕照種類','當事人-機車駕照種類',                   '當事人-住址','當事人-傷亡','當事人-車損狀況',                   '關係人-姓名','關係人-當事人','關係人-關係',                   '關係人-地址','關係人-性別','證人-姓名',                   '證人-住址','證人-性別','會議編號','會議日期',                   '不鑑定意見','鑑定意見或分析意見','處理狀況說明',                   '處理狀況','鑑定案件受理分類','一般狀況-其他','特別狀況']

for file in files:
    data = pd.read_csv('data/'+file,header=1)
    for col in useless_columns:
        try: data = data.drop(columns=col)
        except: pass
    data = data.drop(data[data['鑑定意見'].str.find('不鑑定')!=-1].index)
    data = data.dropna(subset=['鑑定意見','駕駛行為'])
    data_list.append(data)

list_behavior= []
for data in data_list:
    list_behavior = list_behavior + list(data['駕駛行為'])


# In[7]:


list_behavior[0]


# In[8]:


list_data = []
for case in df_laws['law_content']:
    list_data.append(case)
#print(len(list_data))
for case in list_behavior:
    list_data.append(case)
#print(len(list_data))


# ### CKIP

# In[9]:


list_word = ws(
    list_data
    # sentence_segmentation = True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced
)

#pos_sentence_list = pos(list_behavior)

#entity_sentence_list = ner(word_sentence_list, pos_sentence_list)


# #### shared words

# In[10]:


file = open("shared_words.txt",'r')
data = file.read()

shared_words = data.split('\n')


# In[11]:


# corpus
from gensim import corpora
cases = []
for words in list_word:
    s = [word for word in words if word.strip() in shared_words]
    cases.append(s)
dictionary = corpora.Dictionary(cases)


# In[12]:


# 将文档存入字典，字典有很多功能，比如
# diction.token2id 存放的是单词-id key-value对
# diction.dfs 存放的是单词的出现频率
dictionary.save('./all_topic/dictionary_all_sharedwords.dict')  # store the dictionary, for future reference
corpus = [dictionary.doc2bow(words) for words in list_word]
corpora.MmCorpus.serialize('./all_topic/dictionary_all_sharedwords.mm', corpus)  # store to disk, for later use

#print(dictionary.token2id)


# ### LDA MODEL

# In[13]:


#model
LDA = models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=10,passes = 100)
# for topic in LDA.print_topics(num_words=5):
#     print(topic)
#     print(LDA.inference(corpus))
print(LDA.print_topics())


# In[14]:


def format_topics_sentences(ldamodel=None, corpus=corpus, texts=cases):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=LDA, corpus=corpus, texts=cases)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#df_dominant_topic.insert(0,'law_num',df['law_num'])
df_dominant_topic = df_dominant_topic.drop(['Document_No'],axis = 1)
df_dominant_topic


# In[15]:


df_dominant_topic.to_excel('./all_topic/all_toipic_sharedwords.xlsx')


# In[16]:


sent_topics_outdf_grpd = df_dominant_topic.groupby('Dominant_Topic')
for key, df in sent_topics_outdf_grpd:
    print(key)
    display(df)
    df.to_excel('./all_topic/all_topics__sharedwords_'+str(key)+'.xlsx')


# #### Visualizing keywords for each topic

# In[17]:


# https://guhtac.com/topic-modeling-in-python-discover-how-to-identify-top-n-topics/
# https://blog.csdn.net/qq_39496504/article/details/107125284
vis = pyLDAvis.gensim_models.prepare(LDA, 
                                     corpus, 
                                     dictionary, 
                                     mds="mmds", 
                                     R=20) #This choses the number of word a topic should contain.
vis


# In[18]:


#pyLDAvis.save_html(vis, 'lda.html')
pyLDAvis.save_html(vis, 'lda_sharedwords.html')


# In[20]:


from gensim.test.utils import datapath

#saving model to disk.

#temp_file = datapath("./all_topic/lda_model")
temp_file = datapath("./all_topic/lda_sharedwords_model")


LDA.save(temp_file)


# In[23]:


import pickle
pickle.dump(LDA, open('./all_topic/LDA_model_sharedwords.pkl', 'wb'))


# In[24]:


pickle.dump(LDA, open('./all_topic/LDA_model_sharedwords.sav', 'wb'))


# In[ ]:




