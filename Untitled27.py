#!/usr/bin/env python
# coding: utf-8

# In[3]:


from autoscraper import AutoScraper
amazon_url="https://www.cricbuzz.com/cricket-series/3130/indian-premier-league-2020/matches"

wanted_list=["MUMBAI INDIANS vs CHENNAI SUPER KINGS, 1st Match","Chennai Super Kings won by 5 wkts"]

scraper=AutoScraper()
result=scraper.build(amazon_url,wanted_list)

print(scraper.get_result_similar(amazon_url,grouped=True))


# In[4]:


scraper.set_rule_aliases({'rule_ounw':'match','rule_0zcd':'Title'})
scraper.keep_rules(['rule_ounw','rule_0zcd'])
scraper.save('cricbuzz_search')


# In[5]:


results=scraper.get_result_similar("https://www.cricbuzz.com/cricket-series/3248/big-bash-league-2020-21/matches",group_by_alias=True)


# In[4]:


results['match'].to_csv('names.csv')


# In[6]:


results


# In[5]:


results['Title']


# In[12]:


import pandas as pd
results=pd.DataFrame.from_dict(results)


# In[13]:


results


# In[14]:


results.to_csv('name.csv')


# In[ ]:




