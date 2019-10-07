
# coding: utf-8

# In[240]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pandas import read_csv
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)


# https://www.kaggle.com/faizunnabi/autism-screening
# 
# Attribute Information:
# 
# Attribute Type Description
# 
# Age Number Age in years
# 
# Gender String Male or Female
# 
# Ethnicity String List of common ethnicities in text format
# 
# Born with jaundice Boolean (yes or no) Whether the case was born with jaundice
# 
# Family member with PDD Boolean (yes or no) Whether any immediate family member has a PDD
# 
# Who is completing the test String Parent, self, caregiver, medical staff, clinician ,etc.
# 
# Country of residence String List of countries in text format
# 
# Used the screening app before Boolean (yes or no) Whether the user has used a screening app
# 
# Screening Method Type Integer (0,1,2,3) The type of screening methods chosen based on age category (0=toddler, 1=child, 2= adolescent, 3= adult)
# 
# Question 1 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 2 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 3 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 4 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 5 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 6 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 7 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 8 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 9 Answer Binary (0, 1) The answer code of the question based on the screening method used
# 
# Question 10 Answer Binary (0, 1) The answer code of the question based on the screening method used

# In[241]:


autism = pd.read_csv("Autism_Data.csv")


# In[242]:


autism.tail()


# In[243]:


autism.head()


# In[244]:


has_autism = autism['Class/ASD'].value_counts()


# In[245]:


has_autism


# In[246]:


has_autism.plot.bar()


# In[247]:


autism_filter = ((autism["Class/ASD"] == "YES"))
has_autism = autism[autism_filter]
has_autism.head()


# In[248]:


asd_ethnicity = has_autism["ethnicity"].value_counts()


# In[249]:


#determines whether patient with autism has parent(s) with autism
asd_genetic = has_autism["austim"].value_counts()
asd_genetic


# In[250]:


#checking if a parent with autism can pass down the trait to the next generation
autism_parent_filter = (autism["austim"] == "yes")
parents_have_autism = autism[autism_parent_filter]
parents_have_autism


# In[251]:


#checking if a parent with autism can pass down the trait to the next generation
graph_parents_with_autism = parents_have_autism['Class/ASD'].value_counts()
graph_parents_with_autism.plot.bar()
plt.ylabel('# of surveyed parents having autism')
plt.title('Parents with autism who have child(ren) with autism')


# In[252]:


asd_genetic.plot.bar()
plt.ylabel('# of surveyed people having autism')
plt.title('Having parent(s) that have autism')


# In[253]:


asd_gender = has_autism["gender"].value_counts()
asd_gender
asd_gender.plot.bar()
plt.ylabel('# of surveyed people having autism')
plt.title('Gender that was surveyed')


# In[254]:


asd_severity = autism["result"].value_counts()
asd_severity
asd_severity.plot.bar()
plt.xlabel('Score (A score of 7-10 = the person has autism)')
plt.ylabel('# of surveyed people having autism')
plt.title('Severity Score')


# In[255]:


asd_relation = has_autism["relation"].value_counts()
asd_relation
asd_relation.plot.bar()
plt.ylabel('# of surveyed people having autism')
plt.title('Group of person who did the survey')


# In[256]:


asd_relation = autism["relation"].value_counts()
asd_relation
asd_relation.plot.bar()
plt.ylabel('# of surveyed people')
plt.title('Group of person who did the survey')


# In[257]:


whole_ethnicity = autism["ethnicity"].value_counts()


# In[258]:


((asd_ethnicity/whole_ethnicity)*100).plot.bar()


# In[259]:


asd_coutry = has_autism["contry_of_res"].value_counts()
non_asd_coutry = autism["contry_of_res"].value_counts()


# In[260]:


asd_coutry_ratio = asd_coutry/non_asd_coutry
asd_coutry_ratio.plot.bar()


# In[261]:


asd_coutry


# In[262]:


#the whole sample of people who have been surveyed, in country counts
non_asd_coutry


# In[263]:


#non_asd_coutry_above_five = autism(autism[non_asd_coutry] >= 5)


# In[264]:


#non_asd_coutry_above_five.to_csv("non_asd_coutry_above_five.csv")


# In[265]:


#above_five = pd.read_csv("non_asd_coutry_above_five.csv")
#above_five


# In[266]:


#autism = pd.DataFrame({"contry_of_res": [5,6,7,8,9,10]})
#print(autism, "\n") 


# In[267]:


asd_jundice = has_autism["jundice"].value_counts()
non_asd_jundice = autism["jundice"].value_counts()
asd_jundice.plot.bar()


# In[268]:


non_asd_jundice.plot.bar()


# In[269]:


autism.describe()


# In[270]:


autism

