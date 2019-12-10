
# coding: utf-8

# In[1297]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
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

# In[1298]:


autism = pd.read_csv("Autism_Data.csv", na_values = ["?", 383])
autism.dropna()


# In[1299]:


autism.tail()


# In[1300]:


autism.head()


# In[1301]:


has_autism = autism['Class/ASD'].value_counts()


# In[1302]:


has_autism


# In[1303]:


has_autism.plot.bar()


# In[1304]:


autism_filter = ((autism["Class/ASD"] == "YES"))
has_autism = autism[autism_filter]
has_autism.head()


# In[1305]:


asd_ethnicity = has_autism["ethnicity"].value_counts()


# In[1306]:


#determines whether patient with autism has parent(s) with autism
asd_genetic = has_autism["austim"].value_counts()
asd_genetic


# In[1307]:


#checking if a parent with autism can pass down the trait to the next generation
autism_parent_filter = (autism["austim"] == "yes")
parents_have_autism = autism[autism_parent_filter]
parents_have_autism


# In[1308]:


#checking if a parent with autism can pass down the trait to the next generation
graph_parents_with_autism = parents_have_autism['Class/ASD'].value_counts()
graph_parents_with_autism.plot.bar()
plt.ylabel('# of surveyed parents having autism')
plt.title('Parents with autism who have child(ren) with autism')


# In[1309]:


asd_genetic.plot.bar()
plt.ylabel('# of surveyed people having autism')
plt.title('Having parent(s) that have autism')


# In[1310]:


asd_gender = has_autism["gender"].value_counts()
asd_gender
asd_gender.plot.bar()
plt.ylabel('# of surveyed people having autism')
plt.title('Gender that was surveyed')


# In[1311]:


asd_severity = autism["result"].value_counts()
asd_severity
asd_severity.plot.bar()
plt.xlabel('Score (A score of 7-10 = the person has autism)')
plt.ylabel('# of surveyed people having autism')
plt.title('Severity Score')


# In[1312]:


asd_relation = has_autism["relation"].value_counts()
asd_relation
asd_relation.plot.bar()
plt.ylabel('# of surveyed people having autism')
plt.title('Group of person who did the survey')


# In[1313]:


asd_relation = autism["relation"].value_counts()
asd_relation
asd_relation.plot.bar()
plt.ylabel('# of surveyed people')
plt.title('Group of person who did the survey')


# In[1314]:


whole_ethnicity = autism["ethnicity"].value_counts()


# In[1315]:


((asd_ethnicity/whole_ethnicity)*100).plot.bar()


# In[1316]:


asd_coutry = has_autism["contry_of_res"].value_counts()
non_asd_coutry = autism["contry_of_res"].value_counts()


# In[1317]:


asd_coutry_ratio = asd_coutry/non_asd_coutry
asd_coutry_ratio.plot.bar()


# In[1318]:


asd_coutry


# In[1319]:


#the whole sample of people who have been surveyed, in country counts
non_asd_coutry


# In[1320]:


autism.age.dtype


# In[1321]:


asd_jundice = has_autism["jundice"].value_counts()
non_asd_jundice = autism["jundice"].value_counts()
asd_jundice.plot.bar()


# In[1322]:


non_asd_jundice.plot.bar()


# In[1323]:


autism.describe()


# In[1324]:


autism


# In[1325]:


pd.Series(autism["result"]).hist(range = [0,10], bins = 10)
plt.title("# of people who fall under particular score (7-10 = having autism)")
plt.xlabel("Score")
plt.ylabel("# of people who fall under particular score")


# In[1326]:


autism["age"].hist(range = [0, 100], bins = 20)
plt.title("Ages of people being surveyed")
plt.xlabel("Age Range")
plt.ylabel("# of people who fall under age range")


# In[1327]:


age_value_counts = autism["age"].value_counts()
age_value_counts


# In[1328]:


autism_20_30_filter = (autism["age"] >= 20) & (autism["age"] <= 30)
autism_20_30_filter = autism[autism_20_30_filter]
autism_20_30_filter


# In[1329]:


autism_50_60_filter = (autism["age"] >= 50) & (autism["age"] <= 60)
autism_50_60_filter = autism[autism_50_60_filter]
autism_50_60_filter


# In[1330]:


pd.Series(autism_20_30_filter["result"]).hist(range = [0,10], bins = 10)
plt.title("# of people ages 20-30 who fall under particular score")
plt.xlabel("Score")
plt.ylabel("# of people ages 20-30 who fall under particular score")


# In[1331]:


pd.Series(autism_50_60_filter["result"]).hist(range = [0,10], bins = 10)
plt.title("# of people ages 20-30 who fall under particular score")
plt.xlabel("Score")
plt.ylabel("# of people ages 20-30 who fall under particular score")


# In[1332]:


has_autism


# In[1333]:


asd = autism[autism["austim"] == "yes"]
asd


# In[1334]:


autism["age"].hist(bins = 20)


# In[1335]:


plt.scatter(x = 'age', y = "result", data = autism_20_30_filter)


# In[1336]:


age_result_counts = autism["age"]
age_result_counts


# In[1337]:


result_mean = autism["result"].mean()
result_mean


# In[1338]:


result_asd_age = asd["age"].value_counts()
result_asd_age


# In[1339]:


result_asd_age_pct = 100*((asd["age"].value_counts())/(age_result_counts))
result_asd_age_pct


# In[1340]:


result_asd_age_pct = pd.DataFrame(result_asd_age_pct)
result_asd_age_pct


# In[1341]:


sns.relplot(y="result", x="age", data=autism, kind="scatter")


# In[1342]:


sns.relplot(data=result_asd_age_pct, kind="scatter")


# In[1343]:


sns.regplot(data=result_asd_age_pct)


# In[1346]:


sns.regplot(y="result", x="age", data=autism)


# In[1347]:


autism


# In[1348]:


age_result_counts = autism["age"].value_counts()
age_result_counts


# In[1349]:


age_result_counts = pd.DataFrame(age_result_counts)
age_result_counts


# In[1350]:


sns.relplot(data=result_asd_age_pct, kind="scatter")


# In[1351]:


plt.scatter(x = 'age', y = "result", data = autism_20_30_filter)


# In[1352]:


has_autism


# In[1353]:


has_autism["age"].hist(range = [0, 100], bins = 20)


# In[1354]:


autism["age"].hist(range = [0, 100], bins = 20)


# In[1355]:


autism_new_dummies = pd.get_dummies(autism, columns = ["Class/ASD"], drop_first = True)
autism_new_dummies


# In[1356]:


autism_new_dummies["age"].value_counts()


# In[1358]:


ASD_age_count["age"].value_counts()


# In[1359]:


age_counts = autism_new_dummies["age"].value_counts()
 
#autism_new_dummies["Age_Counts"] = age_counts
# create new column that counts number of people surveyed per age
autism_new_dummies['count'] = autism_new_dummies.groupby('age')['age'].transform('count')

ASD_age_count = autism_new_dummies[autism_new_dummies['Class/ASD_YES'] == 1]
ASD_age_count['ASD_age_count'] = ASD_age_count.groupby('age')['age'].transform('count')
ASD_age_count.head()
#autism_new_dummies['ASD_age_count'] = ASD_age_count

#autism_new_dummies['age_asd_pct'] = autism_new_dummies.groupby('age')['age'].transform('age_asd_pct')
#autism_new_dummies['age_asd_pct'] = autism_new_dummies / autism_new_dummies.DAYSLATE.sum()


# In[1360]:


autism_new_dummies.head()


# In[1361]:


sns.regplot(x = 'age', y = "count", data = autism_new_dummies)


# In[1362]:


sns.regplot(x = 'age', y = "ASD_age_count", data = ASD_age_count)


# In[1363]:


autism_new_dummies["age"].hist(range = [0, 100], bins = 20, alpha = 0.5)
ASD_age_count["age"].hist(range = [0, 100], bins = 20, alpha = 0.5)
plt.xlabel('Age')
plt.ylabel('Counts of people per age')
plt.title('Histogram of age counts: Blue = Whole Population, Orange = Tested W/ Autism')
plt.show()


# In[1364]:


autism_new_dummies.plot.scatter(x = "age", y = "Class/ASD_YES", alpha = 0.5)


# In[1365]:


logit_model = smf.logit("Q('Class/ASD_YES') ~ age", autism_new_dummies).fit()
logit_model.summary()


# In[1366]:


sns.regplot(x = "age", y = "Class/ASD_YES", data = autism_new_dummies, logistic = True)


# In[1367]:


autism_new_dummies.plot.scatter(x = "result", y = "Class/ASD_YES", alpha = 0.5)


# In[1371]:


from sklearn import tree
import graphviz
from graphviz import Source
 
from sklearn.tree import export_graphviz
import sklearn.metrics as met
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn import tree

import graphviz
from graphviz import Source
 
from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz
import sklearn.metrics as met
from sklearn.metrics import confusion_matrix


# In[1372]:


autism_new_dummies2 = pd.get_dummies(autism_new_dummies, columns = ["gender", "jundice", "austim", "relation"], drop_first = True)
autism_new_dummies2.head()


# In[1373]:


X = autism_new_dummies2.drop(columns = ["age","result","ethnicity","contry_of_res","used_app_before","age_desc","count","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","Class/ASD_YES"])
X.head()


# In[1374]:


y = autism_new_dummies2["Class/ASD_YES"]


# In[1375]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[1376]:


reg = tree.DecisionTreeRegressor(max_depth = 2)
reg = reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)


# In[1377]:


reg = tree.DecisionTreeRegressor(max_depth = 3)
reg = reg.fit(X, autism_new_dummies2["Class/ASD_YES"])
tree.plot_tree(reg)


# In[1378]:


((y_pred - y_test)**2).mean()


# In[1379]:


dot_data = tree.export_graphviz(reg, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("ASD.dot")

with open ("ASD.dot", "r") as fin:
    with open("fixed_ASD.dot","w") as fout:
        for line in fin.readlines():
            line = line.replace("X[0]","gender_m")
            line = line.replace("X[1]","jundice_yes")
            line = line.replace("X[2]","austim_yes")
            line = line.replace("X[3]","relation_Others")
            line = line.replace("X[4]","relation_Parent")
            line = line.replace("X[5]","relation_Relative")
            line = line.replace("X[6]","relation_Self")
            fout.write(line)


# ![Screenshot%202019-11-23%2017.48.53.png](attachment:Screenshot%202019-11-23%2017.48.53.png)

# X[2] = has autism or not;
# X[6] = whether the person submitted the test themselves or not;
# X[4] = if the relation of the test taker is a parent

# In[1380]:


reg = tree.DecisionTreeRegressor(max_depth = 3)
reg = reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
dot_data = tree.export_graphviz(reg, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("ASD.dot")


# ![Screenshot%202019-11-23%2017.56.59.png](attachment:Screenshot%202019-11-23%2017.56.59.png)

# In[1381]:


import scipy.cluster.hierarchy as shc

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn import datasets


# In[1382]:


asd_k_cluster = autism_new_dummies
#asd_k_cluster = autism_new_dummies(na_values = ["?", 383])
asd_k_cluster.dropna()
#autism = pd.read_csv("Autism_Data.csv", na_values = ["?", 383])
#autism.dropna()


# In[1383]:


autism_new_dummies.dtypes


# In[1384]:


autism_new_dummies


# In[1385]:


plt.scatter(autism_new_dummies["age"], autism_new_dummies["result"])


# In[1386]:


#columns = ["age", "result"]
asd_k_cluster = autism_new_dummies.drop(columns = ["gender","jundice","austim","relation","Class/ASD_YES","count","ethnicity","contry_of_res","used_app_before","age_desc","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score"])
asd_k_cluster.head()


# In[1387]:


#asd_k_cluster['age'] = asd_k_cluster['age'].astype(int)
asd_k_cluster["result"] = asd_k_cluster["result"].astype(float)


# In[1388]:


asd_k_cluster.dropna(axis = 0)
asd_k_cluster.head()


# In[1389]:


scaler = MinMaxScaler(feature_range=(0, 1))
asd_scaled = scaler.fit_transform(asd_k_cluster)
asd_scaled


# In[1390]:


asd_scaled = pd.DataFrame(asd_scaled, columns = asd_k_cluster.columns, index = asd_k_cluster.index)
asd_scaled.head()


# In[1391]:


asd_scaled = asd_scaled.dropna(axis = 0)
asd_scaled.describe()


# In[1392]:


asd_scaled.dtypes


# In[1393]:


#asd_scaled["age"] = asd_k_cluster["age"].astype(int)
#asd_scaled["result"] = asd_k_cluster["result"].astype(int)


# In[1394]:


#for index, row in asd_scaled["age"].iteritems():
  #  if row != row:
   #     print('index:', index, 'isnull')
#for index, row in asd_scaled["result"].iteritems():
  #  if row != row:
       # print('index:', index, 'isnull')


# In[1395]:


kmeans = KMeans(n_clusters=3, random_state = 0)
kmeans_clusters = kmeans.fit_predict(asd_scaled)
kmeans_clusters


# In[1396]:


asd_scaled["kmeans_clusters"] = kmeans_clusters
asd_scaled.head()


# In[1397]:


asd_scaled[asd_scaled["kmeans_clusters"] == 0].head()


# In[1398]:


asd_scaled[asd_scaled["kmeans_clusters"] == 1].head()


# In[1399]:


asd_scaled[asd_scaled["kmeans_clusters"] == 2].head()


# In[1400]:


sns.relplot(x = "age", y = "result", hue = "kmeans_clusters", data = asd_scaled)


# In[1401]:


errors = asd_scaled["result"] - lm.fittedvalues
errors.head()


# In[1402]:


squared_errors = errors**2
squared_errors.head()


# In[1403]:


mse = squared_errors.mean()
mse


# In[1404]:


iX = asd_scaled.drop(columns = ["result"])
iX


# In[1405]:


iy = asd_scaled["result"]
iy


# In[1406]:


iX_train, iX_test, iy_train, iy_test = train_test_split(iX,iy, test_size = 0.2)


# In[1407]:


ik3nn = KNeighborsRegressor(n_neighbors = 3)
ik3nn.fit(iX_train, iy_train)
iy_pred = ik3nn.predict(iX_test)


# In[1408]:


iy_pred


# In[1409]:


iy_test


# In[1414]:


asd_jaundice_dummies = pd.get_dummies(autism_new_dummies, columns = ["jundice"], drop_first = True)
asd_jaundice_dummies = asd_jaundice_dummies.drop(columns=["age","gender","austim","relation","count","ethnicity","contry_of_res","used_app_before","age_desc","A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score"])
asd_jaundice_dummies.head()


# In[1415]:


asd_jaundice_dummies.plot.scatter(x = "result", y = "jundice_yes", alpha = 0.5)


# In[1416]:


sns.regplot(x = "result", y = "jundice_yes", data = asd_jaundice_dummies, logistic = True)


# In[1417]:


asd_scaled.head()


# In[1418]:


# Bayes Classifier
# Probability results are over/under 7 given that age is over/under 35
seven_up = asd_k_cluster[asd_k_cluster["result"] >= 7]
under_seven = asd_k_cluster[asd_k_cluster["result"] < 7]
older_35 = asd_k_cluster[asd_k_cluster["age"] >= 35]
under_35 = asd_k_cluster[asd_k_cluster["age"] < 35]

