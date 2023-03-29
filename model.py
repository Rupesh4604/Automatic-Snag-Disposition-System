#!/usr/bin/env python
# coding: utf-8

# # AUTOMATIC SNAG DISPOSITION USING MACHINE LEARNING ALGORITHMS

# Problem Statement : 
# To Design and Develop a Machine Learning Model using Machine Learning Algorithms like Multinomial Naive Bayes, Support Vector Machine, for the prediction of Snag Disposition based on SQMS (SNAG & QUERY MANAGEMENT SYSTEM) dataset. The live Snags taken for the validation of model.

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn import model_selection,svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, precision_score, recall_score , confusion_matrix , classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV


# In[2]:


df=pd.read_csv("e:\\snag.csv",encoding='ANSI')
pd.set_option('display.max_columns',None)
df.head()


# # Inspecting the DataFrame

# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# The Above Analysis Porovides the Idea about the missing values in the DataFrame

# In[7]:


df.isnull()


# # Analysing the Given Data

# In[8]:


df_categorical=df.select_dtypes(include=['object'])
print("This Shows the data which is not known or not given ")
df_categorical.apply(lambda x:x=='-',axis=0).sum()


# In[11]:


df['SNAG_STROKE'].value_counts().plot.pie()
plt.figure(figsize=(20,20))
print("Analysis of SNAG_STROKE observed ")
plt.show()


# In[12]:


df['SNAG_STROKE'].value_counts().plot.bar()
plt.figure(figsize=(30,20))
plt.show()


# In[13]:


df['SHOP'].head(100).value_counts().plot.pie()


# In[14]:


df['PROJECT'].value_counts().plot.bar()
plt.rcParams['figure.figsize']=(20,15)
plt.show()


# In[35]:


df['SHOP'].value_counts().plot.bar()
plt.rcParams['figure.figsize']=(20,15)
plt.show()


# In[3]:


df['SHOP']=df['SHOP'].astype(str)
df['SHOP']=df['SHOP'].str.replace('VLD','301')
df['SHOP']=df['SHOP'].str.replace('OSOH','302')
df['SHOP']=df['SHOP'].str.replace('STORE','303')
df['SHOP']=df['SHOP'].str.replace('TOOLING','304')
df['SHOP']=df['SHOP'].astype(int)


# In[10]:


plt.rcParams['figure.figsize']=(15,20)
ax=sns.countplot(x=df['SHOP'])


# In[63]:


ax=sns.violinplot(x=df['SHOP'])
plt.rcParams['figure.figsize']=(5,5)


# In[39]:


df['INSP_STAGE'].value_counts().plot.bar()
plt.rcParams['figure.figsize']=(20,15)
plt.show()


# In[40]:


df['INSP_NAME'].value_counts()[0:20].plot.bar()
plt.rcParams['figure.figsize']=(30,20)
plt.show()


# In[41]:


df['SYSTEM'].value_counts()[1:].plot.bar()
plt.rcParams['figure.figsize']=(30,20)
plt.show()


# In[42]:


df['SYSTEM'].value_counts()


# In[43]:


df['DWG_NO'].value_counts()[0:20].plot.bar()
plt.rcParams['figure.figsize']=(30,20)
plt.show()


# # Data Preparation 

# In[44]:


df['DWG_NO'].value_counts()


# In[45]:


df['DISPOSITION'].value_counts()


# In[4]:


df['DISPOSITION']=df['DISPOSITION'].str.lower()


# In[5]:



df['DISPOSITION']=df['DISPOSITION'].str.replace('refer the attachment.','refer the attachment')
df['DISPOSITION']=df['DISPOSITION'].str.replace('please refer the attachment','refer the attachment')
df['DISPOSITION']=df['DISPOSITION'].str.replace('refered not acceptable to design.','not acceptable to design')
df['DISPOSITION']=df['DISPOSITION'].str.replace('rework carried out by shop is acceptable to design.','acceptable to design')


# In[48]:


df['DISPOSITION'].value_counts()/len(df.index)*100


# In[49]:


df['SNAG_STROKE'].value_counts()/len(df.index)*100


# In[6]:


def log(string):
    display(Markdown("> <span style='color:orange'>"+string+"</span>"))


# In[7]:


df['SNAG_STROKE']=df['SNAG_STROKE'].str.replace('Miscellaneus', 'Miscelleneous')
df['SNAG_STROKE']=df['SNAG_STROKE'].str.replace('Material fault', 'Material Fault')


# In[8]:


df['Status']=df['Unnamed: 0']
df.head(50)


# In[9]:


df=df.drop(columns=['Unnamed: 0'])


# In[10]:


def snag_desc_to_no(df):
    for x in range(len(df)):
        z=str(df['DISPOSITION'][x])
        if 'not acceptable' in z:
            df['Status'][x]=1
        elif 'acceptable' in z:
            df['Status'][x]=0
        else :
            df['Status'][x]=2   


# In[11]:


snag_desc_to_no(df)
df1=df


# In[65]:


df.head(30)


# In[12]:


df1=df1.drop(columns=['Forward Date','Disp Date','CLOSE_DATE','SNAG_DATE'])


# In[13]:


df1.columns


# In[14]:


df1=df1.drop(columns=['SNAG_ID', 'ACNO', 'INSP_NAME', 'SHOP', 'INSP_STAGE',
       'SNAG_STROKE', 'ENGR_FLAG', 'PART_NO', 'TASK_NO', 'SYSTEM',
       'SUB_SYSTEM', 'PROJECT',  'DWG_NO'])


# In[14]:


df1.head(30)


# In[15]:


df1['SNAG_DESC']=df1['SNAG_DESC'].str.lower()


# In[16]:


df1['SNAG_DESC']=df1['SNAG_DESC'].str.replace('ref','reference')


# In[ ]:





# 
# # Making Test and Train DataSets
# # Test -Train - Split

# In[62]:


#Splitiing the dataset into training and testing data sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df1['SNAG_DESC'],df1['Status'],test_size=0.15,random_state=0)


# In[63]:


X_train


# In[64]:


y_train


# In[65]:


X_test


# In[66]:


y_test


# In[67]:



from sklearn.feature_extraction.text import CountVectorizer

vec=CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',ngram_range=(0,6), lowercase=True, stop_words='english')
X_train_transformed=vec.fit_transform(X_train)
X_test_transformed=vec.transform(X_test)


# In[68]:


X_train_transformed


# In[69]:


vec.vocabulary_


# In[67]:


len(vec.vocabulary_)


# # MODEL PREPARATION

# In[70]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection,svm
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics

mnb=MultinomialNB()
mnb.fit(X_train_transformed,y_train)
y_pred_class=mnb.predict(X_test_transformed)

print( "Accuracy of the Test is:",metrics.accuracy_score(y_test,y_pred_class)*100,"%")


# In[71]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, precision_score, recall_score
precision, recall, fscore, support = score(y_test, y_pred_class)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[72]:


print("The Training Accuracy of the Model is:" ,metrics.accuracy_score(y_train,mnb.predict(X_train_transformed))*100, "%")


# In[73]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_class)
cm


# In[74]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))

cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,
xticklabels=['Acceptable','Not Acceptable','Rework'], yticklabels=['Acceptable','Not Acceptable','Rework'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# ## Using SVM 

# In[75]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',ngram_range=(0,5), lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

from sklearn.svm import SVC
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM.predict(X_train_cv), y_train)*100,"%")


# In[32]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, precision_score, recall_score
precision, recall, fscore, support = score(y_test, predictions_SVM)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[63]:


cm = confusion_matrix(y_test,predictions_SVM )
cm


# In[62]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))

cm = confusion_matrix(y_test,predictions_SVM )
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,
xticklabels=['Acceptable','Not Acceptable','Rework'], yticklabels=['Acceptable','Not Acceptable','Rework'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# ### The Model is good with accuracy greater than 70% but still can be enhanced using
# #### (1) Dealing with class imbalance
# #### (2) Hypertuning the parameters

# In[17]:


def to_no(df):
    noc = 0
    acc = 0
    rew = 0
    for x in range(len(df)):
        z=int(df['Status'][x])
        if z == 1:
            noc+=1
        elif z == 0:
            acc+=1
        else:
            rew+=1
    return noc,acc,rew  


# In[18]:


noc,acc,rew=to_no(df)
print("noc:",noc)
print("acc:",acc)
print("rew:",rew)


# #### Upscaling the minority samples

# In[19]:


df_majority=pd.DataFrame()
df_majority1=pd.DataFrame()
df_minority=pd.DataFrame()
df_minority_upsample=pd.DataFrame()
df_upsample=pd.DataFrame()

df_majority=df[(df['Status']==2)]
df_majority1=df[(df['Status']==0)]
df_minority=df[(df['Status']==1)]

for i in range(8):
    df_minority_upsample=pd.concat([df_minority,df_minority_upsample],ignore_index=True)
    
print(df_minority_upsample.shape)
df_minority_upsample.head()


# In[20]:


df_majority=pd.concat([df_majority,df_majority1],ignore_index=True)
df_upsample=pd.concat([df_majority,df_minority_upsample],ignore_index=True)
df_upsample.shape


# In[21]:


noc,acc,rew=to_no(df_upsample)
print("noc:",noc)
print("acc:",acc)
print("rew:",rew)


# In[22]:


X_train,X_test,y_train,y_test = train_test_split(df_upsample['SNAG_DESC'],df_upsample['Status'],test_size=0.15,random_state=0)
vec=CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',ngram_range=(0,6), lowercase=True, stop_words='english')
X_train_transformed=vec.fit_transform(X_train)
X_test_transformed=vec.transform(X_test)


# In[23]:


from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train_transformed,y_train)
predictions=mnb.predict(X_test_transformed)

print( "MultinomialNB Test Accuracy is :",metrics.accuracy_score(y_test,predictions)*100,"%")
print("MultinomialNB Training Accuracy is:" ,metrics.accuracy_score(y_train,mnb.predict(X_train_transformed))*100, "%" , "\n")
precision, recall, fscore, support = score(y_test, predictions)
print(metrics.classification_report(y_test,predictions))


# In[85]:


cm = confusion_matrix(y_test,predictions )
cm


# In[88]:


plt.figure(figsize=(15,10))
cm = confusion_matrix(y_test,predictions )
sns.heatmap(cm, square=True, annot=True, cbar=False,xticklabels=['Acceptable','Not Acceptable','Rework'], yticklabels=['Acceptable','Not Acceptable','Rework'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# In[86]:



from sklearn import model_selection,svm
from sklearn.svm import SVC

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.01)
SVM.fit(X_train_transformed, y_train)
predictions_SVM = SVM.predict(X_test_transformed)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%" )
print("SVM Training Accuracy Score :",accuracy_score(SVM.predict(X_train_transformed), y_train)*100,"%" , "\n")
precision, recall, fscore, support = score(y_test, predictions_SVM)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print(metrics.classification_report(y_test,predictions_SVM))


# In[87]:


cm = confusion_matrix(y_test,predictions_SVM )
cm


# In[89]:


plt.figure(figsize=(15,10))
cm = confusion_matrix(y_test,predictions_SVM )
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,xticklabels=['Acceptable','Not Acceptable','Rework'], yticklabels=['Acceptable','Not Acceptable','Rework'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# Hypertuning the parameters

# In[23]:


X_train_cv = X_train_transformed
X_test_cv = X_test_transformed


# In[24]:


from sklearn.svm import SVC
SVM_t1 = svm.SVC(C=10.0, kernel='linear', degree=3, gamma=0.01)
SVM_t1.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t1.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t1.predict(X_train_cv), y_train)*100,"%")


# In[25]:


from sklearn.svm import SVC
SVM_t2 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.01)
SVM_t2.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t2.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t2.predict(X_train_cv), y_train)*100,"%")


# In[26]:


from sklearn.svm import SVC
SVM_t3 = svm.SVC(C=0.1, kernel='linear', degree=3, gamma=0.01)
SVM_t3.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t3.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t3.predict(X_train_cv), y_train)*100,"%")


# In[27]:


from sklearn.svm import SVC
SVM_t4 = svm.SVC(C=0.1, kernel='rbf', degree=3, gamma=0.01)
SVM_t4.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t4.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t4.predict(X_train_cv), y_train)*100,"%")


# In[28]:


from sklearn.svm import SVC
SVM_t5 = svm.SVC(C=1, kernel='rbf', degree=3, gamma=0.01)
SVM_t5.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t5.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t5.predict(X_train_cv), y_train)*100,"%")


# In[29]:


from sklearn.svm import SVC
SVM_t6 = svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.01)
SVM_t6.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t6.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t6.predict(X_train_cv), y_train)*100,"%")


# In[30]:


from sklearn.svm import SVC
SVM_t7 = svm.SVC(C=100, kernel='rbf', degree=3, gamma=0.01)
SVM_t7.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t7.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t7.predict(X_train_cv), y_train)*100,"%")


# ## Upsampling again using randomClassifiers and Smote and proceding further

# In[33]:


df2=pd.read_csv("e:\\snagsover1.csv",encoding='ANSI')
df2=df2.drop(columns=['Forward Date','Disp Date','CLOSE_DATE','SNAG_DATE'])
pd.set_option('display.max_columns',None)
df2.shape


# In[34]:


df2['Status']=df2["target"]
df2.head()


# In[35]:


def snag_desc_to_no(df):
    for x in range(len(df)):
        z=str(df['DISPOSITION'][x])
        if 'not acceptable' in z:
            df['Status'][x]=1
        elif 'acceptable' in z:
            df['Status'][x]=0
        else :
            df['Status'][x]=2  


# In[36]:


noc,acc,rew=to_no(df2)
print("noc:",noc)
print("acc:",acc)
print("rew:",rew)


# In[37]:


df2=df2.drop(columns=['P','target'])
snag_desc_to_no(df2)


# In[33]:


X_train,X_test,y_train,y_test = train_test_split(df2['SNAG_DESC'],df2['Status'],test_size=0.15,random_state=0)
vec=CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',ngram_range=(0,6), lowercase=True, stop_words='english')
X_train_transformed=vec.fit_transform(X_train)
X_test_transformed=vec.transform(X_test)

mnb=MultinomialNB()
mnb.fit(X_train_transformed,y_train)
Predictions=mnb.predict(X_test_transformed)

print( "Accuracy of the Test is:",metrics.accuracy_score(y_test,Predictions)*100,"%")
print("The Training Accuracy of the Model is:" ,metrics.accuracy_score(y_train,mnb.predict(X_train_transformed))*100, "%" , "\n")
precision, recall, fscore, support = score(y_test,Predictions)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[34]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Predictions)
cm


# In[35]:


print(metrics.classification_report(y_test,Predictions))


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
cm = confusion_matrix(y_test,Predictions )
sns.heatmap(cm, square=True, annot=True, cbar=False,
xticklabels=['Acceptable','Not Acceptable','Rework'], yticklabels=['Acceptable','Not Acceptable','Rework'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# In[39]:


from sklearn.svm import SVC
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.01)
SVM.fit(X_train_transformed, y_train)
predictions_SVM = SVM.predict(X_test_transformed)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM.predict(X_train_transformed), y_train)*100,"%" , "\n")
precision, recall, fscore, support = score(y_test, predictions_SVM)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[38]:


cm = confusion_matrix(y_test,predictions_SVM )
cm


# In[39]:


print(metrics.classification_report(y_test,predictions_SVM))


# In[40]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
cm = confusion_matrix(y_test,predictions_SVM )
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,
xticklabels=['Acceptable','Not Acceptable','Rework'], yticklabels=['Acceptable','Not Acceptable','Rework'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# ### Hyperparameter tuning

# In[9]:


X_train,X_test,y_train,y_test = train_test_split(df2['SNAG_DESC'],df2['Status'],test_size=0.15,random_state=0)
vec=CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',
                    ngram_range=(0,6), lowercase=True, stop_words='english')
X_train_transformed=vec.fit_transform(X_train)
X_test_transformed=vec.transform(X_test)

#Kfold cross-validation 

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)
hyper_params = [{'gamma':[1e-1,1e-2,1e-3],'C':[1,10,100]}]
model = svm.SVC(kernel='rbf')
model_cv_svm = GridSearchCV(estimator=model,param_grid=hyper_params,scoring="accuracy",cv=folds,
                            n_jobs=-1,verbose=1,return_train_score=True)
model_cv_svm.fit(X_train_transformed,y_train)


# In[13]:


def display_stats(cv_results,param_value):
    gamma = cv_results[cv_results['param_gamma']==param_value]
    plt.plot(gamma['param_C'],gamma['mean_test_score'])
    plt.plot(gamma['param_C'],gamma['mean_train_score'])
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.title("Gamma="+str(param_value))
    plt.ylim([0.6,1])
    plt.legend(['test accuracy','train accuracy'],loc='lower right')
    plt.xscale('log')


# In[16]:


svm_cv_results = pd.DataFrame(model_cv_svm.cv_results_)
svm_cv_results['param_C'] = svm_cv_results['param_C'].astype('int')
gamma=[1e-1,1e-2,1e-3]
plt.figure(figsize=(16,5))
plt.subplot(131)
display_stats(svm_cv_results,gamma[0])
plt.subplot(132)
display_stats(svm_cv_results,gamma[1])
plt.subplot(133)
display_stats(svm_cv_results,gamma[2])
plt.show()


# In[48]:


X_train,X_test,y_train,y_test = train_test_split(df2['SNAG_DESC'],df2['Status'],test_size=0.15,random_state=0)
vec=CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',ngram_range=(0,6), lowercase=True, stop_words='english')
X_train_transformed=vec.fit_transform(X_train)
X_test_transformed=vec.transform(X_test)
X_train_cv = X_train_transformed
X_test_cv = X_test_transformed


# In[24]:


from sklearn.svm import SVC
SVM_t1 = svm.SVC(C=10.0, kernel='linear', degree=3, gamma=0.01)
SVM_t1.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t1.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t1.predict(X_train_cv), y_train)*100,"%")


# In[26]:


from sklearn.svm import SVC
SVM_t2 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.01)
SVM_t2.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t2.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t2.predict(X_train_cv), y_train)*100,"%")


# In[49]:


from sklearn.svm import SVC
SVM_t3 = svm.SVC(C=0.1, kernel='linear', degree=3, gamma=0.01)
SVM_t3.fit(X_train_cv, y_train)
# predict the labels on validation dataset
predictions_SVM = SVM_t3.predict(X_test_cv)

print("SVM Accuracy Score :",accuracy_score(predictions_SVM, y_test)*100,"%")
print("SVM Training Accuracy Score :",accuracy_score(SVM_t3.predict(X_train_cv), y_train)*100,"%")


# In[ ]:





# In[40]:


testing_predictions = []
for i in range(len(X_test)):
    if predictions_SVM[i] == 1:
        testing_predictions.append('NOT acc')
    elif predictions_SVM[i] == 0:
        testing_predictions.append('acceptable')
    else:
        testing_predictions.append('rework')
check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'SNAG_DESC':list(X_test)})
check_df.replace(to_replace=1, value='Not Acc', inplace=True)
check_df.replace(to_replace=0, value='acceptable', inplace=True)
check_df.replace(to_replace=2, value='rework', inplace=True)


# In[41]:


predictions_SVM


# In[42]:


check_df.head(30)


# In[43]:


SNAG_DESC="draw"
ch = pd.DataFrame({SNAG_DESC})
chv=vec.transform(ch[0])
print(SVM.predict(chv))


# In[44]:


SNAG_DESC1=['Dent/Tool mark observd in the peice','As per main view of drw gap noticed due to lack of material up to length 165.5mm, dpth 51mm,width 120m ref attached sketch','LEAK','acceptable','not acceptable',
            'lh air intake ext i/b p.s.s number 245 qty 1 joe bolt found pulled up approx 1.2 mm at frame number 18','refer drawing number 11.2003.2.000.000 zone 4 during installation of verticle bolt part number 11.2003.7.208.900 on lh/rh wing it is observed that length is short by 2.5 mm for split pinning even using the washer of minimu dimension of 1 mm  and maintaining the 1.5 mm gap between head of bolt and wing structure. design/ppo is requested to give disposition']
ch = pd.DataFrame(SNAG_DESC1)
chv=vec.transform(ch[0])
print(SVM.predict(chv))


# In[45]:


def num_to_disposition(tar):
    for i in tar:
        if i == 0 : print("Design is Acceptable")
        elif i==1 : print("Design is Not Acceptable")
        else : print("Rework is Required")


# In[55]:


SNAG_DESC1=['Dent/Tool mark observd in the peice','As per main view of drw gap noticed due to lack of material up to length 165.5mm, dpth 51mm,width 120m ref attached sketch','LEAK','acceptable','not acceptable',
            'lh air intake ext i/b p.s.s number 245 qty 1 joe bolt found pulled up approx 1.2 mm at frame number 18','refer drawing number 11.2003.2.000.000 zone 4 during installation of verticle bolt part number 11.2003.7.208.900 on lh/rh wing it is observed that length is short by 2.5 mm for split pinning even using the washer of minimu dimension of 1 mm  and maintaining the 1.5 mm gap between head of bolt and wing structure. design/ppo is requested to give disposition',
           'reference. attached sketch & dwg detail view g...' ]
ch = pd.DataFrame(SNAG_DESC1)
print(ch.head(30),"\n")
chv=vec.transform(ch[0])
print(chv,"\n")
tar=SVM.predict(chv)
print("Status array :",tar)


# In[56]:


num_to_disposition(tar)


# # Using Various Machine Learning Models

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df2.SNAG_DESC).toarray()
labels = df2.Status
features.shape
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df2['SNAG_DESC'], df2['Status'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[13]:


import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[14]:


cv_df.groupby('model_name').accuracy.mean()


# In[17]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df2['SNAG_DESC'],df2['Status'],test_size=0.15,random_state=0)

vec=CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' or '\D',ngram_range=(0,6), lowercase=True, stop_words='english')
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

model=GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
model.fit(X_train,y_train)
predicts_train = model.predict(X_train)
Training_Accuracy = accuracy_score(y_train,predicts_train)
print("Training_Accuracy :",Training_Accuracy*100 )
predicts_test = model.predict(X_test)
Testing_Accuracy = accuracy_score(y_test,predicts_test)
print("Testing_Accuracy :",Testing_Accuracy*100)


# ### The Mean Accuracy Scores using Various Models are as follows :
model_name                Accuracy Score

LinearSVC                 94.1861%

LogisticRegression        87.1631%

MultinomialNB             78.2713%

RandomForestClassifier    45.1999%
# ### The Highest Accuracy was reached with SVM using the hyperparameters C=10.0, kernel='linear', degree=3, 
# 
# ### gamma=0.01

# #### SVM Accuracy Score : 95.5495251017639 %
# #### SVM Training Accuracy Score : 98.76419025722086 % 
