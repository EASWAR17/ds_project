# Heart Disease Prediction model

## CODE:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("heart.csv")
df.head()
df.info()
df.describe()
df1=df.copy()

def chng(sex):
    if sex == 0:
        return 'female'
    else:
        return 'male'
df1['sex'] = df1['sex'].apply(chng)

def chng2(prob):
    if prob == 0:
        return 'Heart Disease'
    else:
        return 'No Heart Disease'
df1['target'] = df1['target'].apply(chng2)

sns.countplot(x='sex',hue='target',data= df1)
plt.title('Gender v/s target\n')

sns.countplot(x='cp',hue='target',data=df1)
plt.title("chest pain vs heart disease")

sns.barplot(x='target',y='trestbps',data=df1)

df1.loc[df1['cp'] == 0, 'cp'] = 'asymptomatic'
df1.loc[df1['cp'] == 1, 'cp'] = 'atypical angina'
df1.loc[df1['cp'] == 2, 'cp'] = 'non anginal pain'
df1.loc[df1['cp'] == 3, 'cp'] = 'typical angina'
sns.countplot(x='cp',hue='target',data=df1)
plt.title("chest pain vs heart disease")

sns.barplot(y='chol',x='target',data=df1)

sns.countplot(x='fbs',hue='target',data=df1)
plt.title("fasting blood sugar vs heart disease")

# d_f_v= definite left ventricular hypertrophy
df1.loc[df1['restecg'] == 0, 'restecg'] = 'd_f_v'
df1.loc[df1['restecg'] == 1, 'restecg'] = 'normal'
df1.loc[df1['restecg'] == 2, 'restecg'] = 'stw abnormality'

sns.countplot(x='restecg',hue='target',data=df1)
plt.title("fasting blood sugar vs heart disease")

sns.barplot(x='target',y='thalach',data=df1)

sns.barplot(x='target',y='exang',data=df1)

sns.barplot(x='target',y='ca',data=df1)

sns.countplot(x='thal',hue='target',data=df1)

sns.boxplot(x='cp',data=df)

sns.boxplot(x='trestbps',data=df)

sns.boxplot(x='chol',data=df)

sns.boxplot(x='thal',data=df)

x= df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y= df['target']

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,)
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(x_train,y_train)
y_pred = le.predict(x_test)
from sklearn import metrics
MSE = metrics.mean_squared_error(y_test,y_pred)
print("MSE is {}".format(MSE))
r2 = metrics.r2_score(y_test,y_pred)
print("R Squared Error is {} ".format(r2))

print("enter values of certain parameters: age\tsex\tcp\ttrestbps\tchol\tfbs\trestecg\tthalach\texang\toldpeak\tslope\tca\tthal")
le.predict([["enter ordered input"]])

# OUTCOME INSIGHTS:

#using all features -> for target 0
'''
array([0.24947887])
array([0.19439689])
array([0.77405948])
array([0.35718321])
array([-0.04955935])
array([0.32346927])
array([0.24319396])
array([0.49253258])
array([-0.03486405])
'''

#using all features -> for target 1
'''
array([0.71895343])
array([0.62314506])
array([0.93223759])
array([0.80740755])
array([0.80710966])
array([0.77720923])
array([0.71245575])
array([0.66842059])
array([0.7697669])
'''

print(outcome[0])

if(outcome[0]>0.5000):
  print("the person have higher chances of getting heart disease")
else:
  print("the person have very less or no chances of getting heart disease")
```
