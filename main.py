from matplotlib import figure
import pandas as pd 
from scipy.__config__ import show
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


print("Supervised Machine learninig\n")
print("Logistic Regression on Titanic data set\n\n")

df = pd.read_csv('MarvellousTitanicDataset.csv')

copy_df = df
print()
print()

# printing the first 5 rows 
print(df.head())
print()
print()

# printing total number of null values
print("Total number of null values in each column: ")
print(df.isnull().sum())
print()
print()

# Our dataset is clean and ready for analysis.


# Visualization


print("Visualization : Survived and non survived passengers")
plt.figure()
target="Survived"
sns.countplot(data=copy_df,x=target).set_title("Survived and non survived passengers") 
show()


print("Visualisation : Survived and non survived passengers based on gender")
plt.figure()
target="Survived"
sns.countplot(data=copy_df,x=target,hue="Sex").set_title("Survived and Non survived based on gender")
show()

print("Visualization : survived and non survived passenger based on the passenger class")
plt.figure()    
target="Survived"
sns.countplot(data=copy_df,x=target,hue="Pclass").set_title("Survived and non survived passengers based on passenger class")
plt.show()

print("Visualization :Survived or non survived based on Age")
plt.figure()
copy_df["Age"].plot.hist().set_title("Survived and non survived based on age ")
plt.show()

print("Visualization : Survived and non survived based on fare")
plt.figure()
copy_df["Fare"].plot.hist().set_title("Survived and non survived based on Fare")
plt.show()



copy_df.drop("Passengerid",axis=1,inplace=True)

print("First 5 entries from loaded dataset after removing zero column")
print(copy_df.head(5))

print("Values of sex column")
print(pd.get_dummies(copy_df["Sex"]))

print("Values of sex column after removing one field")
sex=pd.get_dummies(copy_df["Sex"],drop_first=True)
print(sex.head(5))



print("Values of Pclass column after removing one field ")
Pclass=pd.get_dummies(copy_df["Pclass"],drop_first=True)
print(Pclass.head(5))

print("Values of data set after concatenating new columns")
copy_df=pd.concat([copy_df,sex,Pclass],axis=1)
print(copy_df.head(5))

print("Values of data set after removing irrelevent columns")
copy_df.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
print(copy_df.head(5))

x=copy_df.drop("Survived",axis=1)
y=copy_df["Survived"]
x.columns=x.columns.astype(str)






xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5) 
logmodel=LogisticRegression()
logmodel.fit(xtrain,ytrain)
prediction=logmodel.predict(xtest)

print("Classification report of logistic regression is :")
print(classification_report(ytest,prediction))
print("COnfusion Matrix Of Logistic Regression is :")
print(confusion_matrix(ytest,prediction))
print("Accuracy of logistic regression is :")
print(accuracy_score(ytest,prediction))

