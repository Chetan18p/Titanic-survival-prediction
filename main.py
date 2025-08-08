import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

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

#Visualizing survived and not survived passenger

sns.countplot(x='Survived', data=df)
plt.title('Survived vs Not Survived')
plt.show()
