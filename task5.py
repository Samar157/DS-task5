#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import warnings 
warnings.filterwarnings('ignore')

#load and read the file
df=pd.read_csv("RTA Dataset.csv")
df.head()

#shape/ size of the data
df.shape

#checking the numerical statistics of the data
df.describe()

df.describe(include="all")

#checking data types of each columns
df.info()

#finding duplicate values
df.duplicated().sum()

#Distribution of Accident severity
df['Accident_severity'].value_counts()

#plotting the final class
sns.countplot(x = df['Accident_severity'])
plt.title('Distribution of Accident severity')
plt.show()

#checking missing values
df.isna().sum()

#dropping columns which has more than 2500 missing values and Time column
df.drop(['Service_year_of_vehicle','Defect_of_vehicle','Work_of_casuality', 'Fitness_of_casuality','Time'],
        axis = 1, inplace = True)
df.head()

#storing categorical column names to a new variable
categorical=[i for i in df.columns if df[i].dtype=='O']
print('The categorical variables are',categorical)

#for categorical values we can replace the null values with the Mode of it
for i in categorical:
    df[i].fillna(df[i].mode()[0],inplace=True)

#checking the current null values
df.isna().sum()

#plotting relationship between Number_of_casualties and Number_of_vehicles_involved
sns.scatterplot(x=df['Number_of_casualties'], y=df['Number_of_vehicles_involved'], hue=df['Accident_severity'])
plt.show()

#joint Plot
sns.jointplot(x='Number_of_casualties',y='Number_of_vehicles_involved',data=df)
plt.show()

#checking the correlation between numerical columns
df.select_dtypes(include='number').corr()

#plotting the correlation using heatmap
sns.heatmap(df.select_dtypes(include='number').corr())
plt.show()

#storing numerical column names to a variable
numerical=[i for i in df.columns if df[i].dtype!='O']
print('The numerica variables are',numerical)

#distribution for numerical columns
plt.figure(figsize=(10,10))
plotnumber = 1
for i in numerical:
    if plotnumber <= df.shape[1]:
        ax1 = plt.subplot(2,2,plotnumber)
        plt.hist(df[i],color='red')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('frequency of '+i, fontsize=10)
    plotnumber +=1
plt.show()

#count plot for categorical values
num_plots = len([col for col in categorical if col != 'Pedestrian_movement'])

plt.figure(figsize=(10, 4 * num_plots))
plotnumber = 1

for col in categorical:
    if col != 'Pedestrian_movement':
        ax = plt.subplot(num_plots, 1, plotnumber)
        sns.countplot(data=df, y=col, palette='muted')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(col.title(), fontsize=14)
        plt.xlabel('')
        plt.ylabel('')
        plotnumber += 1

plt.tight_layout()
plt.show()

df.dtypes

#importing label encoing module
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#creating a new data frame from performing the chi2 analysis
df1=pd.DataFrame()

#adding all the categorical columns except the output to new data frame
for i in categorical:
    if i!= 'Accident_severity':
        df1[i]=le.fit_transform(df[i])

#confirming the data type
df1.info()

plt.figure(figsize=(22,17))
sns.set(font_scale=1)
sns.heatmap(df1.corr(), annot=True)
plt.show()

#label encoded data set
df1.head()

#import chi2 test
from sklearn.feature_selection import chi2
f_p_values=chi2(df1,df['Accident_severity'])

#f_p_values will return Fscore and pvalues
f_p_values

#for better understanding and ease of access adding them to a new dataframe
f_p_values1=pd.DataFrame({'features':df1.columns, 'Fscore': f_p_values[0], 'Pvalues':f_p_values[1]})
f_p_values1

#since we want lower Pvalues we are sorting the features
f_p_values1.sort_values(by='Pvalues',ascending=True)

#after evaluating we are removing lesser important columns and storing to a new data frame
df2=df.drop(['Owner_of_vehicle', 'Type_of_vehicle', 'Road_surface_conditions', 'Pedestrian_movement',
         'Casualty_severity','Educational_level','Day_of_week','Sex_of_driver','Road_allignment',
         'Sex_of_casualty'],axis=1)
df2.head()

df2.shape
df2.info()

#to check distinct values in each categorical columns we are storing them to a new variable
categorical_new=[i for i in df2.columns if df2[i].dtype=='O']
print(categorical_new)

for i in categorical_new:
    print(df2[i].value_counts())

#get_dummies
dummy=pd.get_dummies(df2[['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience',
                          'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                          'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                          'Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident']],drop_first=True)
dummy.head()

#concatinate dummy and old data frame
df3=pd.concat([df2,dummy],axis=1)
df3.head()

#dropping dummied columns
df3.drop(['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience', 'Area_accident_occured', 'Lanes_or_Medians',
          'Types_of_Junction', 'Road_surface_type', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
          'Vehicle_movement','Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident'],axis=1,inplace=True)
df3.head()

x=df3.drop(['Accident_severity'],axis=1)
x.shape
x.head()

y=df3.iloc[:,2]
y.head()

#checking the count of each item in the output column
y.value_counts()

#plotting count plot using seaborn
sns.countplot(x = y, palette='muted')
plt.show()

#importing SMOTE 
from imblearn.over_sampling import SMOTE
oversample=SMOTE()
xo,yo=oversample.fit_resample(x,y)

#checking the oversampling output
y1=pd.DataFrame(yo)
y1.value_counts()

sns.countplot(x = yo, palette='muted')
plt.show()

#converting data to training data and testing data
from sklearn.model_selection import train_test_split
#splitting 70% of the data to training data and 30% of data to testing data
x_train,x_test,y_train,y_test=train_test_split(xo,yo,test_size=0.30,random_state=42)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#KNN model alg
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(x_train,y_train)

y_pred=model_KNN.predict(x_test)
y_pred

report_KNN=classification_report(y_test,y_pred)
print(report_KNN)

accuracy_KNN=accuracy_score(y_test,y_pred)
print(accuracy_KNN)

matrix_KNN=confusion_matrix(y_test,y_pred)
print(matrix_KNN,'\n')
print(ConfusionMatrixDisplay.from_predictions(y_test,y_pred))
plt.show()