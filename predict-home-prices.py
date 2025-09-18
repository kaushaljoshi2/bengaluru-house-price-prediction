
import numpy as np
import pandas as pd
import os

# Check available data files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load dataset
data=pd.read_csv('Bengaluru_House_Data.csv')
data

# View data summary and column names
data.info()
data.columns

# Data cleaning - Drop irrelevant or mostly empty columns
data=data.drop(['area_type','availability','balcony','society'],axis=1)
data

data.isna().sum()

# Drop rows with missing values
data=data.dropna()
data.shape

# Explore unique values in 'size' column (e.g., "2 BHK", "4 Bedroom", etc.)
data['size'].unique()

#Feature Engineering

# Extract number of bedrooms (BHK) from the 'size' column
data['BHK']=data['size'].apply(lambda x: int(x.split(' ')[0]))
data.head()
data['BHK'].unique()

# Check for unrealistic BHK values
data[data.BHK>20]

# Explore total_sqft column
data.total_sqft.unique()

# Define a helper function to check if a value is a float
def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True

# View non-numeric values in total_sqft
data[~data['total_sqft'].apply(isfloat)].head(10)

# Convert range values (e.g., "2100-2850") to their average
def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None

# Apply the sqft conversion
data=data.copy()
data['total_sqft']=data['total_sqft'].apply(convert_sqft_tonum)

data.loc[30]

# Create new feature: price per square foot
data1=data.copy()
data1['price_per_sqft']=data1['price']*1000000/data1['total_sqft']
data1.head()

len(data1.location.unique())

# Clean up location data
data1.location=data1.location.apply(lambda x: x.strip())
location_stats=data1.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats

len(location_stats[location_stats<=10])
len(data1.location.unique())

# Replace rare locations (less than or equal to 10 occurrences) with 'other'
data1.location=data1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
len(data1.location.unique())

data1[data1.total_sqft/data1.BHK<300].head()

# Remove outliers where total sqft per BHK is less than 300
data2=data1[~(data1.total_sqft/data1.BHK<300)]
data2.head(10)
data2.shape

data2["price_per_sqft"].describe().apply(lambda x:format(x,'f'))

# Function to remove outliers based on price per square foot within each location
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

# Apply the outlier removal
data3=remove_pps_outliers(data2)
data3.shape

# Function to visualize pricing for 2BHK vs 3BHK homes in a location
import matplotlib.pyplot as plt
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.BHK==2)]
    bhk3=df[(df.location==location)&(df.BHK==3)]
    plt.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='+',label='3 BHK',s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()

# Example visualization for Rajaji Nagar
plot_scatter_chart(data3,"Rajaji Nagar")

# Function to remove BHK outliers (e.g., 3 BHK priced less than 2 BHK in same location)
def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

# Apply the BHK outlier removal
data4=remove_bhk_outliers(data3)
data4.shape

# Plot histogram of price per square foot after cleaning
plt.rcParams['figure.figsize']=(20,15)
plt.hist(data4.price_per_sqft,rwidth=0.6)
plt.xlabel("Price Per Square Foot")
plt.ylabel("Count")

# Explore bathroom feature
data4.bath.unique()
data4[data4.bath>10]

plt.rcParams['figure.figsize']=(20,15)

# Plot histogram of bathroom counts
plt.hist(data4.bath,rwidth=0.6)
plt.xlabel("Number Of Bathroom")
plt.ylabel("Count")

data4[data4.bath>data4.BHK+2]

# Remove homes where number of bathrooms > BHK + 2 (considered outliers)
data5=data4[data4.bath<data4.BHK+2]
data5.shape

# One-hot encode location column
dummies=pd.get_dummies(data6.location)
dummies.head(10)
data7=pd.concat([data6,dummies.drop('other',axis='columns')],axis='columns')
data7.head()

# Drop original location column
data8=data7.drop('location',axis='columns')
data8.head()
data8.shape

# Prepare features and target
X=data8.drop('price',axis='columns')
X.head()
y=data8.price

# Train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Train a linear regression model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

# Evaluate model performance
model.score(X_test,y_test)

# Cross-validation to verify model stability
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

# Prediction function
def price_predict(location,sqft,bath,BHK):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=BHK
    if loc_index >=0:
        x[loc_index]=1
    return model.predict([x])[0]

# Sample predictions
price_predict('1st Phase JP Nagar',1000,2,2)
price_predict('1st Phase JP Nagar',1000,2,3)
price_predict('5th Phase JP Nagar',1000,2,2)
price_predict('Indira Nagar',1000,2,2)
