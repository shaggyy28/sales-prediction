import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class Model:

    def __init__(self):
        
        warnings.filterwarnings('ignore')
        df = pd.read_csv('./Train.csv')
        df.head()
        df2 = pd.read_csv('./Test.csv')
        df2.head()
        # statistical info
        df.describe()

        # datatype of attributes
        df.info()
        # check unique values in dataset
        df.apply(lambda x: len(x.unique()))
        #Preprocessing the dataset

        #check for NULL values in the dataset.

        # check for null values
        df.isnull().sum()
        # check for categorical attributes
        cat_col = []
        for x in df.dtypes.index:
            if df.dtypes[x] == 'object':
                cat_col.append(x)
        cat_col
        #remove unnecessary columns.

        cat_col.remove('Item_Identifier')
        cat_col.remove('Outlet_Identifier')
        cat_col


        # print the categorical columns
        for col in cat_col:
            print(col)
            print(df[col].value_counts())
            print()
        # fill the missing values
        item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
        item_weight_mean
        #check for the missing values of Item_Weight.

        miss_bool = df['Item_Weight'].isnull()
        miss_bool
        for i, item in enumerate(df['Item_Identifier']):
            if miss_bool[i]:
                if item in item_weight_mean:
                    df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
                else:
                    df['Item_Weight'][i] = np.mean(df['Item_Weight'])
        df['Item_Weight'].isnull().sum()
        #check for the missing values of Outler_Type.

        outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
        outlet_size_mode

        #fill in the missing values for Outlet_Size.

        miss_bool = df['Outlet_Size'].isnull()
        df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
        # check for Item_Visibility.

        sum(df['Item_Visibility']==0)
        #We have some missing values for this attribute. 

        # fill in the missing values. 

        # replace zeros with mean
        df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)
        sum(df['Item_Visibility']==0)

        #combine the repeated Values of the categorical column. 

        # combine item fat content
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
        df['Item_Fat_Content'].value_counts()
        #Creation of New Attributes

        #We can create new attributes 'New_Item_Type' using existing attributes 'item_Identifier'. 

        df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
        df['New_Item_Type']
        #After creating a new attribute, let's fill in some meaningful value in it.

        df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
        df['New_Item_Type'].value_counts()
        #We have three categories of (Food, Non-Consumables and Drinks).

        #We will use this 'Non_Consumable' category to represent the 'Fat_Content' which are 'Non-Edible'.
        df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
        df['Item_Fat_Content'].value_counts()
        #create a new attribute to show small values for the establishment year.

        # create small values for establishment year
        df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
        df['Outlet_Years']
        #print the dataframe.

        df.head()
        #Exploratory Data Analysis

        # log transformation
        df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])

        #Label encoding is to convert the categorical column into the numerical column.

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
        cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
        for col in cat_col:
            df[col] = le.fit_transform(df[col])
        #One Hot Encoding

        #We can also use one hot encoding for the categorical columns.

        df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])
        df.head()
        #Splitting the data for Training and Testing

        # drop some columns before training our model.

        self.X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
        self.y = df['Item_Outlet_Sales']
    
        self.X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
        self.y = df['Item_Outlet_Sales']

    
    def fit(self):
        # Define the hyperparameters and their possible values for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            # 'max_features': ['auto', 'sqrt']
        }
        # Create the model object
        model = RandomForestRegressor()

        # Create the GridSearchCV object with the model and parameter grid
        self.grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',n_jobs = -1)

        # # Fit the GridSearchCV object to the training data
        print("Trainging model.....")
        self.grid_search.fit(self.X, self.y)
        # # Get the best parameters and the best estimator
        self.best_params = self.grid_search.best_params_
        self.best_estimator = self.grid_search.best_estimator_
        self.best_score = self.grid_search.best_score_
    
    def predict(self, pre_dict):
        # Get the best parameters and the best estimator
        
        df = pd.DataFrame(pre_dict)
        df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
        df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
        le = LabelEncoder()
        df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
        cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
        for col in cat_col:
            df[col] = le.fit_transform(df[col])
        y= self.best_estimator.predict(df)
        return y[0]
