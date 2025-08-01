import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that preserves feature names"""
    
    def __init__(self, func):
        self.func = func
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        return self.func(X)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return input_features

class LeadScoringPreprocessor:
    """
    A class for preprocessing lead scoring data.
    Handles data cleaning, feature engineering, and transformation.
    """
    
    def __init__(self):
        """
        Initialize the preprocessor.
        """
        self.preprocessor = None
        self.is_fitted = False
        
        # Define columns to drop based on notebook analysis
        self.cols_to_drop = [
                    'Prospect ID',
                    'Lead Number',
                    'How did you hear about X Education',
                    'Lead Profile',
                    'Lead Quality',
                    'Asymmetrique Profile Score',
                    'Asymmetrique Activity Score',
                    'Asymmetrique Activity Index',
                    'Asymmetrique Profile Index',
                    'Tags',
                    'I agree to pay the amount through cheque',
                    'Get updates on DM Content',
                    'Update me on Supply Chain Content',
                    'Receive More Updates About Our Courses',
                    'Magazine',
                    'City',
                    'Country',
                    'What matters most to you in choosing a course',
                    'Last Notable Activity',
                    'Do Not Call',
                    'Search',
                    'Newspaper Article',
                    'X Education Forums',
                    'Newspaper',
                    'Digital Advertisement',
                    'Through Recommendations',
                    'A free copy of Mastering The Interview'
                ]
    
    def get_categorical_columns(self, df):
        """Get categorical columns after preprocessing"""
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.values
        return [col for col in categorical_cols if col not in self.cols_to_drop]
    
    def get_numerical_columns(self, df):
        """Get numerical columns after preprocessing"""
        numerical_cols = df.select_dtypes(include=["number"]).columns.values
        return [col for col in numerical_cols if col != 'Converted']
    
    def replace_select_with_nan(self, df):
        """Replace 'Select' values with NaN"""
        df = df.replace('Select', np.nan)
        return df
    
    def feature_engineering(self, df):
    # Lead Source
        df['Lead Source'] = df['Lead Source'].str.replace('|'.join(['google','Pay per Click Ads']),'Google')
        df['Lead Source'] = df['Lead Source'].apply(lambda x: "Referral Sites" if 'blog' in str(x) else x)
        df['Lead Source'] = df['Lead Source'].str.replace('Live Chat','Olark Chat')
        df['Lead Source'] = df['Lead Source'].str.replace('bing','Organic Search')
        df['Lead Source'] = df[df['Lead Source'] != 'Other']['Lead Source'].apply(lambda x: "Other" if str(x) not in df['Lead Source'].value_counts()[:8].index else x)

        # Last Activity 
        activity = ['Last Activity']
        df[activity] = df[activity].replace(['Email Received','SMS Sent'],'SMS/Email Sent')
        df[activity] = df[activity].replace(['Email Marked Spam','Email Bounced','Unsubscribed'],'Not Interested in Email')
        df[activity] = df[activity].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                            'Visited Booth in Tradeshow', 'Approached upfront',
                                                            'Resubscribed to emails', 'Form Submitted on Website'], 'Others')

        # Specialization
        df['Specialization'] = df['Specialization'].str.replace('|'.join(['E-COMMERCE','E-Business']),'E-commerce')
        df['Specialization'] = df['Specialization'].str.replace('Banking, Investment And Insurance','Finance Management')
        df['Specialization'] = df['Specialization'].str.replace('Media and Advertising','Marketing Management')

        return df
    
    def binary_encoding(self, df):
        """Convert binary categorical variables to numeric"""
        df[['Do Not Email']] = df[['Do Not Email']].map(lambda x: 0 if x == 'No' else 1)
        return df
    
    def cat_impute(self, df):
        """Handle missing values in categorical columns"""
        # replace 'Lead Source' missing values with 'Google'
        df['Lead Source'] = df['Lead Source'].replace(np.nan, 'Google')
        
        # replace 'Last Activity' missing values with 'Email Opened'
        df['Last Activity'] = df['Last Activity'].replace(np.nan, 'Email Opened')
        
        # replace missing values with 'Unknown'
        unknown_cols = ['Specialization', 'What is your current occupation']
        df[unknown_cols] = df[unknown_cols].fillna('Unknown')
        
        return df
    
    def cat_processing(self, df):
        """Process categorical variables"""
        df = self.binary_encoding(df)
        df = self.cat_impute(df)
        return df
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        # Handle missing values for categorical columns
        categorical_transformer = make_pipeline(
            CustomFunctionTransformer(self.replace_select_with_nan),
            CustomFunctionTransformer(self.cat_processing),
            OneHotEncoder(handle_unknown='ignore')
        )
        
        # Handle missing values for numerical columns
        numerical_transformer = make_pipeline(
            KNNImputer(n_neighbors=5),
            StandardScaler()
        )
        
        return categorical_transformer, numerical_transformer
    
    def fit_transform(self, X):
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            numpy.ndarray: Transformed features X
        """
        # Create a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Drop specified columns
        X_processed = X_processed.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Feature Engineering
        X_processed = self.feature_engineering(X_processed)
        
        # Get column types
        categorical_cols = self.get_categorical_columns(X_processed)
        numerical_cols = self.get_numerical_columns(X_processed)

        print(categorical_cols)
        print(numerical_cols)
        
        # Create transformers
        categorical_transformer, numerical_transformer = self.create_preprocessor()
        
        # Create column transformer
        self.preprocessor = make_column_transformer(
            (categorical_transformer, categorical_cols),
            (numerical_transformer, numerical_cols),
            remainder='drop'
        )
        
        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X_processed)
        self.is_fitted = True
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: New data to transform
            
        Returns:
            numpy.ndarray: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data. Use fit_transform() first.")
        
        # Create a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Drop specified columns
        X_processed = X_processed.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Feature Engineering
        X_processed = self.feature_engineering(X_processed)
        
        
        # Transform the data
        transformed_features = self.preprocessor.transform(X_processed)
        
        return transformed_features
    
    def get_feature_names(self):
        """Get feature names after transformation"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names.")
        return self.preprocessor.get_feature_names_out()

