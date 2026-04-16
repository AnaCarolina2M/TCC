from strategies.cleaning import IDataClean
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

class CleaningPipeline(IDataClean):
    def __init__(self, df):
        self.df = df
        self.file_path = None
        self.output_path = None
    
    def reads_data(self):
        try:
            self.df = pd.read_csv(Path(self.file_path))     
        except Exception as e:
            print(f"Error reading data from {self.file_path}: {e}")
            return None

    def converts_col_to_boolean(self, columns:list):
        self.df[columns] = self.df[columns].astype(bool)
        return self.df

    def converts_data_types(self):
        self.df = self.df.drop(columns=['id', 'age_desc'])
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        # self.df = self.df.dropna(subset=['age_desc'])
        max_age = self.df['age'].max()
        self.df = self.df[self.df['age'] != max_age]
        self.df = self.df.replace(["", "?", "NA", "null"], np.nan)
        return self.df

    def imputes_missing_values(self):
        categorical_cols_for_imputation = ['ethnicity', 'relation']
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_encoded_for_imputation = self.df[categorical_cols_for_imputation].copy()
        df_encoded_for_imputation = encoder.fit_transform(df_encoded_for_imputation)
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        X_imputed_encoded = imputer.fit_transform(df_encoded_for_imputation)
        X_imputed_encoded_rounded = np.round(X_imputed_encoded)
        df_imputed_categorical = encoder.inverse_transform(X_imputed_encoded_rounded)
        self.df.loc[:, categorical_cols_for_imputation] = df_imputed_categorical
        self.df = self.df.dropna()
        return self.df

    def encodes_categorical_variables(self):
        columns_to_one_hot = ['gender', 'ethnicity', 'contry_of_res', 'relation']
        df_encoded = pd.get_dummies(self.df[columns_to_one_hot], drop_first=True)
        self.df = pd.concat([self.df.drop(columns=columns_to_one_hot), df_encoded], axis=1)
        return self.df
    
    def saves_cleaned_data(self):
        self.output_path = Path("./data/cleaned_data/cleaned_data.csv")
        self.df.to_csv(self.output_path, index=False)
        print(f"Cleaned data saved to {self.output_path}")
    

    def cleans(self):
        self.reads_data()
        self.converts_col_to_boolean(columns=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
       'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','jundice', 'used_app_before'])
        self.converts_data_types()
        self.imputes_missing_values()
        self.encodes_categorical_variables()
        self.saves_cleaned_data()
        return self.df