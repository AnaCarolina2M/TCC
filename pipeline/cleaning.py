from interfaces.cleaning import IDataClean
import pandas as pd
import numpy as np
from pathlib import Path

class CleaningPipeline(IDataClean):
    def __init__(self, df):
        self.df = df
        self.output_path = None

    def drops_missing_values(self):
        self.df = pd.read_csv(self.df)
        self.df = self.df.replace(["", "?", "NA", "null"], np.nan).dropna()
        return self.df

    def object_treatment(self):
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        return self.df

    def imputes_missing_values(self):
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        return self.df
        #ou outro tipo de imputação, dependendo do contexto dos dados

    def encodes_categorical_variables(self):
        categorical_variables =self.df.select_dtypes(include="object")
        self.df = pd.get_dummies(categorical_variables, drop_first=True)
        return self.df
    
    def saves_cleaned_data(self):
        self.output_path = Path("data/cleaned_data")
        self.df.to_csv(self.output_path, index=False)
        print(f"Cleaned data saved to {self.output_path}")

    def cleans(self):
        self.drops_missing_values()
        self.object_treatment()
        self.imputes_missing_values()
        self.encodes_categorical_variables()
        self.saves_cleaned_data()
        return self.df