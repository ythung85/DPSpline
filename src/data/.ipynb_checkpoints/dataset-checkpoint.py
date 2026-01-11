
import torch
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import os

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

def df2tensor_bike(df):
    X_df = df.iloc[:, :-1]   
    y_df = df.iloc[:, -1]    
    
    X_np = X_df.values
    y_np = y_df.values
    
    X_tensor = torch.from_numpy(X_np).float()
    y_tensor = torch.from_numpy(y_np).float() 
    
    return X_tensor, y_tensor.view(-1, 1)

def load_year_prediction_msd(file_path):
    """
    
    Args:
        file_path: 'YearPredictionMSD.txt.zip' 
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    print("Loading YearPredictionMSD... this might take a minute.")
    

    df = pd.read_csv(file_path, header=None)
    

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    

    train_size = 463715
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"Data Loaded.")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, Features: {X_train.shape[1]}")
    
    
    return X_train, y_train, X_test, y_test

class Dataset:
	def __init__(self, case):
		self.case = case
		self.data = self._load_data()
	
	def _load_data(self):
		data_dict = {}
		
		if self.case == 'ca':
			housing = fetch_california_housing()
			X, y = torch.tensor(housing.data), torch.tensor(housing.target, dtype=torch.float32)
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
			
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_val = scaler.transform(X_val)
			X_test = scaler.transform(X_test)
			
			X_train = torch.tensor(X_train, dtype=torch.float32)
			X_val = torch.tensor(X_val, dtype=torch.float32)
			X_test = torch.tensor(X_test, dtype=torch.float32)
			
			y_train = y_train.detach().clone().requires_grad_(True).view(-1, 1)
			y_val = y_val.detach().clone().requires_grad_(True).view(-1, 1)
			y_test = y_test.detach().clone().requires_grad_(True).view(-1, 1)
		
			data_dict = {
				'X_train': X_train, 'y_train': y_train,
				'X_val': X_val, 'y_val': y_val,
				'X_test': X_test, 'y_test': y_test
			}
		
		elif self.case == 'bike':
			base_path = '../Real_data/'
			traindf = pd.read_csv(os.path.join(base_path, 'bike_Train.csv'))
			testdf = pd.read_csv(os.path.join(base_path, 'bike_Test.csv'))
			validdf = pd.read_csv(os.path.join(base_path, 'bike_Valid.csv'))
			
			X_train, y_train = df2tensor_bike(traindf)
			X_val, y_val = df2tensor_bike(validdf)
			X_test, y_test = df2tensor_bike(testdf)
		
			data_dict = {
				'X_train': X_train, 'y_train': y_train,
				'X_val': X_val, 'y_val': y_val,
				'X_test': X_test, 'y_test': y_test
			}
		
		elif self.case == 'churn':
			path = '../Real_data/Churn.csv'
			if not os.path.exists(path):
				 print(f"Warning: {path} not found. Check directory.")
				 
			df = pd.read_csv(path)
			df = df.drop(['customerID'], axis=1)
			df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
			
			df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
			
			df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
			
			df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
			
			df = df.apply(lambda x: object_to_int(x))
			
			X = df.drop(columns=['Churn'])
			y = df['Churn'].values
			
			# Stratified Split
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40, stratify=y)
			
			num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
			
			scaler = StandardScaler()
			X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
			X_test[num_cols] = scaler.transform(X_test[num_cols]) # Note: 這裡 user 原 code 是 X_test，後面再 split 出 val
		
			# Split Val from Test
			X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=40, stratify=y_test)
		
			# Convert to Tensor
			X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
			X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
			X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
			
			y_train = torch.from_numpy(y_train).float().view(-1, 1)
			y_val = torch.from_numpy(y_val).float().view(-1, 1)
			y_test = torch.from_numpy(y_test).float().view(-1, 1)
		
			data_dict = {
				'X_train': X_train, 'y_train': y_train,
				'X_val': X_val, 'y_val': y_val,
				'X_test': X_test, 'y_test': y_test
			}
		
		elif self.case == 'year': 
            path = '../Real_data/YearPredictionMSD.txt'
            
            if os.path.exists(path):
                X_train, y_train, X_test, y_test = load_year_prediction_msd(path)
                split_idx = int(X_train.shape[0] * 0.9)
                X_val = X_train[split_idx:]
                X_train = X_train[:split_idx]
                
                y_val = y_train[split_idx:]
                y_train = y_train[:split_idx]
            else:
                 print(f"Warning: {path} not found. Generating dummy data.")
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            
            base = 1992
            y_train -= base
            y_val -= base
            y_test -= base
            
            data_dict = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
		
		else:
			raise ValueError(f"Unknown case: {self.case}")
		
		task = 'classification' if self.case not in ['ca', 'bike', 'year'] else 'regression'
		data_dict['task'] = task
			
		return data_dict

	def get_data(self):
		return self.data