import pandas as pd
import numpy as np

def prune_data(file):

	file.drop(['customerID'], axis=1, inplace=True)

	file.loc[file['MultipleLines'] == 'No phone service', 'MultipleLines'] = 'No'
	file.loc[file['OnlineSecurity'] == 'No internet service', 'OnlineSecurity'] = 'No'
	file.loc[file['OnlineBackup'] == 'No internet service', 'OnlineBackup'] = 'No'
	file.loc[file['DeviceProtection'] == 'No internet service', 'DeviceProtection'] = 'No'
	file.loc[file['TechSupport'] == 'No internet service', 'TechSupport'] = 'No'
	file.loc[file['StreamingTV'] == 'No internet service', 'StreamingTV'] = 'No'
	file.loc[file['StreamingMovies'] == 'No internet service', 'StreamingMovies'] = 'No'

	sex = pd.get_dummies(file['gender'], drop_first=True)
	sex = sex.rename(columns={"Male": "Sex"})

	partner = pd.get_dummies(file['Partner'], drop_first=True)
	partner = partner.rename(columns={"Yes": "Partner"})

	dependents = pd.get_dummies(file['Dependents'], drop_first=True)
	dependents = dependents.rename(columns={"Yes": "Dependents"})

	phone_service = pd.get_dummies(file['PhoneService'], drop_first=True)
	phone_service = phone_service.rename(columns={"Yes": "PhoneService"})

	multiple_lines = pd.get_dummies(file['MultipleLines'], drop_first=True)
	multiple_lines = multiple_lines.rename(columns={"Yes": "MultipleLines"})

	internet_service = pd.get_dummies(file['InternetService'])
	internet_service = internet_service.drop('No', axis=1)

	online_security = pd.get_dummies(file['OnlineSecurity'], drop_first=True)
	online_security = online_security.rename(columns={"Yes": "OnlineSecurity"})

	online_backup = pd.get_dummies(file['OnlineBackup'], drop_first=True)
	online_backup = online_backup.rename(columns={"Yes": "OnlineBackup"})

	device_protection = pd.get_dummies(file['DeviceProtection'], drop_first=True)
	device_protection = device_protection.rename(columns={"Yes": "DeviceProtection"})

	tech_support = pd.get_dummies(file['TechSupport'], drop_first=True)
	tech_support = tech_support.rename(columns={"Yes": "TechSupport"})

	streaming_tv = pd.get_dummies(file['StreamingTV'], drop_first=True)
	streaming_tv = streaming_tv.rename(columns={"Yes": "StreamingTV"})

	streaming_movies = pd.get_dummies(file['StreamingMovies'], drop_first=True)
	streaming_movies = streaming_movies.rename(columns={"Yes": "StreamingMovies"})

	contract = pd.get_dummies(file['Contract'])

	paperless_billing = pd.get_dummies(file['PaperlessBilling'], drop_first=True)
	paperless_billing = paperless_billing.rename(columns={"Yes": "PaperlessBilling"})

	payment_method = pd.get_dummies(file['PaymentMethod'])

	churn = pd.get_dummies(file['Churn'], drop_first=True)
	churn = churn.rename(columns={"Yes": "Churn"})

	file.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
			'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
			'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], axis=1, inplace=True)

	file = pd.concat([file, sex, partner, dependents, phone_service, multiple_lines, internet_service,
			online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
			contract, paperless_billing, payment_method, churn], axis=1)

	file['TotalCharges'] = pd.to_numeric(file['TotalCharges'], errors='coerce')

	file = file.dropna()

	return file
