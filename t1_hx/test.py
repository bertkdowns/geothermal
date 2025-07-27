import pickle
import pandas
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from model import MLP, train_model, empirical_loss, test_model

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)

with open(os.path.join(parent_dir, 'reconcile_all.pk1'), 'rb') as f:
    data = pickle.load(f)

# Now 'data' contains the loaded object from the pickle file

print(data)
# Print the column names
print(data.columns)


# get the first 1000 rows of the dataframe
df = pandas.DataFrame(data)

# AH - Air Humidity (0-100%) AKA Relative Humidity
# AT - Air Temperature (°C)
# BCOCF - Brine-Condensate Outlet Combined flow 
# CIT - Condenser Inlet Temperature
# COT - Condenser Outlet Temperature
# CP - Condenser Pressure (Outlet) bar
# KW - Parasitic load (kW)
# MW - Turbine power 
# NCGOT - Non-condensable gas outlet temperature
# PHBOT - Preheater brine outlet temperature
# PHMFIT - Preheater motive fluid inlet temperature
# PHMFOT - Preheater motive fluid outlet temperature
# SD - Pump speed
# SIF - Steam inlet flow
# SIT - Steam inlet temperature
# TOT - Turbine outlet temperature
# VMFL - vaporizer motive fluid level
# VMFOT - vaporizer motive fluid outlet temperature
# VP - vaporizer pressure
 

# do a pairplot of the input features



head = df[1000:2000]
# sns.pairplot(head[['BCOCF', 'PHMFIT', 'SIF','SIT', 'PHBOT']], hue='PHBOT', palette='viridis')
# plt.show(block=True)

# # Do a line graph of each feature over time
# plt.figure(figsize=(16, 10))
# for col in head.columns:
#     plt.plot(head.index, head[col], label=col)
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Features over time')
# plt.ylim(0, 300)
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
# plt.tight_layout()
# plt.show(block=True)


# surrogate model the preheater
# Inputs: BCOCF, PHMFIT, SIF, 
# Output: PHBOT

X = df[['BCOCF', 'PHMFIT', 'SIF', 'SIT']][0:3000]
y = df['PHBOT'][0:3000]


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pandas.DataFrame(X_scaled, columns=X.columns)
# y = y.values.reshape(-1, 1)
# y_scaler = StandardScaler()
# y_scaled = y_scaler.fit_transform(y)
# y = pandas.Series(y_scaled.flatten())

X_train = X[1000:2000]
y_train = y[1000:2000]
X_test = X[2000:3000]
y_test = y[2000:3000]

# convert to torch tensors
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)


# Create a MLP model

MyMLP = nn.Sequential(
    nn.Linear(X_tensor.shape[1], 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

train_model(MyMLP, X_tensor, y_tensor, epochs=1000, lr=1e-3)

predictions = test_model(MyMLP, X_tensor, y_tensor)


# plot with BCOCF on x-axis and PHBOT on y-axis
plt.scatter(X_tensor[:, 1].numpy(), y_tensor.numpy(), label='True PHBOT', alpha=0.5)
plt.scatter(X_tensor[:, 1].numpy(), predictions.numpy(), label='Predicted PHBOT', alpha=0.5)
plt.xlabel('BCOCF')
plt.ylabel('PHBOT')
plt.legend()
plt.show(block=True)

# Plot true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_tensor.numpy(), predictions.numpy(), alpha=0.5)
plt.plot([y_tensor.min(), y_tensor.max()], [y_tensor.min(), y_tensor.max()], 'r--', lw=2)
plt.xlabel('True PHBOT')
plt.ylabel('Predicted PHBOT')
plt.title('True vs Predicted PHBOT')
plt.xlim(y_tensor.min(), y_tensor.max())
plt.ylim(y_tensor.min(), y_tensor.max())
plt.grid()
plt.show(block=True)
