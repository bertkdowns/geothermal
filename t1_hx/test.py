import pickle
import pandas
import os
import torch

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


head = df.head(1000)

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
 

# surrogate model the preheater
# Inputs: BCOCF, PHMFIT, SIF, 
# Output: PHBOT

X = head[['BCOCF', 'PHMFIT', 'SIF']]
y = head['PHBOT']

# convert to torch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

from model import MLP, train_model, empirical_loss, test_model

m = MLP(input_dim=X_tensor.shape[1], hidden_dim=32, output_dim=1)

train_model(m, X_tensor, y_tensor, epochs=1000000, lr=1e-3)

predictions = test_model(m, X_tensor, y_tensor)


# plot with BCOCF on x-axis and PHBOT on y-axis
import matplotlib.pyplot as plt
plt.scatter(X_tensor[:, 2].numpy(), y_tensor.numpy(), label='True PHBOT', alpha=0.5)
plt.scatter(X_tensor[:, 2].numpy(), predictions.numpy(), label='Predicted PHBOT', alpha=0.5)
plt.xlabel('BCOCF')
plt.ylabel('PHBOT')
plt.legend()
plt.show(block=True)