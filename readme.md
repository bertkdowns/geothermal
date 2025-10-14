
# Geothermal Data Project

This project tries to build some DT tools using some geothermal data.

See also: [my writeup](https://bertkdowns.github.io/thesis/notes/5nanq7k2v18kt5y3byvw0pa/)

A diagram of the flowsheet is avaliable [here](https://bertkdowns.github.io/thesis/notes/i8l8oat3t5nubw5ls0tdglm/)

# OPCUA Geothermal server

This just loads in a row of data from the geothermal data file every second, 
simulating what it might be like if you were connected to an actual opc plant.

```
cd opcua-server
pip install -r requirements.txt
python server.py
```

Running the client:

```
pip install opcua-client
opcua-client
```


