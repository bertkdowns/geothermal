# OPC UA Server that simulates the process being running. 
# it reads the data from a pickle file and serves it via OPC UA protocol.
# Values are updated every second.

import pickle
#import pandas
#import os
#import torch
#import seaborn as sns
#import matplotlib.pyplot as plt
import asyncio
import logging
from asyncua import Server, ua
from asyncua.common.methods import uamethod


with open('./reconcile_all.pk1', 'rb') as f:
    df = pickle.load(f)


@uamethod
def func(parent, value):
    return value * 2


async def main():
    _logger = logging.getLogger(__name__)
    # setup our server
    server = Server()
    await server.init()
    # Load cert + key
    # server.load_certificate("server_cert.der")
    # server.load_private_key("server_private_key.pem")

    # # Enable encryption
    # server.set_security_policy([
    #     ua.SecurityPolicyType.Basic128Rsa15_Sign,
    #     ua.SecurityPolicyType.Basic128Rsa15_SignAndEncrypt,
    # ])
    server.set_endpoint("opc.tcp://0.0.0.0:4841/freeopcua/server/")

    # set up our own namespace, not really necessary but should as spec
    uri = "http://examples.freeopcua.github.io"
    idx = await server.register_namespace(uri)

    # populating our address space
    # server.nodes, contains links to very common nodes like objects and root
    myobj = await server.nodes.objects.add_object(idx, "GeothermalProcess")
    vars = {}
    for col in df.columns:
        # Add each column as a variable to the OPC UA server
        vars[col] = await myobj.add_variable(idx, col, float(df[col].iloc[0]))
        #await var.set_writable()  # Set the variable to be writable by clients

    await server.nodes.objects.add_method(
        ua.NodeId("ServerMethod", idx),
        ua.QualifiedName("ServerMethod", idx),
        func,
        [ua.VariantType.Int64],
        [ua.VariantType.Int64],
    )


    _logger.info("Starting server!")
    async with server:
        row_index = 0
        while True:
            await asyncio.sleep(1)
            row_index = (row_index + 1) % len(df)
            for col, myvar in vars.items():
                await myvar.write_value(float(df[col].iloc[row_index]))
            _logger.info("Updated values using row %.0f", row_index)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main(), debug=True)