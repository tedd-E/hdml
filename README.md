# FedML-HD: Federated Learning with Hyperdimensional Computing
The aim of this project is to explore and evaluate the performance of HD computing in Federated Learning as compared with traditional federated learning techniques through a physical deployment using Raspberry Pis. Our metrics are based on the accuracy, communication cost, energy efficiency, and robustness against noise of the system.

## How to set up and run FedML-HD across a network of Raspberry Pis
This project runs on a network of Raspberry Pis. The provided resources are guaranteed to run on the Raspberry Pi 4 and 400 with the Raspbian Bullseye OS. Running the code on other RPI or OS versions may require additional configuration.

To run the deployment, there are two necessary systems to configure. The first component of the network is the server side, with the source code located in the folder FedML-Server-HD from the github repository. The second is the client side, with the source code located in the folder FedML-IoT-HD.
### Setting up the environment
The setup process is the same for both the server and client side. 

To set up the Raspberry Pi itself, follow the instructions from the official documentation here. Select Bullseye for your operating system.

To install the necessary package dependencies, first download the requirements.txt file from the repository. Navigate to the directory where the file is saved and run the following commands:

- ```sudo apt-get update```
- ```pip3 install -r requirements.txt ```

Clone the github repository with the following command: 

```git clone https://github.com/tedd-E/hdml```

### Running the server
1. After completing the environment setup, navigate to the following filepath: hdml/FedML-Server-HD/mqtt
2. Start the mosquitto server by running the following command: 

    ```./run_mosquitto_server.sh```

3. To run the server, navigate to ```hdml/FedML-Server-HD/executor``` and run app_hd.py (make sure to specify python3 as the python version). You can adjust the training parameters and set the number of clients to communicate with by changing the flags. If not, the default parameters are: dataset=mnist, data partition=IID, preprocessed partition=true,  hypervector dimension=10000, client number in total/per round=8, batch size=100, learning rate=0.01, epochs=1.

    - --D: number of dimensions of the hypervector
    - --dataset: Our supported datasets are cifar10, mnist, and fashionmnist
    - --client_num_in_total: Total number of clients throughout training
    - --client_num_per_round: Number of clients training each round
    - --epochs: Number of epochs to run the training on the specified dataset
    - --comm_round: Number of rounds per epoch
    - --batch_size: Size of batch from the specified dataset to be distributed to each client
    - --lr: Specify the learning rate parameter

    An example command for a deployment using the CIFAR10 dataset on 5 clients, with 20 epochs and 10 communication rounds would be:

    ```python3 app_hd.py --dataset cifar10 --client_num_in_total 5 --client_num_per_round 5 --epochs 20 --comm_round 20```

4. Once the command is run, the server should load the data and display information on the server IP address as it waits for the clients to connect. Save this address to reference when running the clients.

5. The server will automatically start training once it detects that the amount of connected clients matches the specified number given to it during startup. All training information will be displayed in the terminal. 

### Running the clients
The process for running the clients is the same across all clients. Repeat the following for as many clients as you are using in the deployment.

1. Before starting the client, follow the instructions to start up the server.

2. After completing the environment setup, navigate to the following filepath: ```hdml/FedML-IoT-HD/raspberrypi/fedhd```

3. Run the following command, and replace the X with a unique nonzero integer and the default server IP (127.0.0.1) with the server IP shown in the terminal of the server following its startup. The client uuid is just meant to be an identifying tag so the server can differentiate each client, so the specific value doesn’t matter as long as it’s not 0 (since the server’s id is 0) and unique from other clients.

    ```python3.7  fedhd_rpi_client.py --client_uuid X --server_ip http://127.0.0.1```


    The server will automatically start training once it detects that the amount of connected clients matches the specified number given to it during startup.

### Measuring Power
To record power measurements on a client device we used the following repository:

https://github.com/UCSD-SEELab/powermeter

This is meant to be used for the HIOKI 3334 powermeter, available in SEELab UCSD.

To use:
1. Connect the powermeter to the server device (will record the power measurements)
2. Connect the server to the client device you want to measure the power of using an Ethernet cable
3. Connect the power of the client device to the powermeter outlet
4. The power recording script can be found in the github along with the scripts used to parse the log and calculate the energy consumption.

Explanations of power scripts:

- powerdemo.py
    - Adjust run time
    - Records power for the run duration specified and saves it in a file called ```power.txt```
- phase_calc.py
    - This file was used to separate the power log into different phases of the Federated Learning process. It uses ```phase.txt``` from below and ```power.txt``` which is the power log
    - Outputs (prints to console):
        - Average training time & variance
        - Average communication time & variance
        - Average pain and suffering & variance
        - Average training power & variance
        - Average communication power & variance
- phase_log.py
    - Set Host IP
    - Listens for messages from the client and records the time for each phase
    - Saves output in ```phase.txt```
<br></br>
## Acknowledgements
This project builds upon the open libary provided by the creators of [FedML-AI](https://github.com/FedML-AI). 

A special thanks to Xiaofan Yu and Rishikanth Chandrasekaran for generously providing their time and support towards this project.