# IID
python3 app_nn.py --dataset mnist --model nn --partition_method iid --lr 0.01 --momentum 0.9 \
  --data_size_per_client 600 --client_num_in_total 5 --client_num_per_round 5 --comm_round 20 --epochs 5 \
  --backend MQTT --mqtt_host 10.0.137.51 --mqtt_port 61613 --trial 0

# non-IID, all labels from one class on one client
python3 app_nn.py --dataset mnist --model nn --partition_method noniid --partition_label uniform \
  --partition_alpha 1.0 --data_size_per_client 600 --lr 0.01 --momentum 0.9 \
  --client_num_in_total 5 --client_num_per_round 5 --comm_round 20 --epochs 5 \
  --backend MQTT --mqtt_host 10.0.137.51 --mqtt_port 61613 --trial 0

# non-IID, all labels from two class on one client
python3 app_nn.py --dataset mnist --model nn --partition_method noniid --partition_label uniform \
  --partition_alpha 0.5 --partition_secondary --data_size_per_client 600 --lr 0.01 --momentum 0.9 \
  --client_num_in_total 5 --client_num_per_round 5 --comm_round 20 --epochs 5 \
  --backend MQTT --mqtt_host 10.0.137.51 --mqtt_port 61613 --trial 0