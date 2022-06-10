import time
import pandas as pd

#Revisied

PHASE_FILE = './phase.txt'
POWER_FILE = './power.txt'

trainstart = []
trainend = []
sendstart = []
sendend = []

ROUND_NUM = 10

rounds_train_time = []
rounds_comm_time =[]

rounds_train_power = []
rounds_train_energy =[]

rounds_comm_power = []
rounds_comm_energy =[]


def cal_mean(data):
    n = len(data)
    mean = sum(data) / n
    return mean



def cal_var(data):
    n = len(data)
    mean = sum(data) / n
    deviation = [(x - mean)**2 for x in data]
    var = sum(deviation) / (n-1)
    return var


def power(start_time,end_time,file):
    start_time = start_time*1000
    end_time = end_time*1000
    df = pd.read_csv(file)
    df=df.drop(['volt', 'curr','pf'], axis=1)
    df = df[(df["time_stamp"]>start_time) & (df["time_stamp"]<end_time)]
    
    dur = (end_time - start_time) / 1000
    avg_watt = df['mul'].sum() / len(df)
    energy = avg_watt * dur
    return avg_watt, energy
    


with open(PHASE_FILE,'r') as phaseFile:
    data = []
    for line in phaseFile: data.append(line.strip())

    for i in range(ROUND_NUM):
        trainstart.append(float(data[i + 1]))
        trainend.append(float(data[i + ROUND_NUM + 2]))
        sendstart.append(float(data[i + ROUND_NUM*2 + 3]))
        sendend.append(float(data[i + ROUND_NUM*3 + 4]))


for i in range(len(trainstart)):
    rounds_train_time.append(trainend[i] - trainstart[i])
    rounds_comm_time.append(sendend[i] - sendstart[i])


Average_train_time = cal_mean(rounds_train_time)
Average_train_time_var = cal_var(rounds_train_time)
print(rounds_train_time)
Average_comm_time = cal_mean(rounds_comm_time)
Average_comm_time_var = cal_var(rounds_comm_time)




for i in range(len(trainstart)):
    a,b = power(trainstart[i],trainend[i],POWER_FILE)
    rounds_train_power.append(a)
    rounds_train_energy.append(b)
    
    c,d = power(sendstart[i],sendend[i],POWER_FILE)
    rounds_comm_power.append(c)
    rounds_comm_energy.append(d)


Average_train_power = cal_mean(rounds_train_power)
Average_train_power_var = cal_var(rounds_train_power)

Average_comm_power = cal_mean(rounds_comm_power)
Average_comm_power_var = cal_var(rounds_comm_power)




print("Average_train_time  "+str(Average_train_time))
print("Average_train_time_var  "+str(Average_train_time_var))
print("-------------------------------")
print("Average_comm_time  "+str(Average_comm_time))
print("Average_comm_time_var  "+str(Average_comm_time_var))


print("-------------------------------")



print("Average_train_power  "+str(Average_train_power))
print("Average_train_power_var  "+str(Average_train_power_var))
print("-------------------------------")
print("Average_comm_power  "+str(Average_comm_power))
print("Average_comm_power_var  "+str(Average_comm_power_var))









