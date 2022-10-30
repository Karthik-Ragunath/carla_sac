import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
import collections

actions_list = []
with open('/home/asf170004/data/50KParallel/imitation_training_data_55k.pkl','rb') as af:
    #for i in range(54527):
    actions_list = pickle.load(af)

print("Length of Data: ", len(actions_list))
# for data in actions_list:
#     print(data)
steer_list = [round(data[0],2) for data in actions_list]

dict = {}

for i in range(len(steer_list)):
    if steer_list[i] in dict:
        dict[steer_list[i]] += 1
    else:
        dict[steer_list[i]] = 1


labels = list(dict.keys())
freq = list(dict.values())

import matplotlib.pyplot as plt
plt.scatter(labels, freq)
plt.savefig("/home/asf170004/imitation_learning/_out/55ksteering.png")
#plt.show()
