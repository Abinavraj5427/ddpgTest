import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
str = ""
with open("graph2.txt","r") as file:
	str = file.readlines()
	str = str[0]
	floatArr = np.array(re.findall('\d+\.\d+', str))
	floatArr = floatArr.astype(np.float32)
	# print(floatArr)
	# plt.plot(floatArr)
	# plt.show()
	avg = []
	for x in range(floatArr.shape[0]):
		avg.append(np.mean(floatArr[max(0, x-40):x]))
	plt.plot(avg)
	plt.show()
