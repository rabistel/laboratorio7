import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('rampa_con errores.csv', skip_header = 2, dtype = 'U')
overrunsChecking =  np.char.find(data,'Overruns')
overrunsPositions = np.where(overrunsChecking == 0)


counter = 0
overruns = 0
for pos in overrunsPositions[0]:
	val = int(data[pos + counter - overruns][9:])
	data = np.insert(data, pos+counter+1 - overruns, np.zeros(val))
	data = np.delete(data,pos+counter - overruns)
	counter += val
	overruns += 1













