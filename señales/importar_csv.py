import time
import numpy as np
import matplotlib.pyplot as plt


nombreArchivo = 'rampa_con errores.csv' #Aca va el nombre del archivo

ti = time.time()

data = np.genfromtxt(nombreArchivo, skip_header = 2, dtype = 'U')
overrunsChecking =  np.char.find(data,'Overruns')
overrunsPositions = np.where(overrunsChecking == 0)


counter = 0
overruns = 0
for pos in overrunsPositions[0]:
	val = int(data[pos + counter - overruns][9:])
	data = np.insert(data, pos+counter+1 - overruns, float(data[pos + counter - overruns-1])*np.ones(val))
	data = np.delete(data,pos+counter - overruns)
	counter += val
	overruns += 1


tf = time.time()

print('tardo:', tf-ti)





