import numpy as np

###################################################
###################################################
###################################################
nro_mics = 5
r_mics = 1
dist = 30
#Primero vamos a definir las posiciones del aparato
mics = np.zeros((nro_mics, 3))
delta =np.zeros((nro_mics, 3))

mics[0] = np.array([1, 0, 0])*r_mics
mics[1] = np.array([0, 1, 0])*r_mics
mics[2] = np.array([0, 0, 1])*r_mics
mics[3] = np.array([-1, 0, 0])*r_mics
mics[4] = np.array([0, -1, 0])*r_mics

delta[:,0] = 1
mics0 = mics + delta*dist/2
mics1 = mics - delta*dist/2

tot_mics = np.zeros((nro_mics*2,3))

tot_mics[:nro_mics, :] = mics0
tot_mics[nro_mics:, :] = mics1

###################################################
###################################################
###################################################

nro_puntos = 500000
nro_samples = 1
radio = 50
c = 350
f = 22000

delays_real = np.ones((nro_puntos*nro_samples, (nro_mics*2)*(nro_mics*2 -1)//2))
delays_medidos = np.ones((nro_puntos*nro_samples, (nro_mics*2)*(nro_mics*2 -1)//2))
fuentes_pos = np.ones((nro_puntos*nro_samples, 3))


for i in range(nro_puntos):
	fuente = np.ones(3)*radio*2
	while np.linalg.norm(fuente) > radio:
		#fuente = np.random.rand(3)*radio
		fuente = (np.random.rand(3)*2 - 1) *radio
	#Calculo los delays
	fuentes = np.ones((len(tot_mics),3)) #Es por el numero de microfonos
	fuentes = fuentes*fuente
	tiempos = np.linalg.norm((tot_mics - fuentes), axis = 1)/350
	fuentes_pos[i*nro_samples:(i+1)*nro_samples] *= fuentes[0]
	pos = 0
	for j in range(nro_mics*2):
		for k in range(j+1, nro_mics*2):
			delays_real[i*nro_samples:(i+1)*nro_samples, pos] *= tiempos[j] - tiempos[k]
			delays_medidos[i*nro_samples:(i+1)*nro_samples, pos] *= (tiempos[j] - tiempos[k]) + (np.random.rand(nro_samples)*2 -1) * (2/f)
			pos += 1

np.save('delays_real_test.npy', delays_real)
np.save('delays_medidos_test.npy', delays_medidos)
np.save('fuentes_pos_test', fuentes_pos)
