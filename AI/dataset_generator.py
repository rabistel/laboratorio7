import numpy as np

####### Parametros Editables ###############
default 		= True
nro_puntos 		= 30
nro_samples 	= 7
radio 			= 50
c 				= 350
f 				= 22000
error_medicion 	= 2/f
# -----------------------------------------
nro_mics 		= 4
tot_mics 		= np.zeros((nro_mics, 3))
tot_mics[0] 	= np.array([1,0,0])
tot_mics[1] 	= np.array([0,1,0])
tot_mics[2] 	= np.array([0,0,1])
tot_mics[3] 	= np.array([1,0,1])
############################################

if default:
	###################################################
	###################################################
	###################################################
	nro_mics = 10
	r_mics = 1
	dist = 30

	mics = np.zeros((nro_mics/2, 3))
	delta =np.zeros((nro_mics/2, 3))

	mics[0] = np.array([1, 0, 0])*r_mics
	mics[1] = np.array([0, 1, 0])*r_mics
	mics[2] = np.array([0, 0, 1])*r_mics
	mics[3] = np.array([-1, 0, 0])*r_mics
	mics[4] = np.array([0, -1, 0])*r_mics

	delta[:,0] = 1
	mics0 = mics + delta*dist/2
	mics1 = mics - delta*dist/2

	tot_mics = np.zeros((nro_mics,3))

	tot_mics[:int(nro_mics/2), :] = mics0
	tot_mics[int(nro_mics/2):, :] = mics1

	###################################################
	###################################################
	###################################################
	
delays_real 	= np.ones((nro_puntos*nro_samples, (nro_mics)*(nro_mics -1)//2))
delays_medidos 	= np.ones((nro_puntos*nro_samples, (nro_mics)*(nro_mics -1)//2))
fuentes_pos 	= np.ones((nro_puntos*nro_samples, 3))

for i in range(nro_puntos):
	fuente = np.ones(3)*radio*2
	while np.linalg.norm(fuente) > radio:
		fuente = (np.random.rand(3)*2 - 1) *radio
	fuentes = np.ones((len(tot_mics),3)) 
	fuentes = fuentes*fuente
	tiempos = np.linalg.norm((tot_mics - fuentes), axis = 1)/c
	fuentes_pos[i*nro_samples:(i+1)*nro_samples] *= fuentes[0]
	pos = 0
	for j in range(nro_mics):
		for k in range(j+1, nro_mics):
			delays_real[i*nro_samples:(i+1)*nro_samples, pos] *= tiempos[j] - tiempos[k]
			delays_medidos[i*nro_samples:(i+1)*nro_samples, pos] *= (tiempos[j] - tiempos[k]) + (np.random.rand(nro_samples)*2 -1) * (error_medicion)
			pos += 1

np.save('delays_real.npy', delays_real)
np.save('delays_medidos.npy', delays_medidos)
np.save('fuentes_pos.npy', fuentes_pos)



