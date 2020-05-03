##
import numpy as np
import matplotlib.pyplot as plt
import torch

#Primero cargo los datos que voy a usar para las pruebas de efectividad

trn_delay = np.load('delays_medidos.npy')
trn_tgt = np.load('fuentes_pos.npy')

trn_data = np.zeros( (len(trn_delay), len(trn_delay[0]) + 3) ) 
trn_data[:, :len(trn_delay[0])] = trn_delay
trn_data[:, len(trn_delay[0]):] = trn_tgt

#trn_data son los datos de entrenamiento 
print('Tamaño del dataset:', len(trn_delay))
####

#seteo los parametros del dataset
B = 32 #batch_size
P = len(trn_data) #Longitud de mi dataset
N = len(trn_delay[0]) #Cantidad de unidades de entrada
C = 3 #Cantidad de unidades de salida son 3 porque es la salida x, y, z
#####
#Uso el data loader para tener todo bien puesto 
trn_load = torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True)

#Declaro la clase de mlp
class MLP(torch.nn.Module):
    def __init__(_,sizes):
        super().__init__()
        _.layers = torch.nn.ModuleList()
        for i in range(len(sizes)-1):
            _.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(_,x):
        h = x
        for hidden in _.layers[:-1]: #Todas las capas ocultas que no son la ultima, les pongo una sigmoid
            h = torch.tanh(hidden(h))
        output = _.layers[-1] #A la ultima capa le pongo una softmax
        y = output(h)
        #y = output(h)
        return y
#########

#arqs = [ [256], [512], [64, 32], [64, 64], [128, 64], [128, 128], [128, 64, 32] , [256, 128], [32, 32, 32], [64, 64, 64], [32,32,32,32], [64,64,64,64], [128,64,32,16]]
arqs = [[64,64,64,64], [0]] 
i = 0

for arq in arqs:
	arq.insert(0,N)
	arq.append(C)
	model = MLP(arq)
	costf = torch.nn.MSELoss()
	optim = torch.optim.RMSprop(model.parameters(),lr = 1e-4)

	E, t = 1, 0
	errs = []

	model.train() #Le aviso al programa que voy a empezar a entrenar

	while  E > 1e-5 and t < 1500: #Dos condiciones para poder salir 
	    e = []
	    for row in trn_load: #Esto es un minibatch
	        optim.zero_grad() #Reseteo los gradientes
	        x =  row[:, :len(trn_delay[0])].float() #El -1 significa que no le especifico ese tamano, solo me importa el N. #Osea calculo la cantidad de filas a mano para que este todo bien
	        #x = x[:,[0,1,2,3,9,10,11,17,18,24,35,36,37,38,39,40,41,42,43,44]]
	        z =  row[:, len(trn_delay[0]):].float() #Tantas filas como labels me vinieron en mi dataset y tantas columnas como clases tengo
	        y = model(x) #Calculo el modelo para mis inputs x
	        error = costf(y,z) #Calculo el error
	        error.backward() #'Propago' los errores
	        optim.step() #Aplico el gradiente
	        e.append(error.item()) #Estos son los errores dentro del batch
	        
	    t += 1 #Recien cuando termino todos los batch, termine una epoca
	    E = sum(e)/len(e) #El error en cada epoca es el promedio del error que obtuve en cada batch, en esa epoca
	    errs.append(E)
	    if t%10 == 0:
	        print(t) #Cada 10 epocas, printeo el numero de epoca
	        print('error', E)

	model.eval() #Le aviso al programa que apartir de ahora voy a emepzar a evaluar el modelo

	nombre = str(i)
	np.save(nombre + '.npy' , errs)
	torch.save(model.state_dict(), nombre)
	i += 1
