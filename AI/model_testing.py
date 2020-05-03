#Cargo el modelo ya entrenado y lo testeo

import numpy as np
import matplotlib.pyplot as plt
import torch

#Cargo los datos de testeo
tst_delay = np.load('delays_medidos_test.npy')
tst_tgt = np.load('fuentes_pos_test.npy')

tst_data = np.zeros((len(tst_delay), len(tst_delay[0]) + 3)) 
tst_data[:, :len(tst_delay[0])] = tst_delay
tst_data[:, len(tst_delay[0]):] = tst_tgt

#Uso el data loader para tener todo bien puesto 
BB = 1 #Batch size del dataset de testeo
tst_load = torch.utils.data.DataLoader(dataset = tst_data, batch_size = BB, shuffle = True)

N = len(tst_data[0])
C = 3 #Cantidad de unidades de salida son 3 porque es la salida x, y, z

#Declaro la clase de mlp
class MLP(torch.nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(self,x):
        h = x
        for hidden in self.layers[:-1]:
            h = torch.tanh(hidden(h))
        output = self.layers[-1]
        y = output(h)
        return y

arqs = [[256], [512], [64, 32], [64, 64], [128, 64], [128, 128], [128, 64, 32] , [256, 128], [32, 32, 32], [64, 64, 64], [32,32,32,32], [64,64,64,64], [128,64,32,16]]
arq = [[64,64,64,64]] ##############Elijo que arquitectura quiero

arq.insert(0,N)
arq.append(C)
model = MLP(arq)

###############################################################################

###CARGO MI MODELO YA ENTRENADO
model.load_state_dict(torch.load('0'))

###TESTEO EL MODELO
model.eval() #Empiezo a evaluar el modelo


#Funcion de costo que uso
costf = torch.nn.MSELoss()

e = [] #Cada elemento de e va a ser el error de cada batch del tst_load
counter = 0
with torch.no_grad():
    for row in tst_load: #Esto es un minibatch
        counter += 1
        if counter%50000 == 0:
            print(counter)
        x = row[:, :len(tst_delay[0])].float()
        z = row[:, len(tst_delay[0]):].float()
        y = model(x)
        error = costf(y,z)
        e.append(error.item())
        
    error_promedio = np.mean(e) #El error en cada epoca es el promedio del error que obtuve en cada batch, en esa epoca
    desviacion_error = np.std(e) #Desviacion estandar de los errores
    
print('El error promedio de la arquitectura es ' + str(error_promedio))
print('El desvio estandar de la arquitectura es ' + str(desviacion_error))

print('#####')
#El error real que queremos (distancia entre la fuente real y la obtenida) no es exactamente lo que me da la cost function MSELoss
#Aca defino bien el error
e_posta = np.zeros(len(e))
for i in range(len(e)):
    e_posta[i] = np.sqrt(e[i]*3)
    
#np.save('11_errores_testeo_m', e_posta)
    
e_posta_prom = np.mean(e_posta)
e_posta_desv = np.std(e_posta)

print('El error promedio posta es ' + str(e_posta_prom) + 'm')
print('El desvio estandar posta es ' + str(e_posta_desv) + 'm')

######################
'''
#Ploteo los errores de entrenamiento
err  = np.load('10.npy')

f1 = plt.figure()
plt.plot(err,'.-')
plt.title('Errores de entrenamiento')
'''

'''
#Para ver los pesos de una capa:
f = plt.plot()
plt.imshow(np.abs(model.layers[0].weight.detach().numpy()), cmap = 'plasma')
plt.colorbar(label = 'Módulo de las activaciones')
plt.xlabel('Diferencias temporales')
plt.ylabel('Neuronas de la primera capa oculta')
'''
