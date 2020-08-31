#
# Este codigo tiene como fin el testeo del modelo con un dataset particular
#
import numpy as np
import matplotlib.pyplot as plt
import torch

####### Parametros Editables ###############
arq             = [64,64,64,64] 
BB              = 500
modelo_testeo   = 'modelo' 
############################################

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

tst_delay   = np.load('delays_medidos_test.npy')
tst_tgt     = np.load('fuentes_pos_test.npy')

tst_data = np.zeros((len(tst_delay), len(tst_delay[0]) + 3)) 
tst_data[:, :len(tst_delay[0])] = tst_delay
tst_data[:, len(tst_delay[0]):] = tst_tgt

tst_load = torch.utils.data.DataLoader(dataset = tst_data, batch_size = BB, shuffle = False)
N = len(tst_delay[0])
C = 3 

print('Tamaño del dataset:', len(tst_delay))

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


arq.insert(0,N)
arq.append(C)
model = MLP(arq).to(device)
model.load_state_dict(torch.load(modelo_testeo))
costf = torch.nn.MSELoss(reduce = False)

model.eval()
e = np.array([]) 

with torch.no_grad():
    for row in tst_load: 
        x        = row[:, :len(tst_delay[0])].float().to(device)
        z        = row[:, len(tst_delay[0]):].float().to(device)
        y        = model(x)
        error    = costf(y,z).sum(dim=1).sqrt()
        e        = np.append(e,error.detach().numpy())
        
error_promedio   = np.mean(e)
desviacion_error = np.std(e) 
median_error     = np.median(e)

print('---------------------------------------- FIN DEL TESTEO ----------------------------------------')    
print('El error promedio del testeo para la arquitectura es: '  + str(error_promedio)   + ' m')
print('El desvio estandar del testeo para la arquitectura es: ' + str(desviacion_error) + ' m')
print('La mediana del testeo para la arquitectura es: '         + str(median_error)     + ' m')

np.save('errores_testeo_' + modelo_testeo, e)




