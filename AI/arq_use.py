#
# Este codigo tiene como fin aplicar el modelo a un unico set de delays
#

import numpy as np
import torch

####### Parametros Editables ###############
tst_delay = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
c_real = 343
c_dataset = 350
modelo = 'modelo_140_epocas'
arq = [64,64,64,64] 
############################################

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu") 
tst_delay = tst_delay.reshape(1,len(tst_delay))
tst_delay = tst_delay*(c_real/c_dataset)
tst_delay = torch.Tensor(tst_delay)
N = len(tst_delay[0])
C = 3

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
model.load_state_dict(torch.load(modelo))
model.eval()
res = model(tst_delay)
print(res)

