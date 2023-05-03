#conda create --name ### cudatoolkit=11.0.3 cudnn=8.2.1

#conda install cudatoolkit=11.0.3 python=3.8
#download dgl cuda(11.0) version:  python -m pip install dgl-cu110==0.6.0
#python -m pip install torch==1.7.1+cu110 （torchvision==0.8.2+cu110 torchaudio==0.7.2） -f https://download.pytorch.org/whl/torch_stable.html

import random

import utils
from Models import HDRIL_Models
from torch import optim
import torch
import torch.nn as nn
import gc
from torch.utils import data
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# print(torch.__version__)
# print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device("cuda:0")   # 使用第一个 GPU 设备
if torch.cuda.is_available():
    device = torch.device("cuda")          # 如果有可用的 GPU 设备就使用 GPU
else:
    device = torch.device("cpu")           # 否则使用 CPU
  
import dgl



params = {


    "runs" : 60,
    "runsize" : 71,
    "load_model" : False,
    "train_model" : True,
    "n_epochs" : 2,
    "sequencelength" : 70,
    "n_hidden" :  21,
    "n_latent" : 512,
    "n_iters" : 20,
    "lr_encoder": .00005,
    "lr_decoder": .00005

}

PATH = 'planningmodel.pt'



trainsize = params["runs"]*params["runsize"]##2500*70
# trainsize = 4260##total 6087
train_set = utils.BaxterDataset()


input_size = train_set.columns.size()[1]##数据集的列数
maxsize = len(train_set)##数据集的行数
sequencelength = params["sequencelength"]

trainingdata = data.Subset(train_set, indices=list(range(0, trainsize)))
testdata = data.Subset(train_set, indices=list(range(trainsize, maxsize)))




g2 = dgl.DGLGraph().to(device)
nodes = input_size
g2.add_nodes(nodes)
# A couple edges one-by-one
for i in range(0, nodes):
    for j in range(0, nodes):
        g2.add_edge(i, j)


"""The model"""
# params = {


#     "runs" : 2500,
#     "runsize" : 70,
#     "load_model" : False,
#     "train_model" : True,
#     "n_epochs" : 2,
#     "sequencelength" : 70,
#     "n_hidden" :  21,
#     "n_latent" : 512,
#     "n_iters" : 20,
#     "lr_encoder": .00005,
#     "lr_decoder": .00005

# }
input_size = input_size
output_size = input_size
##n_latent 是一个超参数，表示我们希望将输入数据映射到的潜在空间（latent space）的维度大小
#在变分自编码器中，潜在空间通常被限制为一个低维的空间，因为这有助于学习到数据的低维表示，并且可以减少过度拟合的风险。
#通常，较小的 n_latent 会导致更少的模型参数，更快的训练速度，但可能会牺牲一些模型的表示能力。而较大的 n_latent 会使模型具有更强的表示能力，但会增加过度拟合的风险，并增加训练时间和计算资源的需求。
n_latent = params["n_latent"]
n_iters = params["n_iters"]

current_loss = 0
test_loss_total = 0
all_losses = []
test_losses = []
losses = []

force = []


gcn1 = HDRIL_Models.PlanningGAT(g2, int(input_size / nodes), n_latent, input_size, 1)

decoder = HDRIL_Models.PlanningDecoder(input_size, 1, n_latent)

encoder_optimizer = optim.Adam(gcn1.parameters(), lr= params["lr_encoder"])
decoder_optimizer = optim.Adam(decoder.parameters(), lr= params["lr_decoder"])

criterion = nn.CrossEntropyLoss()

seqmodel = HDRIL_Models.PlanningVAE(input_size, output_size, n_latent, gcn1, decoder, encoder_optimizer,
                                    decoder_optimizer,
                                    criterion)

seqmodel = seqmodel.cuda()



"""Model training"""

if (params["load_model"] == True):
    seqmodel.load_state_dict(torch.load(PATH))



if (params["train_model"] == True):



    for epoch_idx in range(params["n_epochs"]):

        print(epoch_idx)

        i = 0

        while i < trainsize:


            print('hahahahahahahah')
            s = random.randint(0, params["runs"] - 1) * params["runsize"]
            primitive = train_set.labels[s]


            c, p = train_set[s:s + sequencelength]
            c1, _ = train_set[s + 1:s + 1 + sequencelength]
            i += sequencelength

            rows = int(c.size()[0] / sequencelength)

            startcord = c.reshape(rows, sequencelength, input_size).transpose(0, 1).cuda()
            endcord = c1.reshape(rows, sequencelength, input_size).transpose(0, 1).cuda()
            target = p.reshape(rows, sequencelength, 1).transpose(0, 1).cuda()

            trainloss = 0
            testloss = 0

            avgloss = seqmodel.train(startcord.float(), target.float())

            trainavgloss = avgloss

            print(trainavgloss)

            if trainloss > 100000:
                break
            losses.append(trainavgloss)

            size = 0

            iteration = 0
            j = trainsize
            # gc.collect()
            # torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated(device=device))
            print(torch.cuda.max_memory_allocated(device=device))

    print(losses)
    print(test_losses)

    torch.save(seqmodel.state_dict(), PATH)







