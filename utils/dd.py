import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import tqdm as tq

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


IM_SIZE = (20,256)
BATCH_SIZE = 8
LR = 0.002
CHANNEL = 1 
IMG_LR = 0.1

class SyntheticDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def distance(grad1, grad2):
    dist = torch.tensor(0.0).to(DEVICE)

    for gr, gs in zip(grad1, grad2):
        shape = gr.shape

        if len(shape) == 4:
            gr = gr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gs = gs.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:
            gr = gr.reshape(shape[0], shape[1] * shape[2])
            gs = gs.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2:
            tmp = 'do notthing'
        elif len(shape) == 1:
            gr = gr.reshape(1, shape[0])
            gs = gs.reshape(1, shape[0])
            continue
        dis_weight = torch.sum(1 - torch.sum(gr * gs, dim=-1) / torch.norm(gr, dim=-1) + 0.000001)
        dist += dis_weight
    return dist

def train_synthetic(model, dataloader, ipc, epochs, network_steps=100):
    sample, train_label = [], []
    print("Training synthetic data...")

    for (_, batch_data, batch_labels) in dataloader:
      #batch_data, batch_labels = batch["img"], batch["label"]
      sample.append(batch_data)
      train_label.append(batch_labels)

    sample = torch.cat(sample, dim=0)
    trainlabel = torch.cat(train_label, dim=0)

    n_labels = len(np.unique(trainlabel))
    numlabels = list(np.unique(trainlabel))

    trainset = SyntheticDataset(sample, trainlabel)

    indices_class = [[] for c in range(10)]
    images_all = [torch.unsqueeze(trainset[i][0], dim=0) for i in range(len(trainset))]
    labels_all = [trainset[i][1] for i in range(len(trainset))]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    #x
    indices_class = [cls for cls in indices_class if len(cls) > 0]
    numlabels = [i for i, cls in enumerate(indices_class) if len(cls) > 0]

    images_all = torch.cat(images_all, dim=0).to(DEVICE)
    labels_all = torch.tensor(labels_all, dtype=torch.long).to(DEVICE)

    def get_images(c, n):
    # Kiểm tra xem lớp có đủ mẫu không
      if len(indices_class[c]) < n:
        idx_shuffle = indices_class[c]
      else:
        idx_shuffle = np.random.permutation(indices_class[c])[:n - 1]
      return images_all[idx_shuffle]

    data_syn = []
    for c in numlabels:
        img = get_images(c, 1)[0]
        data_syn.append(img.unsqueeze(0))
    data_syn = torch.cat(data_syn, dim=0).clone().detach().to(DEVICE).requires_grad_(True)
    target_syn = torch.repeat_interleave(torch.tensor(numlabels, device=DEVICE), repeats=ipc)
    img_optim = torch.optim.SGD([data_syn], lr=IMG_LR)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    net = model.to(DEVICE)
    #net.load_state_dict(model.state_dict())

    for k in tq.tqdm(range(epochs)):

        net.train()

        optimizer_net = torch.optim.SGD(params=net.parameters(), lr=LR)
        loss_avg = 0

        for t in range(ipc):
            loss = torch.tensor(0.0).to(DEVICE)
            for c in numlabels:
                img_real = get_images(c, 256).to(DEVICE)
                target_real = torch.ones(img_real.shape[0], device=DEVICE, dtype=torch.long) * c
                pred_real = net(img_real)['logits']

                loss_real = loss_fn(pred_real, target_real)
                gw_real = torch.autograd.grad(loss_real, net.parameters(), retain_graph=True)

                data_synth = data_syn[c * ipc: (c + 1) * ipc].reshape((ipc, CHANNEL, IM_SIZE[0], IM_SIZE[1]))
                target_synth = torch.ones(ipc, device=DEVICE, dtype=torch.long) * c
                pred_syn = net(data_synth)['logits']

                loss_syn = loss_fn(pred_syn, target_synth)
                gw_syn = torch.autograd.grad(loss_syn, net.parameters(), create_graph=True)

                loss += distance(gw_syn, gw_real)

            img_optim.zero_grad()
            loss.backward()
            img_optim.step()
            loss_avg += loss.item()

            if t == ipc - 1:
                break

            net.train()
            for _ in range(network_steps):
                pred_sn = net(data_syn)
                loss_sn = loss_fn(pred_sn, target_syn)
                optimizer_net.zero_grad()
                loss_sn.backward()
                optimizer_net.step()

        loss_avg /= (ipc * n_labels)
        if k % 10 == 0:
            print(f'iter = {k}, loss = {loss_avg:.2f}')
    data_syn_list = data_syn.tolist()
    target_syn_list = target_syn.tolist()

    return {'x': data_syn_list, 'y': target_syn_list}