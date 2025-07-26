import copy
import logging
from torch.utils.data import DataLoader, IterableDataset, Dataset
import torch.nn as nn 
import torch
from torch import autograd
import numpy as np
import higher
from contflame.data.utils import Buffer

def distill(model, buffer, config, criterion, train_loader, valid_loader=None, id=1):
    print(f"Distilling model {id} with {len(buffer)} samples")
    model = copy.deepcopy(model)
    run_config = config['run_config']
    param_config = config['param_config']
    # log_config = config['log_config']

    model.train()
    eval_trainloader = copy.deepcopy(train_loader)

    buff_imgs, buff_trgs = next(iter(DataLoader(buffer, batch_size=len(buffer))))
    buff_imgs, buff_trgs = buff_imgs.to(run_config['device']), buff_trgs.to(run_config['device'])

    buff_imgs.requires_grad = True

    init_valid = DataLoader(ModelInitDataset(model, 10), batch_size=1, collate_fn=lambda x: x)
    init_loader = DataLoader(ModelInitDataset(model, -1), batch_size=1, collate_fn=lambda x: x)
    init_iter = iter(init_loader)

    buff_opt = torch.optim.SGD([buff_imgs], lr=param_config['meta_lr'],)

    lr_list = []
    lr_opts = []
    for _ in range(param_config['inner_steps']):
        lr = np.log(np.exp([param_config['model_lr']]) - 1)  # Inverse of softplus (so that the starting learning rate is actually the specified one)
        lr = torch.tensor(lr, requires_grad=True, device=run_config['device'])
        lr_list.append(lr)
        lr_opts.append(torch.optim.SGD([lr], param_config['lr_lr'],))

    for i in range(param_config['outer_steps']):
        for step, (ds_imgs, ds_trgs) in enumerate(train_loader):
            try: init_batch = next(init_iter)
            except StopIteration: init_iter = iter(init_loader); init_batch = next(init_iter)

            ds_imgs = ds_imgs.to(run_config['device'])
            ds_trgs = ds_trgs.to(run_config['device'])

            acc_loss = None
            epoch_loss = [None for _ in range(param_config['inner_steps'])]

            for r, sigma in enumerate(init_batch):
                model.load_state_dict(sigma)
                model_opt = torch.optim.SGD(model.parameters(), lr=1, )
                with higher.innerloop_ctx(model, model_opt) as (fmodel, diffopt):
                    for j in range(param_config['inner_steps']):
                        # Update the model
                        buff_out = fmodel(buff_imgs)
                        buff_loss = criterion(buff_out, buff_trgs)
                        buff_loss = buff_loss * torch.log(1 + torch.exp(lr_list[j]))
                        diffopt.step(buff_loss)


                        ds_out = fmodel(ds_imgs)
                        ds_loss = criterion(ds_out, ds_trgs)

                        epoch_loss[j] = epoch_loss[j] + ds_loss if epoch_loss[j] is not None else ds_loss
                        acc_loss = acc_loss + ds_loss if acc_loss is not None else ds_loss

                        # Metrics (20 samples of loss and accuracy at the last inner step)
                        if (((step + i * len(train_loader)) % int(round(len(train_loader) * param_config['outer_steps'] * 0.05)) == \
                                int(round(len(train_loader) * param_config['outer_steps'] * 0.05)) - 1) or (step + i * len(train_loader)) == 0) \
                                and j == param_config['inner_steps'] - 1 and r == 0:

                            lrs = [np.log(np.exp(lr.item()) + 1) for lr in lr_list]
                            lrs_log = {f'Learning rate {i} - {id}': lr for (i, lr) in enumerate(lrs)}
                            train_loss, train_accuracy = test_distill(init_valid, lrs, [buff_imgs, buff_trgs], model, criterion, eval_trainloader, run_config)
                            if valid_loader is not None:
                                test_loss, test_accuracy = test_distill(init_valid, lrs, [buff_imgs, buff_trgs], model, criterion, valid_loader, run_config)
                            metrics = {f'Distill train loss {id}': train_loss, f'Distill train accuracy {id}': train_accuracy,
                                       #f'Distill test loss {id}': test_loss, f'Distill test accuracy {id}': test_accuracy,
                                       f'Distill step {id}': step + i * len(train_loader)}

                            # if log_config['wandb']:
                            #     wandb.log({**metrics, **lrs_log})

                            if log_config['print']:
                                print(metrics)

            # Update the lrs
            for j in range(param_config['inner_steps']):
                lr_opts[j].zero_grad()
                grad, = autograd.grad(epoch_loss[j], lr_list[j], retain_graph=True)
                lr_list[j].grad = grad
                lr_opts[j].step()

            buff_opt.zero_grad()
            acc_loss.backward()
            buff_opt.step()

    aux = []
    buff_imgs, buff_trgs = buff_imgs.detach().cpu(), buff_trgs.detach().cpu()
    for i in range(buff_imgs.size(0)):
        aux.append([buff_imgs[i], buff_trgs[i]])
    lr_list = [np.log(1 + np.exp(lr.item())) for lr in lr_list]
    print('Done distill')
    return Buffer(aux, len(aux), ), lr_list


def test_distill(init_valid, lrs, buffer, model, criterion, eval_trainloader, run_config):
    buff_imgs, buff_trgs = buffer

    avg_loss = avg_accuracy = 0

    for init_batch in init_valid:
        for init in init_batch:
            model.load_state_dict(init)

            for lr in lrs:
                opt = torch.optim.SGD(model.parameters(), lr=lr, )
                pred = model(buff_imgs)
                loss = criterion(pred, buff_trgs)

                opt.zero_grad()
                loss.backward()
                opt.step()

            test_loss, test_accuracy = test(model, criterion, eval_trainloader, run_config)
            avg_loss += (test_loss / len(init_valid))
            avg_accuracy += (test_accuracy / len(init_valid))

    return avg_loss, avg_accuracy


class ModelInitDataset(IterableDataset):

    def __init__(self, target, len):
        self.target = copy.deepcopy(target)
        self.len = len
        self.inits = []
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.len and self.len >= 0:
            raise StopIteration

        if len(self.inits) - 1 >= self.i:
            res = self.inits[self.i]
        else:
            res = copy.deepcopy(self.target.apply(initialize_weights).state_dict())
            if self.len >= 0:
                self.inits.append(res)

        self.i += 1
        return res

    def __len__(self):
        return self.len
    
def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()

def test(model, criterion, test_loader, config):
    model.eval()

    correct = 0
    loss_sum = 0
    tot = 0

    for step, (data, targets) in enumerate(test_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        correct += preds.eq(targets).sum().item()
        tot += data.size(0)

    accuracy = correct / tot
    loss = loss_sum / tot

    return loss, accuracy