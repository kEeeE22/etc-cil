import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

from utils.ssd_utils.buffer import DynamicBuffer
from utils.ssd_utils.supcon_loss import SupConLoss

batch_size = 1
num_workers = 4

init_lr_decay = 0.1
init_epoch = 200
init_lr = 0.1
init_weight_decay = 1e-4
init_milestones = [60, 120, 160]
init_lr_decay = 0.1

epochs = 200
lrate_decay = 0.1
weight_decay = 1e-4
milestones = [60, 120, 160]
lrate = 0.1

mem_iters = 1
queue_size = 1000

class SSD(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.buffer = DynamicBuffer(self._network, args)
        self.aff_x = []
        self.aff_y = []
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(20, 256), scale=(0.2, 1.0)),
            RandomHorizontalFlip(),
            RandomGrayscale(p=0.2)

        )
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self.buffer is not None:
                    for j in range(mem_iters):
                        mem_x, mem_y = self.buffer.retrieve(x=inputs, y=targets)
                        mem_x, mem_y = mem_x.to(self._device), mem_y.to(self._device)
                        combines_batch = torch.cat((inputs, mem_x), dim=0)
                        combines_targets = torch.cat((targets, mem_y), dim=0)
                        combines_batch_aug = self.transform(combines_batch)

                        features_ori = self.model.forward(combines_batch)["features"].unsqueeze(1)
                        features_aug = self.model.forward(combines_batch_aug)["features"].unsqueeze(1)
                        features = torch.cat([features_ori, features_aug], dim=1)

                        loss_clf = SupConLoss(features, combines_targets)

                        loss = loss_clf

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses += loss.item()

                        logits = self._network(combines_batch)["logits"]
                        _, preds = torch.max(logits, dim=1)
                        correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                        total += len(targets)
                self.aff_x.append(inputs)
                self.aff_y.append(targets)
                if len(self.aff_x) > queue_size:
                    self.aff_x.pop(0)
                    self.aff_y.pop(0)
                self.buffer.update(inputs, targets, aff_x=self.aff_x, aff_y=self.aff_y, update_index=i)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)