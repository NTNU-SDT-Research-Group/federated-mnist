#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets.dataset_factory import DatasetFactory
from losses.loss_factory import LossFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from transformers.transformer_factory import TransformerFactory
from eval import eval
import torch


def train(config, device):
    # transformer factory
    transformer_factory = TransformerFactory()

    # dataset factory
    dataset_factory = DatasetFactory()

    # model factory
    model_factory = ModelFactory()

    # optimiser facotry
    optimiser_factory = OptimiserFactory()

    # loss factory
    loss_factory = LossFactory()

    # load data
    train_dataset, test_dataset, user_groups = dataset_factory.get_dataset(
        config["dataset"],
        transformer=transformer_factory.get_transformer(
            pipe_type="default"
        ),
        num_users=config["num_users"]
    )

    # load model
    global_model = model_factory.get_model(
        config["model"],
        num_channels=config["num_channels"],
        num_classes=config["num_classes"]
    ).to(device)

    # load optimiser
    optimiser = optimiser_factory.get_optimiser(
        global_model.parameters(),
        config["optimiser"],
        hyper_params={
            "learning_rate": config["lr"],
            "momentum": config["momentum"]
        }
    )

    # load loss
    criterion = loss_factory.get_loss_function(
        "negative-log-likelihood-loss").to(device)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    start_time = time.time()

    for epoch in tqdm(range(config["epochs"])):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(
            range(config["num_users"]), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for _ in range(config["num_users"]):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, config["epochs"], config["frac"], args.iid,
                args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
      self.args = args
      self.logger = logger
      self.trainloader, self.validloader, self.testloader = self.train_val_test(
          dataset, list(idxs))
      self.device = 'cuda' if args.gpu else 'cpu'
      # Default criterion set to NLL loss function
      self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
      """
      Returns train, validation and test dataloaders for a given dataset
      and user indexes.
      """
      # split indexes for train, validation, and test (80, 10, 10)
      idxs_train = idxs[:int(0.8*len(idxs))]
      idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
      idxs_test = idxs[int(0.9*len(idxs)):]

      trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True)
      validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=int(len(idxs_val)/10), shuffle=False)
      testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                              batch_size=int(len(idxs_test)/10), shuffle=False)
      return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
      # Set mode to train model
      model.train()
      epoch_loss = []

      # Set optimizer for the local updates
      if self.args.optimizer == 'sgd':
          optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                      momentum=0.5)
      elif self.args.optimizer == 'adam':
          optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)

      for iter in range(self.args.local_ep):
          batch_loss = []
          for batch_idx, (images, labels) in enumerate(self.trainloader):
              images, labels = images.to(self.device), labels.to(self.device)

              model.zero_grad()
              log_probs = model(images)
              loss = self.criterion(log_probs, labels)
              loss.backward()
              optimizer.step()

              if self.args.verbose and (batch_idx % 10 == 0):
                  print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      global_round, iter, batch_idx * len(images),
                      len(self.trainloader.dataset),
                      100. * batch_idx / len(self.trainloader), loss.item()))
              self.logger.add_scalar('loss', loss.item())
              batch_loss.append(loss.item())
          epoch_loss.append(sum(batch_loss)/len(batch_loss))

      return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
      """ Returns the inference accuracy and loss.
      """

      model.eval()
      loss, total, correct = 0.0, 0.0, 0.0

      for batch_idx, (images, labels) in enumerate(self.testloader):
          images, labels = images.to(self.device), labels.to(self.device)

          # Inference
          outputs = model(images)
          batch_loss = self.criterion(outputs, labels)
          loss += batch_loss.item()

          # Prediction
          _, pred_labels = torch.max(outputs, 1)
          pred_labels = pred_labels.view(-1)
          correct += torch.sum(torch.eq(pred_labels, labels)).item()
          total += len(labels)

      accuracy = correct/total
      return accuracy, loss

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg