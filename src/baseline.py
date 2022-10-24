from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets.dataset_factory import DatasetFactory
from losses.loss_factory import LossFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from transformers.transformer_factory import TransformerFactory

from utils import get_dataset
from options import args_parser
from update import test_inference

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


  # load datasets
  train_dataset, test_dataset, _ = get_dataset(args)

  # BUILD MODEL
  if args.model == 'cnn':
      # Convolutional neural netork
      if args.dataset == 'mnist':
          global_model = CNNMnist(args=args)

  global_model.to(device)
  global_model.train()
  print(global_model)

  # Training
  # Set optimizer and criterion
  if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                  momentum=0.5)

  trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  criterion = torch.nn.NLLLoss().to(device)
  epoch_loss = []

  for epoch in tqdm(range(args.epochs)):
      batch_loss = []

      for batch_idx, (images, labels) in enumerate(trainloader):
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()
          outputs = global_model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          if batch_idx % 50 == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch+1, batch_idx * len(images), len(trainloader.dataset),
                  100. * batch_idx / len(trainloader), loss.item()))
          batch_loss.append(loss.item())

      loss_avg = sum(batch_loss)/len(batch_loss)
      print('\nTrain loss:', loss_avg)
      epoch_loss.append(loss_avg)

  # testing
  test_acc, test_loss = test_inference(args, global_model, test_dataset)
  print('Test on', len(test_dataset), 'samples')
  print("Test Accuracy: {:.2f}%".format(100*test_acc))
