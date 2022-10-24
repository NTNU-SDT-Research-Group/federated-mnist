from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets.dataset_factory import DatasetFactory
from losses.loss_factory import LossFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from transformers.transformer_factory import TransformerFactory
from eval import eval


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
    train_dataset, test_dataset, _ = dataset_factory.get_dataset(
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

    epoch_loss = []

    for epoch in tqdm(range(config["epochs"])):
        global_model.train()

        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # testing
    test_acc, _ = eval(device, global_model, criterion, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
