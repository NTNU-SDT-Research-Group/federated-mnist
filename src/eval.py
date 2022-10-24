from torch.utils.data import DataLoader
import torch

def eval(device, model, criterion, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss