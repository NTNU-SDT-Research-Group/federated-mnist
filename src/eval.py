from torch.utils.data import DataLoader
import torch
import pandas as  pd

def eval(device, model, criterion, test_dataset, wandb):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    output_df = pd.DataFrame(columns=['label', 'prediction'])

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

        # Save results
        output_df_temp = pd.DataFrame({
            'label': labels.cpu().numpy(),
            'prediction': pred_labels.cpu().numpy()
        })
        output_df = pd.concat([output_df, output_df_temp], axis=0)

    output_df.to_csv('output.csv', index=False)

    accuracy = correct/total
    wandb.log({
        "test_acc": accuracy, 
        "test_loss": loss
    })
    return accuracy, loss