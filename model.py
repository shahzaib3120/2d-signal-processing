import torch
import torch.nn as nn
import numpy as np
from dataloader import BitmapDataset
from torch.utils.data import DataLoader, random_split
import pickle
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# add tensorboard support
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# input shape: (batch_size, 140, 1000)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(64 * 17 * 125, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    
def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    train_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx,batch in enumerate(dataloader):
        progress_bar.set_description(f'Batch {idx + 1}/{len(dataloader)}')
        bitmaps = batch['bitmap']
        labels = batch['label']
        bitmaps = bitmaps.unsqueeze(1).float()
        output = model(bitmaps)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_accuracy = accuracy_score(labels, output.argmax(dim=1))
        batch_precision = precision_score(labels, output.argmax(dim=1))
        batch_recall = recall_score(labels, output.argmax(dim=1))
        batch_f1 = f1_score(labels, output.argmax(dim=1))
        
        accuracy += batch_accuracy
        precision += batch_precision
        recall += batch_recall
        f1 += batch_f1
        train_loss += loss.item()
        progress_bar.set_description(f'Loss: {loss.item()}')
        progress_bar.update()
    progress_bar.close()
    
    return {
        'accuracy': accuracy / len(dataloader),
        'precision': precision / len(dataloader),
        'recall': recall / len(dataloader),
        'f1': f1 / len(dataloader),
        'loss': train_loss / len(dataloader)
    }
    

def validate_epoch(model, dataloader, loss_fn, return_preds=False):
    model.eval()
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    eval_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx,batch in enumerate(dataloader):
        progress_bar.set_description(f'Batch {idx + 1}/{len(dataloader)}')
        bitmaps = batch['bitmap']
        labels = batch['label']
        bitmaps = bitmaps.unsqueeze(1).float()
        output = model(bitmaps)
        preds = output.argmax(dim=1)
        
        if return_preds:
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
        
        loss = loss_fn(output, labels)
        batch_accuracy = accuracy_score(labels, preds)
        batch_precision = precision_score(labels, preds)
        batch_recall = recall_score(labels, preds)
        batch_f1 = f1_score(labels, preds)
        
        accuracy += batch_accuracy
        precision += batch_precision
        recall += batch_recall
        f1 += batch_f1
        eval_loss += loss.item()
        
        progress_bar.set_description(f'Loss: {loss.item()}')
        progress_bar.update()
    progress_bar.close()
    
    if return_preds:
        return {
            'accuracy': accuracy / len(dataloader),
            'precision': precision / len(dataloader),
            'recall': recall / len(dataloader),
            'f1': f1 / len(dataloader),
            'loss': eval_loss / len(dataloader),
            'preds': all_preds,
            'labels': all_labels
        }
    return {
        'accuracy': accuracy / len(dataloader),
        'precision': precision / len(dataloader),
        'recall': recall / len(dataloader),
        'f1': f1 / len(dataloader),
        'loss': eval_loss / len(dataloader)
    }
    
    
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, n_epochs, log_dir='logs'):
    writer = SummaryWriter(log_dir)
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        train_metrics = train_epoch(model, train_dataloader, loss_fn, optimizer)
        print(f'Train: Accuracy: {train_metrics["accuracy"]}, Precision: {train_metrics["precision"]}, Recall: {train_metrics["recall"]}, F1: {train_metrics["f1"]}, Loss: {train_metrics["loss"]}')
        if epoch == n_epochs - 1:
            val_metrics = validate_epoch(model, val_dataloader, loss_fn, return_preds=True)
            print(f'Validation: Accuracy: {val_metrics["accuracy"]}, Precision: {val_metrics["precision"]}, Recall: {val_metrics["recall"]}, F1: {val_metrics["f1"]}, Loss: {val_metrics["loss"]}')
            writer.add_pr_curve('Precision-Recall Curve', val_metrics['labels'], val_metrics['preds'])
        else:
            val_metrics = validate_epoch(model, val_dataloader, loss_fn, return_preds=False)
        print(f'Validation: Accuracy: {val_metrics["accuracy"]}, Precision: {val_metrics["precision"]}, Recall: {val_metrics["recall"]}, F1: {val_metrics["f1"]}, Loss: {val_metrics["loss"]}')
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Precision/train', train_metrics['precision'], epoch)
        writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
    writer.close()
    
    return val_metrics
    
if __name__ == '__main__':
    with open('data/dataset.pkl', 'rb') as f:
        pk_dataset = pickle.load(f)
    dataset = BitmapDataset(pk_dataset)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    del pk_dataset
    del dataset
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    val_metrics = train(model, train_dataloader, val_dataloader, loss_fn, optimizer, 2)
    
    # save model
    torch.save(model.state_dict(), 'model.pth')
    
    # classification report
    preds = val_metrics['preds']
    labels = val_metrics['labels']
    print(classification_report(labels, preds))
    
    
    # plot confusion matrix
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # load model
    # model = CNN()
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    
    
    