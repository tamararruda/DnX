import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from tqdm.notebook import tqdm


from utils import *
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(adj, x, train_mask, y,model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x, adj)[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(adj, x, train_mask, y, model):
    model.eval()
    logits = model(x, adj)
    for mask in [train_mask]:
        pred = torch.argmax(logits[mask], dim=1)
        acc = pred.eq(torch.argmax(y[mask], dim=1)).sum().item() / mask.sum().item()
    return acc


def main(dataset_name, database, num_epochs,iterations, layer, input_dim, num_class, out_path):
    avg = []


    for i in range(iterations):
        train_acc_list = []
        best_acc = 0
        epochs_no_improve = 0

        X,  A, edge_index, L_model =  database
        
        model, X, edge_index, L_model = SGC(layer, input_dim, num_class).to(device), X.to(device), edge_index.to(device), L_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-6)

        criterion = F.cross_entropy

        mask_train, mask_val, mask_test = mask(A)
        for epoch in tqdm(range(1, num_epochs)):
            loss = train(edge_index,X,mask_train.numpy(),L_model,model, criterion, optimizer)
            train_acc = test(edge_index,X,mask_train.numpy(),L_model, model)
            if train_acc > best_acc:
                best_acc = train_acc
                best_model = model.state_dict()
                
            log = 'Epoch: {:03d},loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            train_acc_list.append(train_acc)
        avg.append(train_acc)
        print('Mean Accuracy:{} |  Std Accuracy:{}'.format(np.mean(avg), np.std(avg)))


        save_data = {
            "adj": A,
            "feat": X,
            "label": L_model,
        }

        save_checkpoint(best_model, optimizer,out_path, dataset_name, num_epochs=epoch, save_data=save_data)
