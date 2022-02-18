
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse


from utils import *
from model import *

#torch.manual_seed(12)
#torch.random.manual_seed(12)


dataset='syn6'
dataset_path = 'dataset/'
in_path = 'trained_gcn/'
out_path = '/trained_distiller/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load model
A, X = load_XA(dataset, datadir = dataset_path)
num_nodes = X.shape[0]
L = load_labels(dataset, datadir = dataset_path)
num_classes = max(L) + 1

ckpt = load_ckpt(dataset,datadir = in_path)


A = torch.tensor(A, dtype=torch.float32)
X = torch.tensor(X, dtype=torch.float32)

X = F.one_hot(torch.sum(A,1).type(torch.LongTensor)).type(torch.float32)

input_dim = X.shape[1]

pred = ckpt["save_data"]["pred"].squeeze(0)

L_model = torch.softmax(torch.tensor(pred),1)
#L_model = torch.argmax(torch.tensor(pred),1)

edge_index,_ = dense_to_sparse(A)


def train(adj, x, train_mask, y,model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x, adj)[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(adj, x, train_mask, test_mask, val_mask, y, model):
    model.eval()
    logits, accs = model(x, adj), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = torch.argmax(logits[mask], dim=1)
        acc = pred.eq(torch.argmax(y[mask], dim=1)).sum().item() / mask.sum().item()
        # acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs




avg_test = []
avg_train = []

iterations = 1

for i in range(iterations):
  list_loss = []
  val_accuracies = []
  train_acc_list = []
  test_accs1 = []
  test_accs2 = []
  best_val_acc = test_acc = 0
  epochs_no_improve = 0

  model, X, edge_index, L_model = Net(3, input_dim, num_classes).to(device), X.to(device), edge_index.to(device), L_model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-6)

  criterion = F.cross_entropy

  mask_train, mask_val, mask_test = mask(A)
  for epoch in range(1, 10001):
      loss = train(edge_index,X,mask_train.numpy(),L_model,model, criterion, optimizer)
      train_acc, val_acc, tmp_test_acc = test(edge_index,X,mask_train.numpy(),mask_test.numpy(),mask_val.numpy(),L_model, model)
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          test_acc = tmp_test_acc
      log = 'Epoch: {:03d},loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
      list_loss.append(loss)
      train_acc_list.append(train_acc)
      if epoch%1 == 0:
        print(log.format(epoch, loss, train_acc, best_val_acc, test_acc))
      val_accuracies.append(best_val_acc)
      test_accs1.append(test_acc)
      if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2-epochs_no_improve]:
          epochs_no_improve = epochs_no_improve + 1
      else:
          epochs_no_improve = 0
          best_model = model.state_dict()
          print('*')

      if epochs_no_improve >= 500:
          print('Early stopping!')
          avg_test.append(test_acc)
          avg_train.append(train_acc)
          break
      if epoch == 10000:
          avg_test.append(test_acc)
          avg_train.append(train_acc)
print('Mean Train:{} | Mean test:{}| Std test:{}'.format(np.mean(avg_train), np.mean(avg_test), np.std(avg_test)))



save_data = {
    "adj": A,
    "feat": X,
    "label": L_model,
}

save_checkpoint(best_model, optimizer,out_path, dataset, num_epochs=epoch, save_data=save_data)