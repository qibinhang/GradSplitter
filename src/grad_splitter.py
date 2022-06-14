# "head" means that a FC layer (with dimension (10, 1) for 10-class classification) is added
# after the last layer of a module.
# "head" is trained jointly with mask.
import argparse
import torch
import torch.nn.functional as F
import time
from utils.model_loader import load_model
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.checker import check_dir
from utils.splitter_loader import load_splitter
from tqdm import tqdm


def loss_func(predicts, label, grad_splitter):
    module_kernels = grad_splitter.get_module_kernels()
    loss_pred = F.cross_entropy(predicts, label)

    loss_count = torch.mean(module_kernels)  # each module has fewer kernels
    return loss_pred, loss_count


def modularize(model, train_dataset, val_dataset, test_dataset, save_dir):
    grad_splitter = GradSplitter(model, init_type)

    acc_log = []
    best_acc, best_epoch, best_avg_kernel = 0.0, 0, 0
    best_modules = None
    head_param = [param for name, param in grad_splitter.named_parameters()
                  if param.requires_grad and 'head' in name]
    all_param = [param for param in grad_splitter.parameters() if param.requires_grad]
    phase = 0
    optimize = torch.optim.Adam(head_param, lr=lr_head)
    ratio_loss_sim = 0

    for epoch in range(epochs_for_head + epochs_for_modularity):
        if phase != iterative_strategy[epoch]:
            phase = iterative_strategy[epoch]
            if phase == 0:
                print('\n*** Train head ***\n')
                optimize = torch.optim.Adam(head_param, lr=lr_head * 0.1)  # smaller lr_ratio
                ratio_loss_sim = 0
            else:
                print('\n*** Modularize ***\n')
                optimize = torch.optim.Adam(all_param, lr=lr_modularity)
                ratio_loss_sim = alpha

        print(f'epoch {epoch}')
        print('-' * 80)
        grad_splitter, optimize = train_splitter(grad_splitter, optimize, train_dataset, ratio_loss_sim)
        val_acc, avg_kernel = eval_splitter(grad_splitter, val_dataset)
        acc_log.append(val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_avg_kernel = avg_kernel
            best_epoch = epoch
        best_modules = grad_splitter.get_module_params()
        torch.save(best_modules, f'{save_dir}/epoch_{epoch}.pth')

    print('='*100 + '\n')
    print(f'best_epoch: {best_epoch}')
    print(f'best_acc: {best_acc * 100:.2f}%')
    print(f'best_avg_kernel: {best_avg_kernel:.2f}')

    # TEST
    sorted_acc_log = list(sorted(zip(range(len(acc_log)), acc_log), key=lambda x: x[1], reverse=True))
    print(sorted_acc_log)
    #
    return best_modules


def train_splitter(grad_splitter, optimize, train_dataset, ratio_loss_sim):
    epoch_train_loss_pred = []
    epoch_train_loss_sim = []
    epoch_train_acc = []
    grad_splitter.train()
    grad_splitter.model.eval()  # for BN in model
    for batch_inputs, batch_labels in tqdm(train_dataset, ncols=100, desc='train'):
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = grad_splitter(batch_inputs)
        optimize.zero_grad()
        loss_pred, loss_sim = loss_func(outputs, batch_labels, grad_splitter)
        loss = loss_pred + ratio_loss_sim * loss_sim
        loss.backward()
        optimize.step()

        pred = torch.argmax(outputs, dim=1)
        acc = torch.sum(pred == batch_labels)
        epoch_train_acc.append(torch.div(acc, batch_labels.shape[0]))
        epoch_train_loss_pred.append(loss_pred.detach())
        epoch_train_loss_sim.append(loss_sim.detach())

    print(f'## Train ##')
    print(f"loss_pred: {sum(epoch_train_loss_pred) / len(epoch_train_loss_pred):.2f}")
    print(f"loss_sim: {sum(epoch_train_loss_sim) / len(epoch_train_loss_sim):.2f}")
    print(f"acc : {sum(epoch_train_acc) / len(epoch_train_acc) * 100:.2f}%\n")
    return grad_splitter, optimize


@torch.no_grad()
def eval_splitter(grad_splitter, test_dataset, attr='val'):
    epoch_val_acc = []
    epoch_val_loss_pred = []
    epoch_val_loss_sim = []
    grad_splitter.eval()
    # acc
    for batch_inputs, batch_labels in tqdm(test_dataset, ncols=100, desc='val'):
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = grad_splitter(batch_inputs)
        loss_pred, loss_sim = loss_func(outputs, batch_labels, grad_splitter)
        pred = torch.argmax(outputs, dim=1)
        acc = torch.sum(pred == batch_labels)

        epoch_val_acc.append(torch.div(acc, batch_labels.shape[0]))
        epoch_val_loss_pred.append(loss_pred.detach())
        epoch_val_loss_sim.append(loss_sim.detach())
    val_acc = sum(epoch_val_acc) / len(epoch_val_acc)

    # n_kernel
    n_kernel = []
    module_kernels = grad_splitter.get_module_kernels()
    module_kernels = module_kernels > 0.5
    for module_idx in range(len(module_kernels)):
        n_kernel.append(torch.sum(module_kernels[module_idx]).float())
    avg_kernel = torch.mean(torch.stack(n_kernel))

    if attr == 'val':
        print('## Validation ##')
    else:
        print('## Test ##')
    print(f"loss_pred: {sum(epoch_val_loss_pred) / len(epoch_val_loss_pred):.2f}")
    print(f"loss_sim: {sum(epoch_val_loss_sim) / len(epoch_val_loss_sim):.2f}")
    print(f"acc : {val_acc * 100:.2f}%\n")
    print(f'avg_kernels: {avg_kernel:.2f}\n')
    return val_acc, avg_kernel


def main():
    model = load_model(model_name, configs.num_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    load_dataset = get_dataset_loader(dataset_name, for_modular=True)
    _, test_dataset = load_dataset(dataset_dir, is_train=False, batch_size=batch_size,
                                   shuffle_seed=None, is_random=None, num_workers=1, pin_memory=True)

    modularity_train_loader, modularity_val_loader = load_dataset(dataset_dir, is_train=True, split_train_set='8:2',
                                                                  shuffle_seed=estimator_idx, is_random=False,
                                                                  batch_size=batch_size, num_workers=1, pin_memory=True)
    modularize(model, modularity_train_loader, modularity_val_loader, test_dataset, module_save_dir)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    parser.add_argument('--estimator_idx', type=int)
    parser.add_argument('--init_type', type=str, choices=['random', 'ones', 'zeros'], default='ones')
    parser.add_argument('--lr_head', type=float, default=0.01)
    parser.add_argument('--lr_modularity', type=float, default=0.001)
    parser.add_argument('--epoch_strategy', type=str, default='5,140', help='epochs_for_head,epochs_for_modularity')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    torch.random.manual_seed(1)
    model_name = args.model
    dataset_name = args.dataset
    estimator_idx = args.estimator_idx
    init_type = args.init_type
    lr_head = args.lr_head
    lr_modularity = args.lr_modularity
    batch_size = args.batch_size
    epochs = args.epoch_strategy.split(',')
    epochs_for_head, epochs_for_modularity = int(epochs[0]), int(epochs[1])
    alpha = args.alpha
    GradSplitter = load_splitter(model_name, None)
    iterative_for_head = [0] * epochs_for_head
    iterative_strategy = [1, 1, 1, 1, 1, 0, 0]  # 0 means training head, and 1 means modularization.
    iterative_strategy = iterative_strategy * int(epochs_for_modularity / 7)
    iterative_strategy = iterative_for_head + iterative_strategy

    configs = load_configure(model_name, dataset_name)
    configs.set_estimator_idx(estimator_idx)
    dataset_dir = configs.dataset_dir
    trained_model_path = configs.trained_model_path
    module_save_dir = f'{configs.module_save_dir}/lr_{lr_head}_{lr_modularity}_alpha_{alpha}'
    check_dir(module_save_dir)
    main()
