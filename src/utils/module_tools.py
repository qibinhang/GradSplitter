import numpy as np
import torch
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_module(module_conv_info, module_head_para, trained_model):
    if trained_model.__class__.__name__ == 'SimCNN':
        from models.simcnn import SimCNN
        module = _extract_module_sim_res(module_conv_info, module_head_para, trained_model, SimCNN)
    elif trained_model.__class__.__name__ == 'ResCNN':
        from models.rescnn import ResCNN
        module = _extract_module_sim_res(module_conv_info, module_head_para, trained_model, ResCNN)
    elif trained_model.__class__.__name__ == 'InceCNN':
        from models.incecnn import InceCNN
        module = _extract_module_ince(module_conv_info, module_head_para, trained_model, InceCNN)
    else:
        raise ValueError
    return module


def _extract_module_sim_res(module_conv_info, module_head_para, trained_model, model_network):
    """
    for SimCNN and ResCNN.
    conv_info: tensor [[1 0 0 1] ...[0 0 1 1]]
    """
    # get the configures of module from the update_conv_info
    conv_configs = []
    cin = 3
    for each_conv_layer in module_conv_info:
        n_kernels = each_conv_layer.size
        conv_configs.append((cin, n_kernels))
        cin = n_kernels

    module = model_network(num_classes=trained_model.num_classes, conv_configs=conv_configs)

    # extract the parameters of active kernels from model
    active_kernel_param = {}
    model_param = trained_model.state_dict()
    for i in range(len(conv_configs)):
        conv_weight = model_param[f'conv_{i}.0.weight']
        conv_bias = model_param[f'conv_{i}.0.bias']
        bn_weight = model_param[f'conv_{i}.1.weight']
        bn_bias = model_param[f'conv_{i}.1.bias']
        bn_running_mean = model_param[f'conv_{i}.1.running_mean']
        bn_running_var = model_param[f'conv_{i}.1.running_var']

        cur_conv_active_kernel_idx = module_conv_info[i]  # active Cout
        pre_conv_active_kernel_idx = module_conv_info[i-1] if i > 0 else list(range(3))  # active Cin

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

        active_kernel_param[f'conv_{i}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]

    assert model_param[f'fc_{len(conv_configs)}.weight'].size(1) == model_param[f'conv_{len(conv_configs)-1}.0.bias'].size(0)
    first_fc_weight = model_param[f'fc_{len(conv_configs)}.weight']
    pre_conv_active_kernel_idx = module_conv_info[-1]
    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'fc_{len(conv_configs)}.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    model_param.update(module_head_para)
    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module


def _extract_module_ince(module_conv_info, module_head_para, trained_model, model_network):
    pre_layers = [0, 1, 2]
    ince_layer_1 = [3, 4, 5]
    ince_layer_2 = [6, 7, 8]
    ince_layer_3 = [9, 10, 11]

    conv_configs = []
    for layer_idx in range(len(module_conv_info)):
        if layer_idx in pre_layers:
            cin = conv_configs[layer_idx - 1][1] if layer_idx > 0 else 3
            cout = module_conv_info[layer_idx].size
        elif layer_idx in ince_layer_1:
            cin = conv_configs[2][1]
            cout = module_conv_info[layer_idx].size
        elif layer_idx in ince_layer_2:
            cin = sum([conv_configs[i][1] for i in ince_layer_1])
            cout = module_conv_info[layer_idx].size
        elif layer_idx in ince_layer_3:
            cin = sum([conv_configs[i][1] for i in ince_layer_2])
            cout = module_conv_info[layer_idx].size
        else:
            raise ValueError
        conv_configs.append((cin, cout))

    module = model_network(num_classes=trained_model.num_classes, conv_configs=conv_configs)

    # extract the parameters of active kernels from model
    active_kernel_param = {}
    model_param = trained_model.state_dict()
    for i in range(len(conv_configs)):
        conv_weight = model_param[f'conv_{i}.0.weight']
        conv_bias = model_param[f'conv_{i}.0.bias']
        bn_weight = model_param[f'conv_{i}.1.weight']
        bn_bias = model_param[f'conv_{i}.1.bias']
        bn_running_mean = model_param[f'conv_{i}.1.running_mean']
        bn_running_var = model_param[f'conv_{i}.1.running_var']

        cur_conv_active_kernel_idx = module_conv_info[i]  # active Cout

        # active Cin
        if i in pre_layers:
            pre_conv_active_kernel_idx = module_conv_info[i - 1] if i > 0 else list(range(3))
        elif i in ince_layer_1:
            pre_conv_active_kernel_idx = module_conv_info[2]
        elif i in ince_layer_2:
            tmp_last_conv_len = [model_param[f'conv_{tmp_idx}.0.bias'].size(0) for tmp_idx in ince_layer_1]
            pre_3_conv_active_kernel_idx = [module_conv_info[tmp_idx] for tmp_idx in ince_layer_1]
            pre_conv_active_kernel_idx = [
                pre_3_conv_active_kernel_idx[0],
                pre_3_conv_active_kernel_idx[1] + tmp_last_conv_len[0],
                pre_3_conv_active_kernel_idx[2] + tmp_last_conv_len[0] + tmp_last_conv_len[1]
            ]
            pre_conv_active_kernel_idx = np.concatenate(pre_conv_active_kernel_idx, axis=0)
        elif i in ince_layer_3:
            tmp_last_conv_len = [model_param[f'conv_{tmp_idx}.0.bias'].size(0) for tmp_idx in ince_layer_2]
            pre_3_conv_active_kernel_idx = [module_conv_info[tmp_idx] for tmp_idx in ince_layer_2]
            pre_conv_active_kernel_idx = [
                pre_3_conv_active_kernel_idx[0],
                pre_3_conv_active_kernel_idx[1] + tmp_last_conv_len[0],
                pre_3_conv_active_kernel_idx[2] + tmp_last_conv_len[0] + tmp_last_conv_len[1]
            ]
            pre_conv_active_kernel_idx = np.concatenate(pre_conv_active_kernel_idx, axis=0)
        else:
            raise ValueError

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

        active_kernel_param[f'conv_{i}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]

    # check
    tmp_fc_len = model_param[f'fc_{len(conv_configs)}.weight'].size(1)
    tmp_last_conv_len = [model_param[f'conv_{len(conv_configs) - 3}.0.bias'].size(0),
                         model_param[f'conv_{len(conv_configs) - 2}.0.bias'].size(0),
                         model_param[f'conv_{len(conv_configs) - 1}.0.bias'].size(0)]
    assert tmp_fc_len == sum(tmp_last_conv_len)

    first_fc_weight = model_param[f'fc_{len(conv_configs)}.weight']
    conv_a_active_kernel_idx = module_conv_info[-3]
    conv_b_active_kernel_idx = np.array(module_conv_info[-2]) + tmp_last_conv_len[0]
    conv_c_active_kernel_idx = np.array(module_conv_info[-1]) + tmp_last_conv_len[0] + tmp_last_conv_len[1]
    pre_conv_active_kernel_idx = np.concatenate(
        [conv_a_active_kernel_idx, conv_b_active_kernel_idx, conv_c_active_kernel_idx],
        axis=0
    )

    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'fc_{len(conv_configs)}.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    model_param.update(module_head_para)
    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module


def get_target_module_info(modules_info, trained_model, target_class, handle_warning=True):
    if trained_model.__class__.__name__ == 'SimCNN':
        module_conv_info, module_head_para = get_target_module_info_for_simcnn(modules_info, target_class,
                                                                               handle_warning)
    elif trained_model.__class__.__name__ == 'ResCNN':
        module_conv_info, module_head_para = get_target_module_info_for_rescnn(modules_info, target_class,
                                                                               trained_model, handle_warning)
    elif trained_model.__class__.__name__ == 'InceCNN':
        module_conv_info, module_head_para = get_target_module_info_for_incecnn(modules_info, target_class,
                                                                                trained_model, handle_warning)
    else:
        raise ValueError
    return module_conv_info, module_head_para


def get_target_module_info_for_simcnn(modules_info, target_class, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    for conv_idx in range(len(modules_info)):
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break

    if f'module_{target_class}_head.weight' in modules_info:  # head with one layer
        module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
        module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    elif f'module_{target_class}_head.0.weight' in modules_info:  # head with multi-layer
        module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
        module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
        module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
        module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    else:
        raise KeyError
    return module_conv_info, module_head_para


def get_target_module_info_for_rescnn(modules_info, target_class, trained_model, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    residual_layer_indices = trained_model.residual_idx
    for conv_idx in range(len(modules_info)):
        if conv_idx in residual_layer_indices:
            layer_name = f'module_{target_class}_conv_{conv_idx - 2}'
        else:
            layer_name = f'module_{target_class}_conv_{conv_idx}'

        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break
    # module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
    # module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
    module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
    module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
    module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    return module_conv_info, module_head_para


def get_target_module_info_for_incecnn(modules_info, target_class, trained_model, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    for conv_idx in range(len(modules_info)):
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break
    # module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
    # module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
    module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
    module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
    module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    return module_conv_info, module_head_para


def load_module(module_path, trained_model, target_class):
    modules_info = torch.load(module_path, map_location='cpu')
    module_conv_info, module_head_para = get_target_module_info(modules_info, trained_model, target_class)
    module = extract_module(module_conv_info, module_head_para, trained_model)
    return module


@torch.no_grad()
def module_predict(module, dataset):
    outputs, labels = [], []
    for batch_inputs, batch_labels in dataset:
        batch_inputs = batch_inputs.to(device)
        batch_output = module(batch_inputs)
        outputs.append(batch_output)
        labels.append(batch_labels.to(device))
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    return outputs, labels

