import sys
sys.path.append('..')


def load_splitter(model_name, dataset_name):
    if dataset_name is None:
        GradSplitter = load_splitter_normal(model_name)
    else:
        raise ValueError()
    return GradSplitter


def load_splitter_normal(model_name):
    if model_name == 'simcnn':
        from splitters.simcnn_splitter import GradSplitter
    elif model_name == 'rescnn':
        from splitters.rescnn_splitter import GradSplitter
    elif model_name == 'incecnn':
        from splitters.incecnn_splitter import GradSplitter
    else:
        raise ValueError()
    return GradSplitter

