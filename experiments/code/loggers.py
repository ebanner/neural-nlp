# functions that can be consumed by the `TensorCallback` callback

import keras.backend as K

from support import trainable_weights, norm


def updates(model):
    """Compute update magnitudes for every learnable tensor in the model"""

    names, tensors = trainable_weights(model)

    # ignore updates for optimizer-specific tensors
    optimizer = model.optimizer
    update_tups = [(p, p_new) for p, p_new in optimizer.updates if p.name in names]
    update_mags = [norm(new - old) for new, old in update_tups]

    return [name+'_up' for name in names], update_mags

def weights(model):
    """Compute weight magnitude for each parameter in the model"""

    names, tensors = trainable_weights(model)

    return [name+'_mag' for name in names], [norm(tensor) for tensor in tensors]

def update_ratios(model):
    """Compute update ratio tensors for every learnable parameter in the model
    
    Concretely, compute the norm of the update and the norm of the weight.
    Ideally this value should be ~ 1e-3.
    
        source: http://cs231n.github.io/neural-networks-3/#ratio
    
    """
    names, weight_mags = weights(model)
    names, update_mags = updates(model)

    return [name[:-3]+'_ratio' for name in names], [up/mag for mag, up in zip(weight_mags, update_mags)]

def gradients(model):
    """Compute tensors magnitudes which correspond to the gradients for each
    learnable parameter

    Use gradients reported by the optimizer in the event gradient clipping is
    used.

    """
    names, tensors = trainable_weights(model)

    optimizer = model.optimizer
    grad_tensors = optimizer.get_gradients(model.total_loss, tensors)
    grad_mags = [norm(grad_tensor) for grad_tensor in grad_tensors]

    return [name+'_grad' for name in names], grad_mags

def activations(model, filters=['activation', 'dropout']):
    """Compute activation statistics (mean and stddev) for specified layers

    Parameters
    ----------
    model : keras model
    filters : string that specifies which layers to monitor

    """
    tensors = [layer for layer in model.layers if any(layer.name.startswith(filter) for filter in filters)]
    names = [tensor.name for tensor in tensors]

    mean_tensors, std_tensors = [0]*len(tensors), [0]*len(tensors)
    for i, tensor in enumerate(tensors):
        means = K.mean(K.equal(tensor.output, 0), axis=1)

        mean_tensors[i] = K.mean(means)
        std_tensors[i] = K.std(means)

    names = [name+'_mu' for name in names] + [name+'_std' for name in names]
    tensors = mean_tensors + std_tensors

    return names, tensors
