# functions that can be consumed by the `TensorCallback` callback

import keras.backend as K

from support import get_trainable_weights, norm


FULL = False # set to true to return full instances of tensors


def updates(model):
    """Compute update magnitudes for every learnable tensor in the model
    
    Parameters
    ----------
    model : keras model instance

    """
    names, tensors = get_trainable_weights(model)

    # ignore updates for optimizer-specific tensors
    optimizer = model.optimizer
    tensor_pairs = [(p, p_new) for p, p_new in optimizer.updates if p.name in names]
    tensors = [new - old for new, old in tensor_pairs]
    if not FULL:
        tensors = [norm(update_tensor) for update_tensor in tensors]

    return [name+'_up' for name in names], tensors

def weights(model):
    """Compute weight magnitude for each parameter in the model
    
    Parameters
    ----------
    model : keras model instance
    
    """
    names, tensors = get_trainable_weights(model)
    if not FULL:
        tensors = [norm(tensor) for tensor in tensors]

    return [name+'_mag' for name in names], tensors

def update_ratios(model):
    """Compute update ratio tensors for every learnable parameter in the model
    
    Concretely, compute the norm of the update and the norm of the weight.
    Ideally this value should be ~ 1e-3.
    
        source: http://cs231n.github.io/neural-networks-3/#ratio
    
    """
    names, weight_mags = weights(model)
    names, update_mags = updates(model)

    return [name[:-3]+'_ratio' for name in names], [norm(up)/norm(mag) for mag, up in zip(weight_mags, update_mags)]

def gradients(model):
    """Compute tensors magnitudes which correspond to the gradients for each
    learnable parameter

    Parameters
    ----------
    model : keras model instance

    Use gradients reported by the optimizer in the event gradient clipping is
    used.

    """
    names, tensors = get_trainable_weights(model)

    optimizer = model.optimizer
    tensors = optimizer.get_gradients(model.total_loss, tensors)
    if not FULL:
        tensors = [norm(grad_tensor) for grad_tensor in tensors]

    return [name+'_grad' for name in names], tensors

def activations(model, layer_names=['study', 'summary', 'raw_score']):
    """Compute means and standard deviations of `layer_names`

    Parameters
    ----------
    model : keras model
    layer_names : name of layers to compute statistics for

    """
    tensors = [model.get_layer(layer_name).output for layer_name in layer_names]
    if not FULL:
        tensors = [norm(tensor) for tensor in tensors]

    return layer_names, tensors
