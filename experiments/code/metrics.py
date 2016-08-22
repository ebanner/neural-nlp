import keras.backend as K


def compute_acc(class_idx, multi_label, y_true, y_pred):
    """Compute the class-level accuracy for `class_idx`

    Parameters
    ----------
    class_idx : index of the positive class
    multi_label : whether there are multiple labels
    y_true : categorical 2darray one-hot encoding
    y_pred : 2darray of predicted class probabilities

    """
    idx = K.variable(value=class_idx, dtype='int8') # index of class to compute
    
    # select out only class of interest
    y_pred_ = y_pred[:, idx]
    y_true_ = y_true[:, idx]

    # threshold predictions in the multi-label setting
    y_pred__  = K.switch(y_pred_ <= 0.5, 0, 1) if multi_label else y_pred_

    return K.mean(K.equal(y_pred__, y_true_)) # compute accuracy

def compute_f1(class_idx, multi_label, y_true, y_pred):
    """Compute the f1 score taking `class_idx` as the class to care about

    Parameters
    ----------
    class_idx : index of the positive class
    y_true : categorical 2darray one-hot encoding
    y_pred : 2darray of predicted class probabilities

    """
    y_true_ = K.argmax(y_true, axis=-1)
    y_pred_ = K.argmax(y_pred, axis=-1)
    
    idx = K.variable(value=class_idx) # index of class to compute
    
    tpfp = K.equal(y_pred_, idx) # true positives + false positives
    positives = K.equal(y_true_, idx) # all the real positives out there
    
    tp = tpfp * positives
    
    precision = K.sum(tp) / K.sum(tpfp).astype('float32')
    recall = K.sum(tp) / K.sum(positives).astype('float32')

    f1_numer = 2 * precision * recall
    f1 = f1_numer / (precision + recall)
    
    return f1
