import keras.backend as K


def compute_f1(class_idx, y_true, y_pred):
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

def compute_acc(class_idx, y_true, y_pred):
    """Compute the class-level accuracy for `class_idx`

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

    return K.sum(tpfp * positives) / K.sum(positives).astype('float32')
