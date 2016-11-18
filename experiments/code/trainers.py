from collections import OrderedDict

from keras.layers import Input, Dropout, Dense, merge, Lambda
from keras.layers import Activation
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer


class PICOTrainer(Trainer):
    """Multi-input interpretable PICO architecture

    Send each PICO vector through a linear layer to a single score which are
    then multiplied for a final score.
    
    """
    def build_model(self, dropout_pico, backprop_pico, reg, interaction_layer):

        inputs = OrderedDict()
        for input in self.inputs:
            inputs[input] = Input(shape=self.X[input].shape[1:], name=input)

        probs, outputs = OrderedDict(), OrderedDict()
        for input in self.inputs:
            probs[input] = Dense(output_dim=1, activation='sigmoid', name='{}-prob'.format(input))(inputs[input])
            if input in self.outputs:
                outputs[input] = probs[input] # only add prob if it's one of the outputs

        if interaction_layer == 'mul':
            from support import mul_merge
            prob = merge(inputs=probs.values(), mode=mul_merge, output_shape=[1], name='label-prob')
        else:
            concatted_probs = merge(inputs=probs.values(), mode='concat')
            prob = Dense(output_dim=1, name='label-prob')(concatted_probs)

        self.model = Model(input=inputs.values(), output=[prob]+outputs.values())

class LogisticRegressionTrainer(Trainer):
    """Logistic regression model"""

    def build_model(self, dropout_pico, backprop_pico, reg, interaction_layer):
        input = Input(shape=self.X['BoW'].shape[1:], name='BoW')
        prob = Dense(output_dim=1, activation='sigmoid', W_regularizer=l2(reg), name='label-prob')(input)
        self.model = Model(input=input, output=prob)
