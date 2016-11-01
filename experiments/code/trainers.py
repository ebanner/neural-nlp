from collections import OrderedDict

from keras.layers import Input, Dropout, Dense, merge
from keras.layers import Activation
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer


class PICOTrainer(Trainer):
    """Multi-input interpretable PICO architecture

    Send each PICO vector through a linear layer to a single score which are
    then concatenated and sent through a linear sigmoid layer.
    
    """
    def build_model(self, dropout_pico, backprop_pico, reg):

        inputs = OrderedDict()
        for pico_element in self.pico_elements:
            inputs[pico_element] = Input(shape=self.X[pico_element].shape[1:])

        scores = OrderedDict()
        for pico_element in self.pico_elements:
            scores[pico_element] = Dense(output_dim=1, activation='relu')(inputs[pico_element])

        # dot vectors and send through sigmoid
        score = merge(inputs=scores.values(),
                      mode='concat',
                      name='pico_scores')

        prob = Dense(output_dim=1, activation='sigmoid')(score)

        self.model = Model(input=inputs.values(), output=prob)
