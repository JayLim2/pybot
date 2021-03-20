import sys
import os
sys.path.append(os.getcwd().replace("similarity_detector", ""))

from similarity_detector.data_handler import sentences_pair, is_similar, embedding_meta_data
from model import SiameseBiLSTM
from config import siamese_config

##########################
######## Training ########
##########################

best_model_path = None

class Configuration(object):
    """Dump stuff here"""

if __name__ == "__main__":
    CONFIG = Configuration()
    CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
    CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
    CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
    CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
    CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
    CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
    CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
    CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

    siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units,
                            CONFIG.number_dense_units,
                            CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function,
                            CONFIG.validation_split_ratio)

    best_model_path = siamese.train_model(
        sentences_pair,
        is_similar,
        embedding_meta_data,
        model_save_directory='./'
    )
