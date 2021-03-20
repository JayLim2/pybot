from similarity_detector import input_handler, config
import pandas as pd
import os

########################################
############ Data Preperation ##########
########################################

dirname = os.path.dirname(__file__)

df = pd.read_csv(dirname + '/data/dataset.csv')
sentences1 = list(df['sentences1'])
sentences2 = list(df['sentences2'])
is_similar = list(df['is_similar'])
del df

####################################
######## Word Embedding ############
####################################

# creating word embedding meta data for word embedding
tokenizer, embedding_matrix = input_handler.word_embed_meta_data(
    sentences1 + sentences2,
    config.siamese_config['EMBEDDING_DIM']
)

embedding_meta_data = {
    'tokenizer': tokenizer,
    'embedding_matrix': embedding_matrix
}

# creating sentence pairs
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
del sentences1
del sentences2
