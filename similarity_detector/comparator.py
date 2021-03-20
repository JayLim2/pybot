from keras.models import load_model
import sys
from similarity_detector import input_handler
from similarity_detector import config
from operator import itemgetter

from similarity_detector import data_handler

########################
###### Testing #########
########################

def check_similarity():
    try:
        checkpoint = open('./similarity_detector/checkpoints/last_checkpoint.txt')
    except:
        print("Your model haven't been trained yet.")
        sys.exit()

    with checkpoint:
        last_checkpoint_path = checkpoint.readline()

    if last_checkpoint_path:
        last_checkpoint_path = "./similarity_detector/" + last_checkpoint_path.replace("./", "/")
        print('Check-point path:', last_checkpoint_path)

        model = load_model(last_checkpoint_path)

        test_sentence_pairs = [
            ('What can make Physics easy to learn?', 'How can you make physics easy to learn?'),
            ('How many times a day do a clocks hands overlap?',
             'What does it mean that every time I look at the clock the numbers are the same?'),
            ('Do you like burgers?', 'Do you like burger?'),
            ('Do you like burgers?', 'Do you like movies?')
        ]

        test_data_x1, test_data_x2, leaks_test = input_handler.create_test_data(
            data_handler.tokenizer,
            test_sentence_pairs,
            config.MAX_SEQUENCE_LENGTH
        )

        predictions = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
        results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, predictions)]
        results.sort(key=itemgetter(2), reverse=True)

        print("Results:")
        print(results)
    else:
        print("Your model haven't been trained yet.")