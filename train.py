#IDEA: Hyperparameter, encoded vector size. Save training time and error. Plot vector size to error at a number of iteration

# --------------------------------------------------------------------------------------------------------------------------------
#   IMPORT
# --------------------------------------------------------------------------------------------------------------------------------

from email.generator import DecodedGenerator
import logging
#Used to search folders for files
import os

import numpy
#Input of the Encoder. Represent a word as a vector of numbers.
from model_word_encoder import Word
#Output of the Encoder, Input of the decoder. Represent a word in a compressed vector space.
from model_word_encoder import Encoded_word
#Encoder. Word->Encoded_word
from model_word_encoder import Encoder_word_to_encoded_word
#Decoder. Encoded_word->Word
from model_word_encoder import Decoder_encoded_word_to_word


from model_word_encoder import Hourglass_encoder_decoder

from model_word_encoder import Train_encoder_decoder

#--------------------------------------------------------------------------------------------------------------------------------
#   CONFIGURATION
#--------------------------------------------------------------------------------------------------------------------------------

TRAINING_FOLDER = "Training"

#--------------------------------------------------------------------------------------------------------------------------------
#   Helper functions
#--------------------------------------------------------------------------------------------------------------------------------

def get_training_files( is_folder : str ) -> list:
    """ Search a folder for .txt files holding a list of training words
    Args:
        is_folder (str): folder with trainind data
    Returns:
        list: list of trainind data files
    """
    #folder with replays
    s_filepath = os.path.join( os.getcwd(), is_folder )
    ls_files = os.listdir( s_filepath )
    #search for all filenames in the list for files with the right extension
    ls_filenames = [ s_filepath +'\\' + s_file for s_file in ls_files if s_file.find(".txt") > -1 ]
    logging.debug(f"Replays found: {len(ls_filenames)} | {ls_filenames} | path: {s_filepath}")

    return ls_filenames

def get_training_data( is_file : str ) -> list:
    """ Search a .txt files for words, return a word training set
    Args:
        is_file (str): file with training words
    Returns:
        list: list of training words
    """

    ls_database = list()
    #try to load the file    
    try:
        with open(is_file ,encoding='unicode_escape') as c_opened_file:
            for s_word in c_opened_file:
                s_cleaned_word = str.join('', [c_digit for c_digit in s_word if c_digit.isalnum()] )
                ls_database.append( s_cleaned_word )
    #failed to open file
    except OSError as problem:
        logging.error(f'problem: {problem}')
        return -1

    s_tmp = f"Found {len(ls_database)} words for a total of  characters"
    logging.debug( s_tmp )
    print(s_tmp) 
    return ls_database

#def get_training_input_word( list() )

#--------------------------------------------------------------------------------------------------------------------------------
#   MAIN
#--------------------------------------------------------------------------------------------------------------------------------

#   if interpreter has the intent of executing this file
if __name__ == "__main__":
    logging.basicConfig( filename="debug.log",level=logging.DEBUG, format='[%(asctime)s] %(module)s:%(lineno)d %(levelname)s> %(message)s' )
    logging.debug(f"Train Word Encoder DecoderModels")
    print("hello world")

    ls_training_files = get_training_files( TRAINING_FOLDER )

    ls_training_words = get_training_data( ls_training_files[0] )

    lc_training_words = list()
    for s_word in ls_training_words:
        lc_training_words.append( Word( s_word ) )
    print( f"length: {len(lc_training_words)} | Word[0] {lc_training_words[0]}" )

    #Test Word interface
    #c_word = Word( ls_training_words[0])
    #c_word = Word( ls_training_words[1000])

    if (False):
        c_my_encoder = Encoder_word_to_encoded_word()
        c_my_encoder.build( Word(""), Encoded_word() )
        c_my_encoder.inference( Word("Shaka") )

        c_my_decoder = Decoder_encoded_word_to_word()
        c_my_decoder.build( Encoded_word(), Word("") )

        c_my_combined_model = Train_encoder_decoder( c_my_encoder, c_my_decoder )
        #c_my_combined_model.xxx()

    if (True):
        c_hourglass = Hourglass_encoder_decoder()
        c_hourglass.build( Word(""), Encoded_word() )
        lnn_prediction = c_hourglass.train( lc_training_words )
        


    #print( ls_training_words )