# --------------------------------------------------------------------------------------------------------------------------------
#   IMPORT
# --------------------------------------------------------------------------------------------------------------------------------

import os
import logging

import matplotlib.pyplot as plt
import numpy

from numpy import savetxt

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from tensorflow.keras import Sequential



#--------------------------------------------------------------------------------------------------------------------------------
#   CONFIGURATION
#--------------------------------------------------------------------------------------------------------------------------------

S_TRAINING_FOLDER = "Training"
N_MAX_DIGIT = 32

#--------------------------------------------------------------------------------------------------------------------------------
#   LIST OF TRAINING WORDS
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
    logging.info(f"Replays found: {len(ls_filenames)} | {ls_filenames} | path: {s_filepath}")

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
    logging.info( s_tmp )
    print(s_tmp) 
    return ls_database

#--------------------------------------------------------------------------------------------------------------------------------
#   DATASET
#--------------------------------------------------------------------------------------------------------------------------------

def construct_dataset( ils_words : list( str() ) ):
    """From a list of words, get a bidimensional numpy array of the training set for the autoencoder
    Parameters:
        ils_words: list of words
    Returns:
        nn_dataset: bidimensional matrix of characters, normalized to be between 0 and 1
        nn_offset: offset needed to denormalize the data
        nn_bias: bias needed to denormalize the data
    """
    #number of words in the list
    n_words = len( ils_words )
    logging.info( f"Words: {n_words}" )
    #allocate the empty set
    nn_dataset = numpy.zeros( (n_words, N_MAX_DIGIT ), dtype=numpy.uint8 )
    #for each training word
    for n_row, s_word in enumerate( ils_words ):
        #encode the word str() as UTF8
        try:
            s_word.encode( encoding='UTF-8', errors='strict' )
        except Exception as s_err:
            logging.error(f"Encode error: {s_err}")
            return True
        #scan the word in str() form
        for n_index, n_digit in enumerate( s_word ):
            #write the index in the mapping (UTF8 is ASCII) as number inside the array
            nn_dataset[n_row][n_index] = ord( n_digit )
    logging.info( f"Dataset Shape: {nn_dataset.shape}" )
    return nn_dataset

def normalize_dataset( inn_dataset ):
    """Normalize between 0 and 1
    Parameters:
        inn_dataset: dataset
    Returns:
        nn_dataset: normalized dataset
        nn_offset: offset needed to denormalize the data
        nn_bias: bias needed to denormalize the data
    """
    #
    n_bias = numpy.min( inn_dataset )
    n_gain = numpy.max( inn_dataset ) -n_bias
    nn_dataset = (inn_dataset -n_bias) /n_gain
    logging.info( f"Bias: {n_bias} | Gain: {n_gain}")
    return (nn_dataset, n_bias, n_gain)


class Autoencoder( Model ):
    """Autoencoder for words"""
    def __init__(self, in_input_word_length, in_latent_dim, in_step ):
        """Constructor for Autoencoder
        Parameters:
            in_input_word_length: input word size
            in_latent_dim: number of hidden dimensions
            in_step: width of each dense step is reduced by the encoder/increased by the decoder by this amount until the next step gets to the final size
        Returns:

        """
        #???
        super(Autoencoder, self).__init__()
        #save inside the autoecoder the number of latent dimensions
        self.n_input_word_length = in_input_word_length
        self.n_latent_dim = in_latent_dim 
        #Construct the Encoder model
        #self.encoder = Sequential( name=f"Encoder|{in_input_word_length}->{in_latent_dim}" )
        self.encoder = Sequential( name=f"Encoder" )
        n_width = in_input_word_length
        while (n_width > in_latent_dim):
            self.encoder.add( layers.Dense(n_width, activation='relu') )
            n_width -= in_step
        self.encoder.add( layers.Dense(in_latent_dim, activation='sigmoid') )
        #construct the Decoder model
        #self.decoder = Sequential( name=f"Decoder{in_latent_dim}->{in_input_word_length}" )
        self.decoder = Sequential( name=f"Decoder" )
        n_width = in_latent_dim
        while (n_width < in_input_word_length):
            self.decoder.add( layers.Dense(n_width, activation='relu') )
            n_width += in_step
        self.decoder.add( layers.Dense( in_input_word_length, activation='sigmoid') )
        #compile the model
        self.compile( optimizer='adam', loss=losses.MeanSquaredError() )
        #self.build()

    def call( self, inn_data ):
        """Overloads round bracket operator to execute the endoder and decoder in sequence on a given input image.
        Parameters:
            inn_data: 
        Returns:
            decoded:
        """
        #encodes an image into a latent vector
        n_latent = self.encoder( inn_data )
        #decodes a latent vector into an image
        nn_decoded = self.decoder( n_latent )
        return nn_decoded

    def summary( self ):
        s_ret = str()
        s_ret += f"Encoder: {self.encoder.summary()} | "
        s_ret += f"Decoder: {self.decoder.summary()} | "
        return s_ret
  
#--------------------------------------------------------------------------------------------------------------------------------
#   EXECUTION
#--------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig( level=logging.INFO, format='[%(asctime)s] %(module)s:%(lineno)d %(levelname)s> %(message)s' )

    n_latent_dim = 10
    n_step_down = 2
    logging.info(f"Latent Dimensions: {n_latent_dim} | Step Down: {n_step_down}")

    #Load the list of words to be used for training the autoencoder
    ls_training_files = get_training_files( S_TRAINING_FOLDER )
    ls_training_words = get_training_data( ls_training_files[0] )
    #convert the list of words into a two dimensional matrix, from there, extract the training and validation datasets
    nn_dataset = construct_dataset( ls_training_words )
    (nn_dataset, n_bias, n_gain) = normalize_dataset( nn_dataset )

    nn_dataset_train = nn_dataset[0:60000]
    nn_dataset_test = nn_dataset[60000:]
    logging.info( f"Train: {nn_dataset_train.shape} | Test: {nn_dataset_test.shape}")
    #construct and train the autoencoder
    autoencoder = Autoencoder( N_MAX_DIGIT, n_latent_dim, n_step_down ) 
    autoencoder.fit( nn_dataset_train, nn_dataset_train, epochs=10, shuffle=True, validation_data=(nn_dataset_test, nn_dataset_test) )
    logging.info(f"Summary: {autoencoder.summary()} ")

    encoded = autoencoder.encoder( nn_dataset_test ).numpy()
    decoded = autoencoder.decoder( encoded ).numpy()

    savetxt("test.csv", nn_dataset_test, delimiter=',', fmt='%1.3f')
    savetxt("encoded.csv", encoded, delimiter=',', fmt='%1.3f')
    savetxt("decoded.csv", decoded, delimiter=',', fmt='%1.3f')
