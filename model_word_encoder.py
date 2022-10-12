import enum
import logging

import numpy
from tensorflow import float32
from tensorflow.keras import Input
#from tensorflow.keras import Output
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import losses
from tensorflow import optimizers
from tensorflow.keras.utils import plot_model


#--------------------------------------------------------------------------------------------------------------------------------
#   Word
#--------------------------------------------------------------------------------------------------------------------------------

class Word:
    """Encapsulate a word as input to a neural network"""

    #----------------    Constructor    ----------------

    def __init__( self ):
        """Empty constructor"""
        #initialize class vars
        self.__set_word( "" )

        return

    def __init__( self, is_word : str() ):
        """Initialized Constructor"""
        #initialize class vars
        self.__set_word( is_word )

        return

    #----------------    Overloads    ---------------

    def __str__(self) -> str:
        """Stringfy class for print method"""
        return f"Length: {self.gn_digits} | Word: {self.gln_word}"

    #----------------    Private Members    ---------------

    def __set_word( self, is_word : str() ) -> bool:
        """Translate a word in str() form into a Word class. It's a fixed length vector with characters in UTF8 representation.
        Returns:
            bool: False=OK | True=FAIL
        """
        #fixed length
        N_MAX_DIGIT = 32
        #allocate a vector of the correct type of a given lenght. This word rapresentation is fixed length to be used by the model later
        ln_word = numpy.zeros( N_MAX_DIGIT, dtype=numpy.uint8 )
        #encode the word str() as UTF8
        try:
            is_word.encode( encoding='UTF-8', errors='strict' )
        except Exception as s_err:
            logging.debug(f"Encode error: {s_err}")
            self.gn_digits = 0
            self.gln_word = ln_word
            return True
        #scan the word in str() form
        for n_index, c_digit in enumerate( is_word ):
            #write the index in the mapping (UTF8 is ASCII) as number inside the array
            ln_word[n_index] = ord( c_digit )
        #save the results inside the class
        self.gn_digits = len( ln_word )
        self.gln_word = ln_word
        logging.debug(f"Length: {self.gn_digits} | Word: {self.gln_word}")
        return False

    def set_word( self, iln_digit : numpy ) -> bool:
        """Set a word according to a numpy array
        Args:
            iln_digit (numpy): [description]
        """
        if len(iln_digit) != self.gn_digits:
            return True
        self.gn_digits = iln_digit
        return False

#--------------------------------------------------------------------------------------------------------------------------------
#   Dictionary
#--------------------------------------------------------------------------------------------------------------------------------

#class Dictionary : Word()
#    def __init__( self ):
#        return


#--------------------------------------------------------------------------------------------------------------------------------
#   Encoded Word
#--------------------------------------------------------------------------------------------------------------------------------

class Encoded_word:
    """Encoded word, output of Word Encoder"""

    #----------------    Constructor    ----------------

    def __init__( self ):
        """Empty constructor"""
        #initialize class vars
        self.N_ENCODED_WORD_LENGTH = 2
        self.ln_encoded_word = numpy.array( self.N_ENCODED_WORD_LENGTH, dtype=numpy.float32 )
        return

#--------------------------------------------------------------------------------------------------------------------------------
#   ENCODER Word -> Encoded_word
#--------------------------------------------------------------------------------------------------------------------------------

class Encoder_word_to_encoded_word:

    #----------------    Constructor    ----------------

    def __init__( self ):

        #initialize Model
        self.c_model = None

        return

    #----------------    Public Methods    ----------------

    def build( self, ic_word: Word, ic_encoding: Encoded_word ) -> bool:
        """Construct a ML model based on the shape of a single instance of Perception and Action classes
        Args:
            ic_word (Word): Word class. Represent a word as an array of numbers in ascii
            ic_encoding (Encoded_word): Encoded_word. Vector space representing a word
        Returns:
            bool: False=OK | True=FAIL
        """
        
        if (True):
            c_model = Sequential( name="Encoder_W_to_eW" )
            c_model.add( layers.Flatten(input_shape=(ic_word.gn_digits,) ) )
            c_model.add( layers.Dense(10) )
            c_model.add( layers.Dense( ic_encoding.N_ENCODED_WORD_LENGTH ) )
            c_model.compile( loss = losses.MeanSquaredError(), optimizer = optimizers.Adam( clipnorm=1 ) )
            #my_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
            c_model.summary()
            #plot_model(c_model,to_file='encoder.png',show_shapes=True)

            #ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.
            print( "Summary: ", c_model.summary() )
            self.c_model = c_model



        else:
            self.c_model = None

    
    def inference( self, c_in : Word ) -> Encoded_word:
        """Run inference using the model"""

        if (self.c_model is None):
            return Encoded_word()

#--------------------------------------------------------------------------------------------------------------------------------
#   DECODER Encoded_word -> Word
#--------------------------------------------------------------------------------------------------------------------------------

class Decoder_encoded_word_to_word:

    #----------------    Constructor    ----------------

    def __init__( self ):
        #initialize Model
        self.c_model = None
        return

    #----------------    Public Methods    ----------------

    def build( self, ic_input_encoding: Encoded_word, ic_output_word: Word ) -> bool:
        """Construct a ML model for a Decoder
            ic_input_encoding (Encoded_word): Encoded_word. Vector space representing a word
            ic_output_word (Word): Word class. Represent a word as an array of numbers in ascii
        Returns:
            bool: False=OK | True=FAIL
        """

        if (True):
            c_model = Sequential(name="Decoder_eW_to_W")
            c_model.add( layers.Flatten(input_shape=(ic_input_encoding.N_ENCODED_WORD_LENGTH,) ) )
            c_model.add( layers.Dense(10) )
            c_model.add( layers.Dense( ic_output_word.gn_digits ) )
            c_model.compile( loss = losses.MeanSquaredError(), optimizer = optimizers.Adam( clipnorm=1 ) )
            #my_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
            c_model.summary()
            plot_model(c_model,to_file='encoder.png',show_shapes=True)

            #ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.
            print( "Summary: ", c_model.summary() )
            self.c_model = c_model
        else:
            c_model = None

    #def train( self, lc_input, l)

#--------------------------------------------------------------------------------------------------------------------------------
#   Unified
#--------------------------------------------------------------------------------------------------------------------------------
#

class Hourglass_encoder_decoder:

    #----------------    Constructor    ----------------

    def __init__( self ):

        #initialize Model
        self.c_model = None

        return

    #----------------    Getters    ----------------

    def get_word_matrix( self, ilc_words : list() ) -> numpy.array:
        """From a list of Words, create numpy matricies useable for training
        Args:
            ilc_words (list): list of Word()
        Returns:
            numpy array
        """
        n_num_words = len( ilc_words )
        #Generate a matrix of shape #words*#digits to feed the fitting. Is used for both input and output
        lnn_digit = numpy.zeros( (n_num_words, ilc_words[0].gn_digits ), dtype=numpy.uint8 )
        for n_index_word, ln_word in enumerate(ilc_words):
            lnn_digit[n_index_word, :] = ln_word.gln_word

        return lnn_digit

    def set_list_word( self, inn_digits : numpy.array ) -> list:
        """from a numpy array create a list of Words"""
        lc_word = list()
        #scan all words
        for ln_word in inn_digits[:,]:
            c_word = Word("")
            c_word.set_word( ln_word )
            lc_word.append( Word("") )


        return lc_word

    #----------------    Public Methods    ----------------

    def build( self, ic_word: Word, ic_encoding: Encoded_word ) -> bool:
        """Construct a ML model based on the shape of a single instance of Perception and Action classes
        Args:
            ic_word (Word): Word class. Represent a word as an array of numbers in ascii
            ic_encoding (Encoded_word): Encoded_word. Vector space representing a word
        Returns:
            bool: False=OK | True=FAIL
        """

        #if (type(ic_word) is not Word) or (type(ic_encoding) is not Encoded_word) :
        #    logging.debug(f"Wrong input types")
        #    return True

        print(f"Input: {ic_word}")
        if (True):
            c_model = Sequential( name="Encoder_W_to_eW" )
            c_model.add( layers.Flatten(input_shape=(ic_word.gn_digits,) ) )
            c_model.add( layers.Dense(10))
            c_model.add( layers.Dense( ic_encoding.N_ENCODED_WORD_LENGTH, activation='relu' ) )
            c_model.add( layers.Dense(10) )
            c_model.add( layers.Dense(ic_word.gn_digits) )

            c_model.compile( loss = losses.MeanSquaredError(), optimizer = optimizers.Adam( clipnorm=1 ) )
            #my_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
            c_model.summary()
            #plot_model(c_model,to_file='encoder.png',show_shapes=True)

            #ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.
            print( "Summary: ", c_model.summary() )
            plot_model(c_model,to_file='hourglass.png',show_shapes=True)
            self.gc_model = c_model

        else:
            self.gc_model = None

    def train( self, ilc_words : list() ):
        """train the model based on a list of Words
        Args:
            ilc_words (list): list of words
        """

        n_num_words = len( ilc_words )

        print(f"Number of Words: {n_num_words}")
        print(f"First Word: {ilc_words[0]}")
        print(f"Last Word: {ilc_words[-1]}")

        lnn_digit = self.get_word_matrix( ilc_words )

        print( f"Database: {lnn_digit}")
        #fit the model to words
        self.gc_model.fit( lnn_digit, lnn_digit, epochs=1)
        #use the module to make prediction
        lnn_prediction = self.gc_model.predict( lnn_digit )
        #Construct a list of Words
        print( lnn_prediction )
        ilc_predicted_words = self.set_list_word( lnn_prediction )
        print( ilc_predicted_words )

        return ilc_predicted_words

    def inference( self, c_in : Word ) -> Encoded_word:
        """Run inference using the model"""

        if (self.c_model is None):
            return Encoded_word()


class Train_encoder_decoder:
    """Receives an encoder and a decoder, and train them on a list of words"""

    #----------------    Constructor    ----------------

    def __init__( self ):
        """Empty constructor"""
        self.gc_model_encoder = None
        self.gc_model_decoder = None
        return

    def __init__( self, ic_model_encoder: Sequential(),ic_model_decoder: Sequential() ):
        """Initialized constructor"""
        self.set_encoder_decoder( ic_model_encoder, ic_model_decoder )

        return

    #----------------    Setter    ----------------

    def set_encoder_decoder( self, ic_model_encoder: Sequential(),ic_model_decoder: Sequential() ):
        """Assign the models
        Args:
            ic_model_encoder: Encoder
            ic_model_decoder: Decoder
        Returns:
            bool: False=OK | True=FAIL
        """
        self.gc_model_encoder = ic_model_encoder
        self.gc_model_decoder = ic_model_decoder
        return False


    def xxx(self):
        layers.concatenate( self.gc_model_encoder, self.gc_model_decoder )