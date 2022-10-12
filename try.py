import string
import numpy

def string_to_array_a( is_word ) -> numpy.array:
    ln_word = numpy.zeros( 32 )
    for n_digit_index,n_digit in enumerate( is_word ):
        if (str.isalnum(n_digit) == True):
            #numpy.float = str()
            ln_word[n_digit_index] = n_digit

def string_to_array_b( is_word ) -> numpy.array:
    return numpy.fromstring( is_word )

def string_to_array_c( is_word ) -> numpy.array:
    return numpy.fromstring( is_word, dtype=numpy.uint8 )

def string_to_array_d( is_word ) -> numpy.array:
    return numpy.frombuffer( is_word.encode(), dtype=numpy.uint8 )

def string_to_array_e( is_word ) -> numpy.array:
    return numpy.frombuffer( is_word.encode(), dtype=numpy.float )

if __name__ == "__main__":
    s_word = "Shaka"
    ln_word = list()
    #ln_word = string_to_array_a( s_word )
    #ln_word = string_to_array_b( s_word )
    #ln_word = string_to_array_c( s_word )
    ln_word = string_to_array_d( s_word )
    #ln_word = string_to_array_e( s_word )

    print( "Word: ", s_word, " -> Length:", len(ln_word)," | Word: ", ln_word )