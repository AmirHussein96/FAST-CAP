""" adapted from https://github.com/keithito/tacotron """

import re
import numpy as np
from collections import defaultdict
from . import cleaners
from .cleaners import Cleaner
from .symbols import get_symbols
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
import logging
import functools
import string
import os
from pathlib import Path

#########
# REGEX #
#########

# Regular expression matching text enclosed in curly braces for encoding
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Regular expression matching words and not words
_words_re = re.compile(r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]+|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)")

_phonemizer_language_map = {
    'de_DE': 'de',
    'du_NL': 'nl',
    'en': 'en-us',
    'en_IN': 'en-us',
    'en_ES': 'en-us',
    'en_MN': 'en-us',
    'en_UK': 'en-gb',
    'en_US': 'en-us',
    'es_AR': 'es-419',
    'es_CL': 'es-419',
    'es_CO': 'es-419',
    'es_ES': 'es',
    'es_MX': 'es-419',
    'es_PE': 'es-419',
    'es_PR': 'es-419',
    'es_VE': 'es-419',
    'fi_FI': 'fi',
    'fr_FR': 'fr-fr',
    'gr_GR': 'el',
    'hi': 'hi',
    'hi_HI': 'hi',
    'hi_CH': 'hi',
    'it': 'it',
    'it_IT': 'it',
    'ko_KO': 'ko',
    'mar_MAR': 'mr',
    'pl_PL': 'pl',
    'pt_BR': 'pt-br',
    'ru_RU': 'ru',
    'sv_SV': 'sv',
    'te_TE': 'te',
    'ben_BEN': 'bn',
    'kan_KAN': 'kn'
}


def lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def get_phonemizer_parser(language):
    return _phonemizer_language_map[language]

def get_phonemizer_phonemes(phonemizer_backend_instance, text):
    separator = Separator(phone='|\p|', word='} {')

    # phonemizer sometimes merges words, so length can be different
    # ignore logging of warnings from phonemizer, only log crtitical errors
    logging.getLogger("phonemizer").setLevel(logging.ERROR)

    lexicon = phonemizer_backend_instance.phonemize([text],
                                                    separator=separator,
                                                    strip=True, njobs=1)[0]
    lexicon = lexicon.replace('|\p|', ' ')
    lexicon = '{' + lexicon + '}'
    return lexicon


def load_pronunciation_dictionary(dict_path, separator='\t'):
    """
    Load pronunciation dictionary from file.
    Expected format: word<separator>pronunciation
    Returns: dict mapping words to their pronunciations
    """
    pronunciation_dict = {}
    print(f"Loading pronunciation dictionary from {dict_path}")
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith(";;;"):
                    continue
                if line and separator in line:
                    # word, pronunciation = line.split(separator, 1)
                    if "\t" in line:
                        word, pronunciation = line.strip().split("\t")
                    else:
                        word, pronunciation = re.split(r'\s+', line.strip(), maxsplit=1)
                pronunciation_dict[word.lower()] = pronunciation.strip()
    except FileNotFoundError:
        logging.warning(f"Pronunciation dictionary not found at {dict_path}")
    except Exception as e:
        logging.error(f"Error loading pronunciation dictionary: {e}")
    
    return pronunciation_dict


def get_dictionary_phonemes(pronunciation_dict, text, fallback_phonemizer=None, dict_name=None):
    """
    Convert text to phonemes using pronunciation dictionary.
    Falls back to phonemizer for words not in dictionary.
    Preserves original punctuation and case.
    
    Args:
        pronunciation_dict: dict mapping words to pronunciations
        text: input text to convert
        fallback_phonemizer: phonemizer instance for fallback (optional)
    
    Returns:
        phonemized text in the same format as get_phonemizer_phonemes
    """
    
    words = text.split()
    phonemized_words = []
    
    for word in words:
        # Extract leading and trailing punctuation
        leading_punct = ''
        trailing_punct = ''
        
        # Find leading punctuation
        for i, char in enumerate(word):
            if char.isalnum():
                leading_punct = word[:i]
                break
        else:
            # If no alphanumeric characters found, the whole word is punctuation
            leading_punct = word
            trailing_punct = ''
            phonemized_words.append(word)
            continue
        
        # Find trailing punctuation
        for i in range(len(word) - 1, -1, -1):
            if word[i].isalnum():
                trailing_punct = word[i+1:]
                break
        
        # Get the clean word (without punctuation)
        clean_word = word[len(leading_punct):len(word)-len(trailing_punct)]
        clean_word_lower = clean_word.lower()
        
        if clean_word_lower in pronunciation_dict:
            # Use dictionary pronunciation
            phoneme = pronunciation_dict[clean_word_lower]
        elif fallback_phonemizer is not None:
            # Fallback to phonemizer
            fallback_phoneme = get_phonemizer_phonemes(fallback_phonemizer, clean_word_lower)
            # Remove the curly braces for individual word
            phoneme = fallback_phoneme.strip('{}')
        else:
            #raise ValueError(f"Word \"{clean_word_lower}\" not found in pronunciation dictionary and no fallback phonemizer available")
            
            if dict_name:
                print(f"Dictionary name: {dict_name}")
            print(f"Word \"{clean_word_lower}\" not found in pronunciation dictionary and no fallback phonemizer available")
            continue
        
        # Restore punctuation to the phonemized word
        phonemized_word = leading_punct + phoneme + trailing_punct
        phonemized_words.append(phonemized_word)
    
    # Join words with spaces and wrap in curly braces
    result = '{' + ' '.join(phonemized_words) + '}'
    return result


class TextProcessing(object):
    def __init__(self, symbol_set, cleaner_name, 
                 prepend_space_to_text=False,
                 append_space_to_text=False, add_bos_eos_to_text=False,
                 language='default', g2p_type='phonemizer', 
                 pronunciation_dict_path=None, separator='\t'):
        """
        Initialize TextProcessing with support for both phonemizer and dictionary-based G2P.
        
        Args:
            symbol_set: symbol set for the model
            cleaner_name: name of the text cleaner to use
            prepend_space_to_text: whether to prepend space to text
            append_space_to_text: whether to append space to text
            add_bos_eos_to_text: whether to add BOS/EOS tokens
            language: language code for phonemizer
            g2p_type: 'phonemizer' or 'dictionary' or 'hybrid'
            pronunciation_dict_path: path to pronunciation dictionary file (for dictionary/hybrid modes)
        """
        self.g2p_type = g2p_type
        self.cleaner_names = cleaner_name
        self.cleaner = Cleaner(cleaner_name)
        if pronunciation_dict_path:
            self.dict_name = os.path.basename(Path(pronunciation_dict_path))
        else:
            self.dict_name = None

        # Initialize phonemizer backend (used for phonemizer mode or as fallback)
        phonemizer_language = get_phonemizer_parser(language)
        self.phonemizer_backend = EspeakBackend(phonemizer_language,
                                                preserve_punctuation=True,
                                                with_stress=True,
                                                words_mismatch='ignore')
        
        # Initialize pronunciation dictionary if needed
        self.pronunciation_dict = None
        if g2p_type in ['dictionary', 'hybrid'] and pronunciation_dict_path:
            self.pronunciation_dict = load_pronunciation_dictionary(pronunciation_dict_path, separator=separator)
            if not self.pronunciation_dict:
                logging.warning(f"Failed to load pronunciation dictionary {pronunciation_dict_path}, falling back to phonemizer")
                self.g2p_type = 'phonemizer'
        
        self.symbols, self.markers, self.placeholder_set, self.dipthongs_set \
            = get_symbols(symbol_set)

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text
        self.add_bos_eos_to_text = add_bos_eos_to_text

        if add_bos_eos_to_text:
            self.symbols.append('<bos>')
            self.symbols.append('<eos>')

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        self.language = language
        self.phoneme_counter = defaultdict(functools.partial(defaultdict, int))
        self.grapheme_counter = defaultdict(functools.partial(defaultdict, int))

    def get_phonemes(self, text, dict_name=None):
        """
        Convert text to phonemes based on the configured G2P type.
        """
        if self.g2p_type == 'phonemizer':
            return get_phonemizer_phonemes(self.phonemizer_backend, text)
        elif self.g2p_type == 'dictionary':
            if self.pronunciation_dict:
                return get_dictionary_phonemes(self.pronunciation_dict, text, dict_name=dict_name)
            else:
                logging.warning("No pronunciation dictionary available, falling back to phonemizer")
                return get_phonemizer_phonemes(self.phonemizer_backend, text)
        elif self.g2p_type == 'hybrid':
            if self.pronunciation_dict:
                return get_dictionary_phonemes(self.pronunciation_dict, text, self.phonemizer_backend)
            else:
                logging.warning("No pronunciation dictionary available, falling back to phonemizer")
                return get_phonemizer_phonemes(self.phonemizer_backend, text)
        else:
            raise ValueError(f"Unsupported G2P type: {self.g2p_type}")

    def text_to_sequence(self, text):
        sequence = []

        # Check for curly braces and treat their contents as phoneme:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(text)
                break
            sequence += self.symbols_to_sequence(m.group(1))
            sequence += self.phoneme_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        result = ''
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose phoneme back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')

    def clean_text(self, text):
        text = self.cleaner(text)
        return text

    def parse_placeholder(self, marker, text, placeholder_type):
        placeholder_set = self.placeholder_set[placeholder_type]
        parsed_token = None

        if placeholder_type == 'right' and len(text) > 1:
            # make sure text at index+1 gets applied the marker
            syllable = text[1]
            parsed_token = marker + syllable
            remaining_text = text[2:]
        elif placeholder_type == 'other':
            # marker is separate
            parsed_token = marker
            remaining_text = text[1:]
        else:
            # to apply marker to text[0]
            syllable = text[0]
            parsed_token = syllable + marker
            remaining_text = text[2:]

        return parsed_token, remaining_text

    def parse_phonemized_text(self, text):
        """
        recursively get the token string and split it based on markers and placeholders
        args: text: input text to be parsed
        returns list of tokens
        """

        if len(text) == 0:
            return []

        parsed_tokens = [] # return can be a list of tokens

        if text[0] in self.placeholder_set['right']:
            # find which marker and apply parsing to the rest of the string
            # marker application with right placeholder
            parsed_token, remaining_text = self.parse_placeholder(text[0], text, 'right')

        elif text[0] in self.placeholder_set['other']:
            # marker application with other placeholder
            parsed_token, remaining_text = self.parse_placeholder(text[0], text, 'other')
        else:
            if len(text) > 1 and text[1] in self.placeholder_set['left']:
                lookahead_character = text[1]
                parsed_token, remaining_text = self.parse_placeholder(lookahead_character, text, 'left')
            elif len(text) > 1:
                parsed_token = text[0]
                remaining_text = text[1:]
                for i in range(len(text)):
                    if text[:i+1] in self.dipthongs_set:
                        parsed_token = text[:i+1]
                        remaining_text = text[i+1:]
            else:
                # no marker match, must be independent syllable, leave as is
                parsed_token = text[0]
                remaining_text = text[1:]

        tokens = [parsed_token] + self.parse_phonemized_text(remaining_text)
        return tokens


    def symbols_to_sequence(self, symbols):
        cur_symbols = []
        for s in symbols:
            if s in self.symbol_to_id:
                cur_symbols.append(self.symbol_to_id[s])
            else:
                if self.placeholder_set == None:
                    for sym in symbols:
                        if sym != '@':
                            if '@' + sym in self.symbol_to_id:
                                cur_symbols.append(self.symbol_to_id['@' + sym])
                else:
                    tokens = self.parse_phonemized_text(s)
                    for token in tokens:
                        if token != '@':
                            if '@' + token in self.symbol_to_id:
                                cur_symbols.append(self.symbol_to_id['@' + token])
                            else:
                                # parse character by character
                                for sym in token:
                                    if sym != '@':
                                        if '@' + sym in self.symbol_to_id:
                                            cur_symbols.append(self.symbol_to_id['@' + sym])
        return cur_symbols

    def phoneme_to_sequence(self, text):
        return self.symbols_to_sequence(['@' + s for s in text.split()])

    def encode_text(self, text, return_all=False, output_grapheme=False):
        text_clean = self.clean_text(text)
        text = text_clean
        if output_grapheme:
            text_phoneme = text
        else:
            text_phoneme = self.get_phonemes(text, self.dict_name)
        text_encoded = []
        text_encoded = self.text_to_sequence(text_phoneme)
        
        if self.prepend_space_to_text:
            text_encoded.insert(0, self.symbol_to_id[' '])

        if self.append_space_to_text:
            text_encoded.append(self.symbol_to_id[' '])

        if self.add_bos_eos_to_text:
            text_encoded.insert(0, self.symbol_to_id['<bos>'])
            text_encoded.append(self.symbol_to_id['<eos>'])

        if return_all:
            return text_encoded, text_clean, text_phoneme

        return text_encoded