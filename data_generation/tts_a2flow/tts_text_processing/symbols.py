""" adapted from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text
that has been run through Unidecode. For other data, you can modify
_characters.'''

arpabet = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1',
    'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0',
    'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1',
    'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T',
    'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y',
    'Z', 'ZH'
]

ipa = [
    'aɪ', 'aʊ', 'b', 'd', 'dʒ', 'e', 'eɪ', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'oʊ', 'p', 'r', 's', 't', 'tʃ', 'u', 'v', 'w', 'z', 'æ', 'ð',
    'ŋ', 'ɑ', 'ɔ', 'ɔɪ', 'ə', 'ə', 'ər', 'ɜr', 'ɪ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'θ'
]

ipa_dict = [
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ŋ', 'ŭ', 'ɐ', 'ɑ', 'ɒ',
    'ɔ', 'ə', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɣ', 'ɪ', 'ɫ', 'ɬ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ',
    'ʌ', 'ʎ', 'ʒ', 'ʝ', 'β', 'θ']

# from https://github.com/espeak-ng/espeak-ng/blob/master/docs/phonemes.md
phonemizer_markers = {
    'stress': ['ˈ', 'ˌ'],
    'length_placeholder_left': ['◌̆', '◌ˑ', '◌ː', '◌ːː'],
    'rhythm': ['.', '◌‿◌'],
    'tones_placeholder_left': ['◌˥', '◌˦', '◌˧', '◌˨', '◌˩', 'ꜛ◌', 'ꜜ◌'],
    'tones_placeholder_right': ['ꜛ◌', 'ꜜ◌'],
    'intonation': ['`', '‖', '↗︎', '↘︎'],
    'fortis_placeholder_left': ['◌͈'],
    'lenis_placeholder_left': ['◌͉'],
    'lesser_oral_pressure_placeholder_left': ['◌͈'],
    'greater_oral_pressure_placeholder_left': ['◌͉'],
    'articulation_placeholder_left': ['◌ʲ', '◌ˠ', '◌̴', '◌ˤ', '◌̴', '◌̃', '◌˞']
}

phonemizer_diacritics = ['!', '[', ';', '^', '<H>', '<h>',
                        '<o>', '<r>', '<w>', '<?>',
                        '~', '-', '.', '"', '`']

phonemizer_extra_symbols = ['ɚ', 'ɝ', 'R', 'R<umd>', '¿',
                            '¡', 'ᵻ', '!', '"', ';', 'ɚ', 'ɟ']

wiki_numbers = '0123456789'
wiki_math = '#%&*+-/[]()'
wiki_special = '_@©°½—₩€$'

wiki_ipa_consonants = [
    # from wikipedia: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet_chart
    # Pulmonic
    'm̥', 'm', 'ɱ', 'n̼', 'n̥', 'n', 'ɳ̊', 'ɳ', 'ɲ̊', 'ɲ', 'ŋ̊', 'ŋ', 'ɴ',
    'p', 'b', 'p̪', 'b̪', 't̼', 'd̼', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'ɡ', 'q', 'ɢ', 'ʡ', 'ʔ',
    'ts', 'dz', 't̠ʃ', 'd̠ʒ', 'tʂ', 'dʐ', 'tɕ', 'dʑ',
    'pɸ', 'bβ', 'p̪f', 'b̪v', 't̪θ', 'd̪ð', 'tɹ̝̊', 'dɹ̝', 't̠ɹ̠̊˔', 'd̠ɹ̠˔', 'cç', 'ɟʝ', 'kx', 'ɡɣ', 'qχ', 'ɢʁ', 'ʡʜ', 'ʡʢ', 'ʔh',
    's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ɕ', 'ʑ',
    'ɸ', 'β', 'f', 'v', 'θ̼', 'ð̼', 'θ', 'ð', 'θ̠', 'ð̠', 'ɹ̠̊˔', 'ɹ̠˔', 'ɻ̊˔', 'ɻ˔', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'h', 'ɦ',
    'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ', 'ʔ̞',
    'ⱱ̟', 'ⱱ', 'ɾ̼', 'ɾ̥', 'ɾ', 'ɽ̊', 'ɽ', 'ɡ̆', 'ɢ̆', 'ʡ̆',
    'ʙ̥', 'ʙ', 'r̥', 'r', 'ɽ̊r̥', 'ɽr', 'ʀ̥', 'ʀ', 'ʜ', 'ʢ',
    'tɬ', 'dɮ', 'tɭ̊˔', 'dɭ˔', 'cʎ̝̊', 'ɟʎ̝', 'kʟ̝̊', 'ɡʟ̝',
    'ɬ', 'ɮ', 'ꞎ', 'ɭ˔', '𝼆', 'ʎ̝', '𝼄', 'ʟ̝',
    'l', 'ɭ', 'ʎ', 'ʟ', 'ʟ̠',
    'ɺ̥', 'ɺ', '𝼈̥', '𝼈', 'ʎ̆', 'ʟ̆',
    # Non-pulmonic
    't̪θʼ', 'tsʼ', 't̠ʃʼ', 'tʂʼ', 'kxʼ', 'qχʼ',
    'ɸʼ', 'fʼ', 'θʼ', 'sʼ', 'ʃʼ', 'ʂʼ', 'ɕʼ', 'xʼ', 'χʼ',
    'tɬʼ', 'c𝼆ʼ', 'k𝼄ʼ',
    'ɬʼ',
    'kʘ', 'qʘ', 'kǀ', 'qǀ', 'kǃ', 'qǃ', 'k𝼊', 'q𝼊', 'kǂ', 'qǂ',
    'ɡʘ', 'ɢʘ', 'ɡǀ', 'ɢǀ', 'ɡǃ', 'ɢǃ', '', 'ɡ𝼊, ɢ𝼊', 'ɡǂ', 'ɢǂ',
    'ŋʘ', 'ɴʘ', 'ŋǀ', 'ɴǀ', 'ŋǃ', 'ɴǃ', 'ŋ𝼊', 'ɴ𝼊', 'ŋǂ', 'ɴǂ', 'ʞ',
    'kǁ', 'qǁ',
    'ɡǁ', 'ɢǁ',
    'ŋǁ', 'ɴǁ',
    'ɓ', 'ɗ', 'ᶑ', 'ʄ', 'ɠ', 'ʛ',
    'ɓ̥', 'ɗ̥', 'ᶑ̊', 'ʄ̊', 'ɠ̊', 'ʛ̥',
    # Co-articulated
    'n͡m', 'ŋ͡m',
    'ɥ̊', 'ɥ',
    'ʍ', 'w',
    'ɧ', 't͡p', 'd͡b', 'k͡p', 'ɡ͡b',
    'q͡ʡ', 'ɫ'
    ]

wiki_ipa_vowels = [
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø','ɘ', 'ɵ', 'ɤ', 'o',
    'e̞', 'ø̞', 'ə', 'ɤ̞', 'o̞',
    'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'ɛ',
    'ɐ', 'æ',
    'a', 'ɶ', 'ä', 'ɑ', 'ɒ'
]

wiki_dipthongs = [
    'eɪ', 'oʊ', 'aʊ', 'ɪə', 'eə', 'ɔɪ', 'aɪ', 'ʊə', 'dʒ'
]

wiki_ipa_markers = {
    'tones_placeholder_left': ['◌̋', '◌˥', '◌́', '◌˦', '◌̏', '◌˩', '◌̌'],
    'tones_placeholder_right': ['꜓◌', '꜒◌', '꜕◌', 'ꜜ◌', 'ꜛ◌', '꜖◌'],
    'aux_symbols_placeholder_left': ['◌̥', '◌̊', '◌̤', '◌̪', '◌͆', '◌̬', '◌̰',
                                    '◌̺', '◌ʰ', '◌̼', '◌̻', '◌̹', '◌͗', '◌˒',
                                    '◌ʷ', '◌̃', '◌̜', '◌͑', '◌˓', '◌ʲ', '◌ⁿ',
                                    '◌̟', '◌˖', '◌ˠ', '◌ˡ', '◌̠', '◌˗', '◌ˤ',
                                    '◌̚', '◌̈', '◌̴', '◌ᵊ', '◌̽', '◌˔', '◌ᶿ',
                                    '◌̩', '◌̍', '◌̞', '◌˕', '◌ˣ', '◌̯', '◌̑',
                                    '◌̘', '◌꭪', '◌ʼ', '◌˞', '◌̙', '◌꭫', '◌͡◌', '◌͜◌'],
    'suprasegmentals': ['ˈ', 'ˌ', 'ː', 'ˑ', '◌̆', '|', '‖', '.', '‿', '↗︎', '↘︎']
}

phonemizer_vowels = ['i', 'y', 'i"', 'i-', 'i"', 'u"', 'u-', 'u', 'ʉ'
                    'I', 'I.', 'U', 'e', 'Y', '@<umd>', 'o-', 'o',
                    '@', '@.',
                    'E', 'W', 'V"', 'O"', 'V', 'O',
                    '&',
                    'a', 'a.', 'A', 'A.']

phonemizer_punctuation = '“”\{\}-!\'"(),.:;? '

def load_symbols_from_file(filepath, cur_symbols):
    external_symbols = []
    with open(filepath, 'r') as fp:
        data = fp.readlines()
    fp.close()

    for line in data:
        symbol = line.rstrip()
        if symbol not in cur_symbols:
            external_symbols.append(line.rstrip())
    return external_symbols

def construct_cross_symbols(markers, placeholder_set, phonemizer_symbols):
    markers = list(set(markers))
    # print(len(markers))
    phonemizer_symbols = list(set(phonemizer_symbols))
    # print(len(phonemizer_symbols))

    symbols = []
    for ph_symbol in phonemizer_symbols:
        for marker in placeholder_set['left']:
            str_symbol = ph_symbol + marker
            symbols.append(str_symbol)

        for marker in placeholder_set['right']:
            str_symbol = marker + ph_symbol
            symbols.append(str_symbol)

    for marker in placeholder_set['other']:
        symbols.append(marker)

    symbols += phonemizer_symbols
    symbols = list(set(symbols))
    return symbols

def get_symbols(symbol_set, external_symbol_set_path=None):
    markers_with_placeholders = None
    markers = None
    dipthongs_set = None
    placeholder_set = None
    if symbol_set == 'english_basic':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _arpabet = ["@" + s for s in arpabet]
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_basic_lowercase':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        _arpabet = ["@" + s for s in arpabet]
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_expanded':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _arpabet = ["@" + s for s in arpabet]
        symbols = list(_punctuation + _math + _special + _accented + _letters) + _arpabet
    elif symbol_set == 'multilingual':
        _punctuation = '¡!\'\"",.:;¿?/¸ '
        _math = '#%&*+-/[]()<>·‹›-—'
        _special = '_@©°½₩€$£'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _accented_upper = "ÀÁÂÃÄÅĀĄĆÇÈÉÊẼËĘÌÍÎÏŁŃÑÒÓÔÕÖÙÚÛŨÜÆŒŚŹŻŽŸÝJ̃SSȎØᬠ"
        _accented_lower = "àáâãäåāąćçèéêẽëęìíîïłńñòóôõöùúûũüæœśźżžÿýj̃ßȏøᬠ"
        symbols = list(_punctuation + _math + _special + _letters +
                       _accented_upper + _accented_lower)
        symbols = sorted(list(set(symbols)))
    elif symbol_set == 'radtts':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _numbers = '0123456789'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _arpabet = ["@" + s for s in arpabet]
        symbols = list(_punctuation + _math + _special + _accented + _numbers + _letters) + _arpabet
    elif symbol_set == 'radmmm':
        _punctuation = '¡!\'\"",.:;¿?-/ '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented_upper = 'ÀÈÌÒÙÁÉÍÓÚĆÂÊÎÔÛÄËÏÖÜÃÕÑÆŒÇØŽÅŸÝ'
        _accented_lower = 'àèìòùáéíóúćâêîôûäëïöüãõñæœçøžåÿýj̃ũẽ'
        hi_accents = [u'\u0951', u'\u0952', u'\u0953', u'\u0954', u'\u0955']
        _extra= 'ß'
        # _extra_accents = ['']
        _numbers = '0123456789'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _ipa_dict = ["'", '(', ')', ',', '.', ':', '?', 'A', 'C', 'D', 'E',
                     'F', 'N', 'O', 'Q', 'R', 'S', 'T', 'U', 'Z', 'a', 'b',
                     'c', 'd',
                     'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                     'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|',
                     'ã', 'æ', 'ç', 'ð', 'õ', 'ø', 'ĭ', 'ŋ', 'œ', 'ɐ', 'ɑ',
                     'ɒ', 'ɔ', 'ɕ', 'ɘ', 'ə', 'ɛ', 'ɜ', 'ɝ', 'ɡ', 'ɣ', 'ɥ',
                     'ɪ', 'ɫ', 'ɬ', 'ɱ', 'ɲ', 'ɹ', 'ɽ', 'ɾ', 'ʀ', 'ʁ', 'ʃ',
                     'ʊ', 'ʋ', 'ʌ', 'ʎ', 'ʏ', 'ʒ', 'ʔ', 'ʝ', 'ʧ', 'ʰ', 'ʲ',
                     'ʼ', 'ˀ', 'ˈ', 'ˌ', 'ː', 'ˑ', '̃', '̆', '̍', '̥', '̩', '̯', '͜',
                     '͡', 'β', 'ε', 'θ', 'χ', 'ᵻ', 'ãː', 'ऑ', 'औ', 'ऍ']
        hi_punctuation = ['॥', '।', '//', '\/']
        hi_vowels = ['ə', 'a', 'aː', 'i', 'iː', 'u', 'uː', 'e', 'æː', 'o', 'ɔ', 'ɔː', 'r̩']
        hi_consonants = ['k', 'kʰ', 'ɡ', 'ɡ̤', 'ŋ', 't͡ʃ', 't͡ʃʰ', 'd͡ʒ', 'd͡ʒ̤', 'ɲ', 'ʈ', 'ʈʰ',
                        'ɖ', 'ɖ̤', 'ɳ', 't', 'tʰ', 'd', 'd̤', 'n', 'p', 'pʰ', 'b', 'b̤', 'm',
                        'j', 'r', 'l', 'v', 'ʃ', 'ʂ', 's', 'ɦ', 'q', 'x', 'ɣ', 'z', 'ʒ',
                        'f', 'ɽ', 'ɽ̤', "ɽ̥"]
        pt_symbols = ['ɐ̃', 'w̃', 'kʷ', 'ɡʷ', '-', 'ũː', 'ə̃', 'æ̃ː']
        _ipa = ["@" + s for s in ipa]
        _ipa_dict = ["@" + s for s in _ipa_dict]
        hi_vowels = ["@" + s for s in hi_vowels]
        hi_consonants = ["@" + s for s in hi_consonants]
        # hi_punctuation = ["@" + s for s in hi_punctuation]
        pt_symbols = ["@" + s for s in pt_symbols]
        symbols = list(_punctuation + _math + _special + _accented_lower +
                       _accented_upper + _extra + _numbers + _letters) + \
                       hi_vowels + hi_consonants + pt_symbols + hi_punctuation + hi_accents
        symbols += _ipa + _ipa_dict
        symbols = list(set(symbols))  # to account for repeated

        if external_symbol_set_path is not None:
            print(f'using external symbols from {external_symbol_set_path}')
            external_symbols = load_symbols_from_file(external_symbol_set_path, symbols)

            external_symbols = ["@" for es in external_symbols]
            symbols = external_symbols + symbols
            symbols = list(set(symbols))  # to account for repeated

        symbols = sorted(symbols)  # to guarantee fixed order
    # exhaustive symbol set, where markers are applied to every syllable
    # results in S*P symbols ~20k symbols.
    elif symbol_set == 'radmmm_phonemizer_exhaustive':
        placeholder_set = {
            'left': [],
            'right': [],
            'other': []
        }

        markers = []
        for marker_key, markers_list in phonemizer_markers.items():
            # placeholder preprocessing
            if 'placeholder_left' in marker_key:
                markers_list_updated = [m[1:] for m in markers_list]
                placeholder_set['left'] += [m[1:] for m in markers_list]
            elif 'placeholder_right' in marker_key:
                markers_list_updated = [m[0] for m in markers_list]
                placeholder_set['right'] += [m[0] for m in markers_list]
            else:
                markers_list_updated = markers_list
                placeholder_set['other'] += markers_list

            markers += markers_list_updated

        wiki_markers = []
        for marker_key, marker_list in wiki_ipa_markers.items():
            if 'placeholder_left' in marker_key:
                wiki_markers += [m[1:] for m in marker_list]
                placeholder_set['left'] += [m[1:] for m in marker_list]
            elif 'placeholder_right' in marker_key:
                wiki_markers += [m[0] for m in marker_list]
                placeholder_set['right'] += [m[0] for m in marker_list]
            else:
                wiki_markers += markers_list
                placeholder_set['other'] += markers_list

        phonemizer_symbols = wiki_ipa_consonants + \
                                wiki_ipa_vowels + \
                                    phonemizer_extra_symbols + \
                                        wiki_dipthongs + \
                                            list(wiki_special)

        markers += wiki_markers
        markers = list(set(markers))
        phonemizer_symbols = list(set(phonemizer_symbols))

        # parse all the syllables and fill dipthong list
        dipthongs_set = []
        for symbol in phonemizer_symbols:
            if len(symbol) > 1:
                dipthongs_set.append(symbol)
        dipthongs_set = list(set(dipthongs_set))

        phonemizer_symbols = construct_cross_symbols(markers,
                                                    placeholder_set,
                                                    phonemizer_symbols)

        phonemizer_symbols += list(phonemizer_punctuation) + \
                                list(wiki_numbers) + \
                                    list(wiki_math)

        phonemizer_symbols = ["@" + ipa for ipa in phonemizer_symbols]

        phonemizer_symbols += list(phonemizer_punctuation)
        symbols = sorted(list(set(phonemizer_symbols)))

    # segregated symbol set, where markers and syllables for separate symbols.
    # results in S+P symbols ~410 symbols.
    elif symbol_set == 'radmmm_phonemizer_marker_segregated':
        placeholder_set = {
            'left': [],
            'right': [],
            'other': []
        }

        markers = []
        for marker_key, markers_list in phonemizer_markers.items():
            # placeholder preprocessing
            if 'placeholder_left' in marker_key:
                markers_list_updated = [m[1:] for m in markers_list]
                placeholder_set['left'] += [m[1:] for m in markers_list]
            elif 'placeholder_right' in marker_key:
                markers_list_updated = [m[0] for m in markers_list]
                placeholder_set['right'] += [m[0] for m in markers_list]
            else:
                markers_list_updated = markers_list
                placeholder_set['other'] += markers_list

            markers += markers_list_updated

        wiki_markers = []
        for marker_key, marker_list in wiki_ipa_markers.items():
            if 'placeholder_left' in marker_key:
                wiki_markers += [m[1:] for m in marker_list]
                placeholder_set['left'] += [m[1:] for m in marker_list]
            elif 'placeholder_right' in marker_key:
                wiki_markers += [m[0] for m in marker_list]
                placeholder_set['right'] += [m[0] for m in marker_list]
            else:
                wiki_markers += markers_list
                placeholder_set['other'] += markers_list

        phonemizer_symbols = wiki_ipa_consonants + \
                                wiki_ipa_vowels + \
                                                phonemizer_extra_symbols + \
                                                    wiki_dipthongs + \
                                                        list(wiki_math) + \
                                                            list(wiki_special)

        markers += wiki_markers

        # parse all the syllables and fill dipthong list
        dipthongs_set = []
        for symbol in phonemizer_symbols:
            if len(symbol) > 1:
                dipthongs_set.append(symbol)
        dipthongs_set = list(set(dipthongs_set))

        phonemizer_symbols += markers

        phonemizer_symbols = ["@" + ipa for ipa in phonemizer_symbols]
        phonemizer_symbols += list(phonemizer_punctuation) + \
                                ["@" + punc for punc in list(phonemizer_punctuation)]
        symbols = sorted(list(set(phonemizer_symbols)))
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    # print("Number of symbols:", len(symbols))
    return symbols, markers, placeholder_set, dipthongs_set
