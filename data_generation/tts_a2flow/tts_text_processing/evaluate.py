import random
from .text_processing import TextProcessing


def main(filepath, index, symbol_set, cleaner_names, heteronyms_path,
         cmu_dict_path, p_arpabet=0.5, handle_arpabet='',
         handle_arpabet_ambiguous='', shuffle=False, interactive=False):

    tp = TextProcessing(
        symbol_set, cleaner_names, heteronyms_path, cmu_dict_path,
        p_arpabet=p_arpabet, handle_arpabet=handle_arpabet,
        handle_arpabet_ambiguous=handle_arpabet_ambiguous)

    with open(filepath, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    if shuffle:
        random.shuffle(filepaths_and_text)

    for i in range(index, len(filepaths_and_text)):
        filepath, text = filepaths_and_text[i][0], filepaths_and_text[i][1]
        print("INDEX\t", i)
        print("FILEPATH\t", filepath)
        print("INPUT\t", text)
        text_encoded, text_clean, text_arpabet = tp.encode_text(text, return_all=True)
        print("CLEAN\t", text_clean)
        print("ARPABET\t", text_arpabet)
        print("ENCODED\t", text_encoded)
        if interactive:
            input("Press return for next sample...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # python -m tts_text_processing.evaluate
    parser.add_argument('-f', '--filepath', required=True)
    parser.add_argument('-i', '--index', type=int, default=0)
    parser.add_argument('-s', '--symbol_set', default='english_basic')
    parser.add_argument('-c', '--cleaner_names', nargs='+', default=['english_cleaners'])
    parser.add_argument('--heteronyms_path', default='tts_text_processing/heteronyms', type=str)
    parser.add_argument('--cmu_dict_path', default='tts_text_processing/cmudict-0.7b_nv22.01', type=str)
    parser.add_argument('--p_arpabet', default=0.0, type=float)
    parser.add_argument('--handle_arpabet', default='word', type=str)
    parser.add_argument('--handle_arpabet_ambiguous', default='ignore', type=str)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--interactive', action='store_true')

    args = parser.parse_args()
    print(args)
    main(args.filepath, args.index, args.symbol_set, args.cleaner_names,
         args.heteronyms_path, args.cmu_dict_path, args.p_arpabet,
         args.handle_arpabet, args.handle_arpabet_ambiguous, args.shuffle,
         args.interactive)
