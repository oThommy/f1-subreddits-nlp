from functools import cache
import pandas as pd
import numpy as np
import re
import urllib.request
import sys
import spacy
from symspellpy import SymSpell
import symspellpy
from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance
from config import config
from src.utils import assert_columns_exist

_ALPHANUMERIC_PATTERN = re.compile(r'\w')

def concatenate_submissions_and_comments(
    submissions_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    in_place: bool = False,
) -> pd.DataFrame:
    '''Concatenate submissions and comments into a single DataFrame for NLP analysis.

    The submission columns `title` and `selftext` are combined into a unified `text`.

    :param submissions_df: DataFrame containing at least the `title` and `selftext` columns.
    :param comments_df: DataFrame containing at least the `body` column.
    :param in_place: If true, modify the input DataFrames in place.
    :raises ValueError: If any of the required columns are missing from the DataFrame.
    :return: A combined DataFrame with a unified `text` column.
    '''
    assert_columns_exist({'title', 'selftext'}, submissions_df, 'submissions')
    assert_columns_exist({'body'}, comments_df, 'comments')

    _submissions_df = submissions_df if in_place else submissions_df.copy()
    _comments_df = comments_df if in_place else comments_df.copy()

    titles: pd.Series[str] = _submissions_df['title'].str.rstrip()
    selftexts: pd.Series[str] = _submissions_df['selftext']

    # TODO: still a bit buggy: title='title', selftext='' -> text='title. ' with trailing space
    _submissions_df['text'] = np.where(
        titles.str[-1].map(lambda ch: _ALPHANUMERIC_PATTERN.match(ch) is not None),
        titles + '. ' + selftexts,
        titles + ' ' + selftexts,
    )
    _submissions_df.drop(columns=['title', 'selftext'], inplace=True)

    _comments_df = comments_df.copy()
    _comments_df.rename(columns={'body': 'text'}, inplace=True)

    df = pd.concat((_submissions_df, _comments_df), ignore_index=True)  
    return df

# TODO: Refactor
F1_names= {
    'max verstappen',
    'charles leclerc',
    'sergio perez',
    'george russell',
    'carlos sainz',
    'lewis hamilton',
    'lando norris',
    'esteban ocon',
    'fernando alonso',
    'valtteri bottas',
    'daniel ricciardo',
    'sebastian vettel',
    'kevin magnussen',
    'pierre gasly',
    'lance stroll',
    'mick schumacher',
    'yuki tsunoda',
    'zhou guanyu',
    'alexander albon',
    'nicholas latifi',
    'nyck de vries',
    'nico hulkenberg',
    'oscar piastri',
    'liam lawson',
    'logan sargeant'
}

F1_DRIVERS = {
    'max', 'verstappen',
    'charles', 'leclerc',
    'sergio', 'perez',
    'george', 'russell',
    'carlos', 'sainz',
    'lewis', 'hamilton',
    'lando', 'norris',
    'esteban', 'ocon',
    'fernando', 'alonso',
    'valtteri', 'bottas',
    'daniel', 'ricciardo',
    'sebastian', 'vettel',
    'kevin', 'magnussen',
    'pierre', 'gasly',
    'lance', 'stroll',
    'mick', 'schumacher',
    'yuki', 'tsunoda',
    'zhou', 'guanyu',
    'alexander', 'albon',
    'nicholas', 'latifi',
    'nyck', 'vries',
    'nico', 'hulkenberg',
    'oscar', 'piastri',
    'liam', 'lawson',
    'logan', 'sargeant'
}

Drivers_dict = {
    'max': 'max verstappen',
    'charles': 'charles leclerc',
    'sergio': 'sergio perez',
    'george': 'george russell',
    'carlos': 'carlos sainz',
    'lewis': 'lewis hamilton',
    'lando': 'lando norris',
    'esteban': 'esteban ocon',
    'fernando': 'fernando alonso',
    'valtteri': 'valtteri bottas',
    'daniel': 'daniel ricciardo',
    'sebastian': 'sebastian vettel',
    'kevin': 'kevin magnussen',
    'pierre': 'pierre gasly',
    'lance': 'lance stroll',
    'mick': 'mick schumacher',
    'yuki': 'yuki tsunoda',
    'zhou': 'zhou guanyu',
    'alexander': 'alexander albon',
    'nicholas': 'nicholas latifi',
    'nyck': 'nyck de vries',
    'nico': 'nico hulkenberg',
    'oscar': 'oscar piastri',
    'liam': 'liam lawson',
    'logan': 'logan sargeant',
    
    'verstappen': 'max verstappen',
    'leclerc': 'charles leclerc',
    'perez': 'sergio perez',
    'russell': 'george russell',
    'sainz': 'carlos sainz',
    'hamilton': 'lewis hamilton',
    'norris': 'lando norris',
    'ocon': 'esteban ocon',
    'alonso': 'fernando alonso',
    'bottas': 'valtteri bottas',
    'ricciardo': 'daniel ricciardo',
    'vettel': 'sebastian vettel',
    'magnussen': 'kevin magnussen',
    'gasly': 'pierre gasly',
    'stroll': 'lance stroll',
    'schumacher': 'mick schumacher',
    'tsunoda': 'yuki tsunoda',
    'guanyu': 'zhou guanyu',
    'albon': 'alexander albon',
    'latifi': 'nicholas latifi',
    'vries': 'nyck de vries',
    'hulkenberg': 'nico hulkenberg',
    'piastri': 'oscar piastri',
    'lawson': 'liam lawson',
    'sargeant': 'logan sargeant',
}

F1_VOCABULARY = F1_DRIVERS

def correct_spelling(word):
    if word in F1_VOCABULARY:
        return word
    
    min_distance = float('inf')
    corrected_word = word
    for term in F1_VOCABULARY:
        distance = edit_distance(word, term)
        if distance < min_distance and distance <= max(1, len(word)//3):  # Allow a maximum edit distance of 33%
            min_distance = distance
            corrected_word = term
    # print(word, corrected_word, min_distance)
    return corrected_word

def download_file(path, url):
    if not path.exists():
        try:
            print('INFO: downloading english word dictionary...')
            urllib.request.urlretrieve(url, path)
            print('downloading complete!!! :)')
        except Exception as error:
            raise Exception(f'Download failed: {error}')

SYM_SPELL_MAX_DICTIONARY_EDIT_DISTANCE = 4

@cache
def load_sym_spell():
    sym_spell = SymSpell(max_dictionary_edit_distance=SYM_SPELL_MAX_DICTIONARY_EDIT_DISTANCE, prefix_length=7)

    # english_words_dictionary_file = config.DATA_DIR / 'english_words_dictionary.txt'
    # download_file(english_words_dictionary_file, 'https://raw.githubusercontent.com/wolfgarbe/SymSpell/refs/heads/master/SymSpell/frequency_bigramdictionary_en_243_342.txt')

    english_words_dictionary_file = config.DATA_DIR / 'english_words_dictionary.txt'
    download_file(english_words_dictionary_file, 'https://raw.githubusercontent.com/wolfgarbe/SymSpell/refs/heads/master/SymSpell/frequency_dictionary_en_82_765.txt')

    with open(english_words_dictionary_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, frequency = line.strip().split()
            frequency = int(frequency)
            sym_spell.create_dictionary_entry(word, frequency)

    for word in F1_VOCABULARY:
        sym_spell.create_dictionary_entry(word, sys.maxsize)

    return sym_spell

def correct_spelling_symspell(word):
    sym_spell = load_sym_spell()
    suggestions = sym_spell.lookup(word, symspellpy.Verbosity.CLOSEST, max_edit_distance=3)
    return suggestions[0].term if suggestions else word

@cache
def load_nlp():
    return spacy.load('en_core_web_sm')

def correct_spelling_in_text_spacy(text):
    nlp = load_nlp()
    doc = nlp(text)
    corrected_tokens = [
        # correct_spelling_symspell(token.text) if token.ent_type_ == 'PERSON' else token.text
        correct_spelling_symspell(token.text) if token.is_alpha else token.text
        for token in doc
    ]

    corrected_tokens = combine_names(corrected_tokens)
    try:
        return ''.join([
            corrected_tokens[i] + (token.whitespace_ if token.whitespace_ else '')
            for i, token in enumerate(doc)
        ])
    except IndexError:
        print([token.text for token in doc])
        print(corrected_tokens)

def combine_names(tokens):
    combined_tokens = []
    skip_next = False

    for i, word in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        # Check if the current word and the next word form a driver name
        if i + 1 < len(tokens):
            combined_word = f'{word} {tokens[i + 1]}'
            if combined_word in F1_names:
                combined_tokens.append(word)
                combined_tokens.append(tokens[i + 1])
                skip_next = True
                continue

        # Check if the current word alone matches a driver name
        if word in F1_DRIVERS:
            combined_tokens.append(Drivers_dict[word])
        else:
            combined_tokens.append(word)

    return combined_tokens

def normalize(comment):
    comment = comment.lower()
    comment = re.sub(r'http\S+', '', comment)
    comment = re.sub(r'[^a-z\s\d]', '', comment)
    return comment

def remove_stopword(tokens, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    new_tokens = []
    for word in tokens:
        if word not in stop_words:
            new_tokens.append(word)
    return new_tokens
