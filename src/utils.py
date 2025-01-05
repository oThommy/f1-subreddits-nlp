import pandas as pd
from pandas.io.formats.style import Styler
from contextlib import contextmanager
from functools import partial, reduce
from config import config
import torch
import random
import numpy as np

# TODO: improve: do you want try/finally here? type annotations, etc.
@contextmanager
def temporary_pandas_options(options):
    old_values = {key: pd.get_option(key) for key in options}

    try:
        for key, value in options.items():
            pd.set_option(key, value)
        yield
    finally:
        for key, value in old_values.items():
            pd.set_option(key, value)

display_full_dataframe = partial(temporary_pandas_options, options={
    'display.max_columns': None,
    'display.max_colwidth': None,
    'display.max_rows': None,
})

def hide_index(df: pd.DataFrame, /) -> Styler:
    return df.style.hide(axis='index')

def _compose_two_functions(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def compose(*functions):
    return reduce(_compose_two_functions, functions)

def set_random_seeds(seed: int = config.RANDOM_SEED) -> None:
    '''Set random seeds for reproducibility across random, numpy.random, and torch.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TODO: Unsure what this does, but it is used in spacy.util.fix_random_seed, so it might be worth using
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

