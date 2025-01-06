import pandas as pd
from collections.abc import Set as ImmutableSet
from pandas.io.formats.style import Styler
from contextlib import contextmanager
from functools import partial, reduce
from typing import TypeVar
from config import config
import torch
import torch.version
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

_T = TypeVar('_T')
_U = TypeVar('_U')

def infer_types(dct: dict[_T, _U], /) -> dict[_T, _U]:
    return dct

def set_random_seeds(seed: int = config.RANDOM_SEED) -> None:
    '''Set random seeds for reproducibility across random, numpy.random, and torch.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TODO: Unsure what this does, but it is used in spacy.util.fix_random_seed, so it might be worth using
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def assert_columns_exist(
    required_columns: ImmutableSet[str],
    df: pd.DataFrame,
    dataframe_name: str | None = None,
) -> None:
    '''
    :param dataframe_name: e.g. 'submissions' -> 'Submissions DataFrame', default is 'The DataFrame'.
    :raises ValueError: If any of the required columns are missing from the DataFrame.
    '''
    missing_columns = required_columns - set(df.columns)
    name = f'{dataframe_name.capitalize()} DataFrame' if dataframe_name is not None else 'The DataFrame'

    if missing_columns:
        raise ValueError(
            f'{name} must contain columns: {required_columns}, '
            f'missing: {missing_columns}.'
        )

def get_device() -> torch.device:
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        device_id = 0  # Default to the first GPU
        device_name = torch.cuda.get_device_name(device_id)
        print(f'CUDA version: {torch.version.cuda}')
        print(f'Selected GPU: {device_name} (device_id={device_id})')
        return torch.device('cuda')
    else:
        print('No GPU available, using CPU')
        return torch.device('cpu')
