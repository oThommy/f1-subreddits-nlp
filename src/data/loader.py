from collections.abc import Set as ImmutableSet
from collections.abc import Generator, Callable
from typing import Any, TypeAlias
import json
import pandas as pd
from pathlib import Path
import warnings
from src.data import constants

def stream_ndjson(ndjson_file: Path, limit: int | None = None) -> Generator[dict[str, Any]]:
    '''Stream NDJSON file line by line, parsing each line to a JSON object.

    :param limit: Maximum number of lines to stream. If None, stream all lines. Raises a UserWarning if limit is <= 0.
    :yield: Parsed JSON object from each line.
    '''
    if limit is not None and limit <= 0:
        warnings.warn(f'Expected `limit` >= 1, got {limit}. No lines will be streamed.', UserWarning)
        return
    
    with open(ndjson_file, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            if limit is not None and index >= limit:
                break
            yield json.loads(line)

NdjsonStreamer: TypeAlias = Callable[[Path], Generator[dict[str, Any]]]

def load_submissions_df(
    ndjson_file: Path,
    ndjson_streamer: NdjsonStreamer = stream_ndjson,
    columns: ImmutableSet[str] = frozenset({'author', 'created_utc', 'gilded', 'id', 'score', 'selftext', 'title'}),
) -> pd.DataFrame:
    '''
    :param columns: The desired columns to load into the dataframe.
    :param ndjson_streamer: Data generator that streams parsed JSON objects from an ndjson file.
    :raises ValueError: If any of the specified columns are not in `src.data.constants.SUBMISSION_COLUMNS`.
    '''
    # TODO: should this be a warning?
    if not columns <= constants.SUBMISSION_COLUMNS:
        raise ValueError(
            f'Got unknown submission column(s): {columns - constants.SUBMISSION_COLUMNS}. '
            'If the column does exist in the data, add it to src.data.constants.SUBMISSION_COLUMN_DTYPES.'
        )
    
    dtypes = {
        column: dtype
        for column, dtype in constants.SUBMISSION_COLUMN_DTYPES.items()
        if column in columns
    }
    
    return pd.DataFrame(
        (
            {column: row.get(column) for column in columns}
            for row in ndjson_streamer(ndjson_file)
        ),
    ).astype(dtypes) # TODO: this should not be enforced here?

def load_comments_df(
    ndjson_file: Path,
    ndjson_streamer: NdjsonStreamer = stream_ndjson,
    columns: ImmutableSet[str] = constants.DEFAULT_COMMENT_COLUMNS,
) -> pd.DataFrame:
    '''
    :param columns: The desired columns to load into the dataframe.
    :param ndjson_streamer: Data generator that streams parsed JSON objects from an ndjson file.
    :raises ValueError: If any of the specified columns are not in `src.data.constants.COMMENT_COLUMNS`.
    '''
    if not columns <= constants.COMMENT_COLUMNS:
        raise ValueError(
            f'Got unknown comment column(s): {columns - constants.COMMENT_COLUMNS}. '
            'If the column does exist in the data, add it to src.data.constants.COMMENT_COLUMN_DTYPES.'
        )
    
    dtypes = {
        column: dtype
        for column, dtype in constants.COMMENT_COLUMN_DTYPES.items()
        if column in columns
    }
    
    return pd.DataFrame(
        (
            {column: row.get(column) for column in columns}
            for row in ndjson_streamer(ndjson_file)
        ),
    ).astype(dtypes)
