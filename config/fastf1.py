'''
Centralized wrapper for the FastF1 library. Re-exports FastF1's API, so you can do e.g.
`from config.fastf1 import get_event`. Always import FastF1 through this module to
ensure cache is enabled and configured correctly.
'''

import fastf1
from fastf1 import * # type: ignore[reportWildcardImportFromLibrary]
from config import config as _config

FASTF1_CACHE_DIR = _config.DATA_DIR / '.fastf1-cache'
FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))