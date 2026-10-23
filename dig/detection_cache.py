"""Small content-addressed caches for repeated, deterministic image detection."""
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import pickle
from threading import RLock

import numpy as np


def cached_detection(function):
    signature = inspect.signature(function)
    cache = OrderedDict()
    lock = RLock()

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        digest = hashlib.sha256()
        for name, value in bound.arguments.items():
            if isinstance(value, np.ndarray):
                digest.update(pickle.dumps((name, value.shape, value.dtype.str)))
                digest.update(value.tobytes())
            else:
                digest.update(pickle.dumps((name, value)))
        key = digest.digest()
        with lock:
            if key in cache:
                cache.move_to_end(key)
                return deepcopy(cache[key])
        result = function(*args, **kwargs)
        with lock:
            cache[key] = deepcopy(result)
            cache.move_to_end(key)
            while len(cache) > 16:
                cache.popitem(last=False)
        return result

    def clear():
        with lock:
            cache.clear()
    wrapped.cache_clear = clear
    return wrapped
