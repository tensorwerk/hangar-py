from .selection import BACKEND_ACCESSOR_MAP
from .selection import backend_decoder
from .selection import backend_from_heuristics
from .selection import is_local_backend
from .selection import backend_opts_from_heuristics

__all__ = [
    'BACKEND_ACCESSOR_MAP', 'backend_decoder', 'backend_from_heuristics',
    'is_local_backend', 'backend_opts_from_heuristics'
]
