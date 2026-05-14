"""Public package API for SpaNECT."""

from .SpaNECT import SpaNECT
from . import characterization
from . import gr
from . import pp
from . import tl
from . import pl
from . import emb
from . import datasets
from . import characterization as ch

# Attach modular capabilities to the pipeline facade class
# so users can write: `from spanect import SpaNECT as sn; sn.gr ...`.
SpaNECT.gr = gr
SpaNECT.pp = pp
SpaNECT.tl = tl
SpaNECT.pl = pl
SpaNECT.emb = emb
SpaNECT.datasets = datasets
SpaNECT.ch = ch

__all__ = [
    "SpaNECT",
    "characterization",
    "ch",
    "gr",
    "pp",
    "tl",
    "pl",
    "emb",
    "datasets",
]
