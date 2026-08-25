"""Sillage - gradient-free test-time memory and learning for frozen LMs.

A frozen language model reads your documents and gets better at them, with no
gradients, no fine-tuning and no growing index: a fixed Hebbian matrix written
as it reads, a semantic tier routed by confidence, a consolidating cold store,
and a rank-16 delta-rule adapter on the readout. Four mechanisms, four papers,
one object:

    from sillage import Sillage
    s = Sillage(model="gpt2")          # any causal LM: a shortcut
                                       # ("qwen", "gpt2"), a hub id or a path
    s.read("notes.md")                 # read, memorize, index -- then save
    s.ask("what did the report say?")  # exact passages, nothing generated
    s.complete("The protocol requires")

The readout (how strongly the memory speaks, and when it abstains) is fitted
on a rolling window of what you read whenever the model is one the papers did
not tune, so any frozen model works, not only the two they measured.

Command line: `sillage read notes.md`, `sillage ask "..."`, `sillage chat`.
"""

__version__ = "1.0.2"

from .core import MODELS, SillageMemory
from .index import Index, strip_latex
from .runtime import Sillage

__all__ = ["Sillage", "SillageMemory", "Index", "MODELS", "strip_latex",
           "__version__"]
