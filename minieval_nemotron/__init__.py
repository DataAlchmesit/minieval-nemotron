"""MiniEval x Nemotron: a verification cascade around minieval-pro's local gate.

Local checks decide the clear cases for free. Hard cases go to one Nemotron
call on Nebius Token Factory. A model's answer only counts if the quote it
cites is really in the source. Anything that fails goes to REVIEW, never STORE.
"""

from .cascade import decide
from .policy import CascadePolicy
from .types import Candidate, Decision, LocalResult

__all__ = ["decide", "CascadePolicy", "Candidate", "Decision", "LocalResult"]
