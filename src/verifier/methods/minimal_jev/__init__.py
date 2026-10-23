"""Method 2, Jev variant: one judge-model request per answer behind the shared harness.

run_single_call is minimal_jev (the whole answer against the whole source); run_minimal_jev
serves minimal_jev_seg and the ablations built on it."""

from verifier.methods.minimal_jev.single import run_single_call
from verifier.methods.minimal_jev.verifier import run_minimal_jev

__all__ = ["run_minimal_jev", "run_single_call"]
