"""Legacy entry point backed by the one semcs_core.Engine."""
from semcs_core import LegacyFacade
from .method import ReliabilityMethod

class SeMCS(LegacyFacade):
    default_verbose=False
    def __init__(self, Sampler):
        super().__init__(Sampler, ReliabilityMethod())
