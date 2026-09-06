"""
Post-Processing Modules for Text Detection.
"""

from .builder import POSTPROCESSORS, build_postprocessor
from .db_postprocessor import DBPostprocessor, DBPostProcessor

__all__ = ["POSTPROCESSORS", "build_postprocessor", "DBPostprocessor", "DBPostProcessor"]
