"""
whatsapp_analyzer — public API.
"""

from whatsapp_analyzer.comparator import GroupComparator
from whatsapp_analyzer.core import WhatsAppAnalyzer

# Specify what we should import doing from whatsapp_analyzer import *
__all__ = ["WhatsAppAnalyzer", "GroupComparator"]
