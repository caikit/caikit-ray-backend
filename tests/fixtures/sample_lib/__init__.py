# Standard
import os

# First Party
from caikit.config import configure

# Local
from . import data_model, modules

# Run configure for sample_lib configuration
configure(os.path.join(os.path.dirname(__file__), "config.yml"))
