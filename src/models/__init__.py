import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


from .hred import HRED
from .vhcr import VHCR
from .vhred import VHRED
from .ae_mapper import AEMapper
from .seq2seq import Seq2Seq


VariationalModels = ['VHRED', 'VHCR']
