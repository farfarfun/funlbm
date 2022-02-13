import tecplot as tp
from tecplot.constant import *


def new_layout_connect(port=7600):
    tp.session.connect(port=port)
    tp.new_layout()
