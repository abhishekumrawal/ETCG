# -*- coding: utf-8 -*-

import sys

sys.path.append('weightcover')

import plot

plot.plot_weightcover()

sys.path.append('nk_dependence')

import plot_n
import plot_k

plot_n.plot_lin_n()
plot_k.plot_lin_k()