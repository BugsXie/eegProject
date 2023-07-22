#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import xml.etree.ElementTree
from xml.etree.ElementTree import parse

raw = mne.read_cov