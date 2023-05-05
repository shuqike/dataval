"""Test explorers"""
import sys
sys.path.append('../')
import src.utils as utils


explorer = utils.LinearExplorer()
for i in range(10):
    print('linear', explorer.exam(10, i))
