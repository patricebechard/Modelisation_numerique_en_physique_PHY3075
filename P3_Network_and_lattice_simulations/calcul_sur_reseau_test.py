# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:34:32 2017

PHY3075 - Modélisation numérique en physique

Chapitre 3 - Calcul sur réseau

@author: Patrice
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

#--------------------------Classes---------------------------

class Grid:
    """Grid onto which the simulation is run"""
    
    class Cell:
        """Content of one of the cell of the grid"""
        
        def __init__(self,status):
            self._status=status
    
    def __init__(self,dimensions,config):
        """Initialize the grid of a certain dimension and
        a certain initial configuration.
        
        config=0 : random initial config
        config=1 : everything is True
        config=-1 : everything is False
        else raise exception
        """
        