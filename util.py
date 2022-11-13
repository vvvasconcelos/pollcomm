#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 10/06/2022
# ---------------------------------------------------------------------------
""" util.py

Some useful functions
"""
# ---------------------------------------------------------------------------
def dA_rate_max(t, rate, dA_max):
    """Drivers of decline: dA(t) = rate * t, with a maximum of dA_max"""
    return min(rate * t, dA_max)
