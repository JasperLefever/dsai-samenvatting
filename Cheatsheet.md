```py
# Importing the necessary packages
import numpy as np                                  # "Scientific computing"
import scipy.stats as stats                         # Statistical tests

import pandas as pd                                 # Data Frame
from pandas.api.types import CategoricalDtype

import random
import math

import matplotlib.pyplot as plt                     # Basic visualisation
from statsmodels.graphics.mosaicplot import mosaic  # Mosaic diagram
import seaborn as sns                                # Advanced data visualisation
from sklearn.linear_model import LinearRegression
import altair as alt                                # Alternative visualisation system
```

# [H1 - Samples](./Pythat0n/H1.md)

---

# [H2 - Analyse van 1 variabele](./Pythat0n/H2.md)

# [H3](./Pythat0n/H3.md)

---

# [H4 -> 2 kwalitatieve variabelen](./Pythat0n/H4.md)

---

# [H5 -> 1 kalitatieve variabele en 1 kwantitatieve variabelen](./Pythat0n/H5.md)

---

# [H6 -> 2 kwantitatieve variabelen](./Pythat0n/H6.md)

---

# [H7 -> time series](./Pythat0n/H7.md)

- moving averages
  - simple moving average
  - weighted moving average
    - exponential moving average
- exponential smoothing
  - single exponential smoothing -> exponential smoothing
    - geen trend of seasonality
  - double exponential smoothing -> Holt's method
    - trend
  - triple exponential smoothing -> Holt-Winters method
    - trend en seasonality
