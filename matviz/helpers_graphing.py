# Backward compatibility - use matviz.helpers instead
from matviz.helpers import *

# Re-apply matplotlib defaults for backward compatibility when this module is reloaded
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12, 9]
plt.rcParams['image.cmap'] = 'viridis'
