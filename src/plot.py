import os
import pandas as pd
import matplotlib.pyplot as plt


file_path = os.path.dirname(os.path.abspath(__file__))
csv = pd.read_csv(os.path.join(file_path, "../outputs/log.csv"))

csv["loss"].plot.line()

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()
