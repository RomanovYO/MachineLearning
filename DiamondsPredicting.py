import numpy as np
import pandas as pd
import seaborn as anb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.ensemble        import RandomForestRegressor

df = pd.read_csv('P1_diamonds.csv')

print(df.head(10).to_string())
