# NASA-Kepler-Exoplanet-Detector-Data-Analysis-Machine-Learning
En base a un dataset proporcionado por el telescopio Kepler de la NASA, analizo los datos recogidos para entrenar un modelo de Machine Learning con Random Forest con el objetivo de clasificar candidatos a exoplaneta como confirmados o falsos positivos.

*Dataset: NASA Exoplanet Archive, operado por el California Institute of Technology bajo contrato con la NASA como parte del Exoplanet Exploration Program.*
*exoplanetarchive.ipac.caltech.edu*

# 1.Descargar el dataset .csv 

# 2.Importamos las librerias necesarias:
  Numpy, Pandas, Matplotlib.pyplot, RandomForest y lightkurve.
  Lightkurve es una librería específica de la NASA para descargar y analizar curvas de luz de los telescopios Kepler y TESS, 
  en base a las curvas de luz podremos calcular la posición y el movimiento de los planetas/estrellas.

```python
!pip install lightkurve numpy pandas matplotlib scikit-learn seaborn
import lightkurve as lk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
```


