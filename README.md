# NASA-Kepler-Exoplanet-Detector-Data-Analysis-Machine-Learning
En base a un dataset proporcionado por el telescopio Kepler de la NASA, analizo los datos recogidos para entrenar un modelo de Machine Learning con Random Forest con el objetivo de clasificar candidatos a exoplaneta como confirmados o falsos positivos.



# 1.Descargar el dataset .csv 

*Dataset: NASA Exoplanet Archive, operado por el California Institute of Technology bajo contrato con la NASA como parte del Exoplanet Exploration Program.*
*exoplanetarchive.ipac.caltech.edu*

# 2.Importamos las librerias necesarias:
  Numpy, Pandas, Matplotlib.pyplot, RandomForest y lightkurve.
  Lightkurve es una librería específica de la NASA para descargar y analizar curvas de luz (el dato clave) de los telescopios Kepler y TESS, 
  en base a las curvas de luz podremos calcular la posición y el movimiento de los planetas/estrellas.

```python
!pip install lightkurve numpy pandas matplotlib scikit-learn seaborn
import lightkurve as lk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
```

# 3. Visualizamos un primer caso verificado
  Busca y descarga (conectandose a los servidores de la NASA mediante API) la curva de luz real de un planeta muy parecido a la tierra (Kepler-452) un exoplaneta ya confirmado.
  El resultado es la curva de luz del planeta, simplemente el recorrido de la estela de luz dejada por el planeta a lo largo del tiempo. 
  
```python
search_result = lk.search_lightcurve("Kepler-452", mission="Kepler")
lc = search_result[0].download()
lc.plot()
```

# 4. Limpiamos los datos 
  Eliminamos los valores nulos que puedan haber.
```python
lc_clean = lc.remove_nans().normalize()
```

# 5. Confirmación del primer caso
  Con .fold, superponemos todos los períodos orbitales del planeta unos sobre otros, usando su período orbital conocido según los datos oficiales de la NASA.
  De esta manera se verán todos los tránsitos apilados en un solo pico, eliminando con el .bin posterior, el ruido visual.
  Es solo una parte demostrativa de que el proceso y el método que usaremos funciona en casos conocidos.
```python
lc_folded = lc_clean.fold(period=384.84)
lc_binned = lc_folded.bin(time_bin_size=0.05)
lc_binned.plot()
```

# 6. Cargamos el dataset real y visualizamos primeros datos de reconocimiento 

  Aquí tendriamos el conjunto completo de datos de candidatos a exoplaneta de la misión Kepler, con miles de filas donde cada fila es un candidato.
  La columna koi_disposition es la etiqueta real de la NASA que indica si el candidato ha sido confirmado, falso positivo o candidato (todavía sin confirmar).
  
```python
  df = pd.read_csv("kepler_data.csv", comment='#')
print(df.shape)
print(df['koi_disposition'].value_counts())
```

<img width="1720" height="323" alt="image" src="https://github.com/user-attachments/assets/e1a09f92-37fa-4fc8-a349-d3afd130c5d3" />
