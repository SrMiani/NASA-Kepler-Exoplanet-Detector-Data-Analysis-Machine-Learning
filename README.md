

<img width="739" height="415" alt="images" src="https://github.com/user-attachments/assets/0df27f5a-f614-403c-bba4-bccd9b0389af" />


# NASA-Kepler-Exoplanet-Detector-Data-Analysis-Machine-Learning
En base a un dataset proporcionado por el telescopio Kepler de la NASA, analizo los datos recogidos para entrenar un modelo de Machine Learning con Random Forest y Gradient Boosting con el objetivo de clasificar candidatos a exoplaneta como confirmados o falsos positivos.



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
  Con comment='#' ignoramos todas las líneas que empiecen por ese símbolo en el .read_csv.
  
```python
  df = pd.read_csv("kepler_data.csv", comment='#')
print(df.shape)
print(df['koi_disposition'].value_counts())
```

<img width="1720" height="323" alt="image" src="https://github.com/user-attachments/assets/e1a09f92-37fa-4fc8-a349-d3afd130c5d3" />


# 7. Preparamos los datos sobre los que trabajaremos (Supervised Learning)

  Nos quedaremos con los datos que YA están clasificados por la NASA, los confirmados y los falsos positivos, esos casos serán el entrenamiento del modelo.
  Guardamos dentro de una variable nueva (label) los datos : 
  1 si el planeta es confirmado
  0 si es falso positivo
  
```python
df_clean = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
df_clean['label'] = (df_clean['koi_disposition'] == 'CONFIRMED').astype(int)

print(df_clean['label'].value_counts())
print(f"Shape: {df_clean.shape}")
```

<img width="469" height="141" alt="image" src="https://github.com/user-attachments/assets/d25c679b-4fe7-4fbc-a9c7-6625bf8c8db0" />


# 8. Selección de features para el modelo.

  Básicamente en este paso lo que hacemos es seleccionar las 'características' que queremos que el modelo use para aprender a distinguir un planeta real de uno falso.
  


```python
features = [
    'koi_period',        # periodo orbital
    'koi_depth',         # profundidad del tránsito
    'koi_duration',      # duración del tránsito
    'koi_prad',          # radio del planeta
    'koi_teq',           # temperatura de equilibrio
    'koi_insol',         # flujo de insolación
    'koi_model_snr',     # ratio señal/ruido
    'koi_steff',         # temperatura de la estrella
    'koi_slogg',         # gravedad superficial
    'koi_srad'           # radio de la estrella
]

df_model = df_clean[features + ['label']].dropna()
print(f"Shape final: {df_model.shape}")

```


# 9. Preparación datos de entrenamiento

  Dividimos los datos de entrenamiento en un 80% oara entrenar al modelo y un 20% para evaluarlo con datos que nunca ha visto.
  Con StandardScaler() lo que hacemos es normalizar a una escala común todas las variables.

  ```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_model[features]
y = df_model['label']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

  ```

# 10. Entrenamiento del modelo con Random Forest

  Entrenamos un modelo mediante Random Forest, un modelo que combina hasta 100 árboles de decisión (n_estimators=100).
  

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred, 
      target_names=['False Positive', 'Confirmed Planet']))
```

con Random Forest podemos visualizar incluso que variables se han utilizado más.

```python
importances = model.feature_importances_
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")
```

```python

importances = pd.Series(
    model.feature_importances_, 
    index=features
).sort_values(ascending=False)

importances.plot(kind='bar')
plt.title('Importancia de cada característica')
plt.tight_layout()
plt.show()
```


<img width="761" height="561" alt="image" src="https://github.com/user-attachments/assets/41a2358e-efcd-4a15-b096-787bd446949e" />


Si queremos visualizar los resultados con aciertos y errores del modelo, podemos utilizar un heatmap de seaborn


```python
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['False Positive', 'Planet'],
            yticklabels=['False Positive', 'Planet'])
plt.title('Matriz de confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.show()

```

<img width="664" height="551" alt="image" src="https://github.com/user-attachments/assets/1d9322e0-0b83-4a04-b57e-dbe7317dac81" />



# 11. Segundo modelo para comparar (Gradient Boosting)

  Gradient Boosting construye árboles de manera secuencial, es deir, cada uno corrige los errores del anterior, en vez de hacerlo en paralelo como 
  Random Forest, es simplemente una manera de comprobar cual de los dos modelos funciona mejor.

```python

from sklearn.ensemble import GradientBoostingClassifier

model_gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

model_gb.fit(X_train_scaled, y_train)
y_pred_gb = model_gb.predict(X_test_scaled)

print(classification_report(y_test, y_pred_gb,
      target_names=['False Positive', 'Confirmed Planet']))

```

```python

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train_scaled, y_train, 
                          cv=5, scoring='recall')
print(f"Recall medio: {scores.mean():.3f} (+/- {scores.std():.3f})")

```

En lugar de evaluar el modelo 1 vez, lo entrenamos y evaluamos 5 veces con diferentes reparticiones de datos cv=5.


# 12. Resultado final

  Con los candidatos no evaluados por la NASA, los 'CANDIDATE' del dataset original, utilizamos el modelo ya validado y entrenado.
  Predict_proba es una variable que guarda la probabilidad de que cada uno sea un planeta real ordenado de mayor a menor probabilidad.




```python
candidates = df[df['koi_disposition'] == 'CANDIDATE'][features].dropna()
candidates_scaled = scaler.transform(candidates)

predictions = model.predict(candidates_scaled)
probabilities = model.predict_proba(candidates_scaled)[:, 1]

candidates_results = candidates.copy()
candidates_results['prediction'] = predictions
candidates_results['probability'] = probabilities

top_candidates = candidates_results.sort_values(
    'probability', ascending=False
).head(20)

print("Top 10 candidatos más prometedores:")
print(top_candidates[['koi_period', 'koi_prad', 'probability']])
```

<img width="383" height="462" alt="image" src="https://github.com/user-attachments/assets/baba86cb-9c14-4810-8f6b-2626f3da1033" />


```python

print(f"Candidatos con prob > 0.9: {(candidates_results['probability'] > 0.9).sum()}")
print(f"Candidatos con prob > 0.8: {(candidates_results['probability'] > 0.8).sum()}")
print(f"Candidatos con prob > 0.5: {(candidates_results['probability'] > 0.5).sum()}")


```
<img width="304" height="67" alt="image" src="https://github.com/user-attachments/assets/8cf49ca9-a4c4-4ed0-9a89-0e809c547324" />

<img width="556" height="187" alt="image" src="https://github.com/user-attachments/assets/e8fe672f-e728-4eab-b841-eba457bab0e5" />



Si has llegado hasta aquí, quiero agradecerte que te hayas tomado el tiempo para leerlo, esto es simplemente un proyecto que hice en mi camino de aprendizaje de Python/Data analysis&Machine Learning y mi pasión y curiosidad por el universo.
