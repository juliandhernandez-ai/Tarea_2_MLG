# Tarea 2: Modelos Lineales Generalizados

Este repositorio contiene el material de referencia, los datos y el cuaderno de análisis de la Tarea 2. El cuaderno [`puntos_2_y_3.ipynb`](./puntos_2_y_3.ipynb) desarrolla los ejercicios 2 y 3 con Python.

## Contenido

| Archivo | Descripción |
| --- | --- |
| [`tarea2.pdf`](./tarea2.pdf) | Enunciado de la tarea. |
| [`Argumentos.pdf`](./Argumentos.pdf) | Documento PDF de referencia incluido con la tarea. |
| [`punto_2.ipynb`](./punto_2.ipynb) | Desarrollo ordenado del punto 2: análisis descriptivo, cuasi-verosimilitud y Simplex para los datos de cebada. |
| [`puntos_2_y_3.ipynb`](./puntos_2_y_3.ipynb) | Análisis ejecutable de cebada y grasa corporal. |
| [`cebada.dat`](./cebada.dat) | Datos de proporción de área foliar afectada, lugar y variedad de cebada. |
| [`bodyfat.txt`](./bodyfat.txt) | Datos de porcentaje de grasa corporal y edad. |
| [`fig_cebada_boxplots.png`](./fig_cebada_boxplots.png) | Boxplots generados para la exploración de los datos de cebada. |

## Requisitos

Se necesita Python y las siguientes bibliotecas:

- NumPy
- pandas
- SciPy
- Matplotlib
- Seaborn

Instálalas desde una terminal con:

```bash
python -m pip install numpy pandas scipy matplotlib seaborn
```

Para abrir el cuaderno, utiliza Jupyter Notebook/JupyterLab o la extensión de Jupyter en Visual Studio Code. Asegúrate de que `cebada.dat` y `bodyfat.txt` estén en el directorio de trabajo; si ejecutas el cuaderno desde otra ubicación, configura ese directorio para que apunte a la carpeta del repositorio. Ejecuta las celdas en orden.

## Análisis incluidos

### Ejercicio 2: cebada

El cuaderno carga `cebada.dat` y compara gráficamente la proporción de área foliar afectada (`area`) según el lugar geográfico y la variedad. Genera y guarda los diagramas en `fig_cebada_boxplots.png`.

Luego ajusta modelos por cuasi-verosimilitud con enlace logit y dos funciones de varianza: \(V(\pi)=\pi(1-\pi)\) y \(V(\pi)=\{\pi(1-\pi)\}^2\). Para cada ajuste reporta el estadístico \(X^2\) de Pearson y la dispersión estimada. También documenta por qué el ajuste Simplex directo considerado en el cuaderno no admite las observaciones de respuesta iguales a cero.

### Ejercicio 3: grasa corporal

El archivo `bodyfat.txt` contiene el porcentaje de grasa corporal y la edad. El cuaderno convierte el porcentaje a una proporción y modela su media mediante una regresión logística de la edad para comparar:

- **Regresión Simplex**, estimada por máxima verosimilitud.
- **Regresión Beta**, también estimada por máxima verosimilitud.

La comparación utiliza AIC y BIC. En la ejecución guardada en el cuaderno, el modelo Beta obtiene AIC = -593.23 y BIC = -582.66; el modelo Simplex obtiene AIC = -559.49 y BIC = -548.91. Según el criterio de comparación empleado, la diferencia de AIC (33.74) favorece marcadamente al modelo Beta.

## Reproducibilidad

Los archivos de datos se leen usando rutas relativas al directorio de trabajo. La celda de cebada vuelve a crear `fig_cebada_boxplots.png` al ejecutarse. Los resultados numéricos descritos arriba corresponden a las salidas actualmente guardadas en el cuaderno y pueden variar si se modifican los datos, el código o el entorno de ejecución.
