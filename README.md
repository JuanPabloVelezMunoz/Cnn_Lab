# Laboratorio: Redes Neuronales Convolucionales

## Fashion-MNIST — Análisis y Experimentación

## Descripción del problema

Clasificación de prendas de vestir en 10 categorías usando Fashion-MNIST.
Se compara un modelo baseline (fully connected) contra una CNN diseñada
desde cero, y se experimenta con el efecto del tamaño de kernel.

## Dataset

- **Fuente:** `keras.datasets.fashion_mnist`
- **Tamaño:** 70,000 imágenes (60,000 train / 10,000 test)
- **Dimensiones:** 28×28 píxeles, escala de grises
- **Clases:** 10, perfectamente balanceadas (6,000 por clase)
- **Preprocesamiento:** normalización a [0,1], añadir dimensión de canal

## Arquitectura CNN

```
Input (28×28×1)
+ Conv2D(32, 3×3, relu, same)
+ MaxPooling2D(2×2)
+ Conv2D(64, 3×3, relu, same)
+ MaxPooling2D(2×2)
+ Flatten
+ Dense(64, relu)
+ Dense(10, softmax)
```

## Resultados

| Modelo        | Parámetros | Accuracy test |
| ------------- | ---------- | ------------- |
| Baseline (FC) | 235,146    | 88.10%        |
| CNN 3×3       | 220,234    | 91.48%        |
| CNN 5×5       | 253,514    | 91.43%        |

## Experimento controlado

Variable analizada: tamaño de kernel (3×3 vs 5×5).  
Conclusión: el kernel 3×3 logra mejor accuracy con menos parámetros
para imágenes pequeñas de 28×28 píxeles.

## Interpretación

La CNN supera al baseline porque preserva la estructura espacial de las
imágenes. La convolución introduce localidad e invariancia a la traslación
como inductive bias, lo cual es apropiado para datos con estructura espacial.

## Cómo ejecutar

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanPabloVelezMunoz/cnn_lab.git
cd cnn_lab
```

### 2. Instalar dependencias

```bash
pip install tensorflow numpy matplotlib
```

### 3. Abrir el notebook localmente

```bash
jupyter notebook notebooks/cnn_lab.ipynb
```

### 4. Ejecutar en SageMaker

1. Abrir SageMaker Studio
2. Abrir una terminal y clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/cnn_lab.git
```

3. Abrir `notebooks/cnn_lab.ipynb`
4. Seleccionar kernel `Python 3 (ipykernel)`
5. Ejecutar todas las celdas con `Run All`

## Estructura del proyecto

```
cnn_lab/
├── notebooks/
│   └── cnn_lab.ipynb
├── scripts/
├── .gitignore
└── README.md
```

## Tecnologías

- Python 3.11
- TensorFlow 2.x / Keras
- NumPy, Matplotlib
- AWS SageMaker (entrenamiento y despliegue)
