# 🧠📷 Clasificador y Detector de Objetos con OpenCV y TensorFlow

Este proyecto implementa un **sistema de visión por computadora** capaz de **detectar objetos en tiempo real** mediante una cámara web y **clasificarlos en categorías personalizadas** utilizando una **Red Neuronal Convolucional (CNN)** entrenada por el usuario.

---

##  Características

* ✅ Detección de objetos en tiempo real con **SSD MobileNet (OpenCV DNN)**
* ✅ Clasificación personalizada con **CNN (TensorFlow / Keras)**
* ✅ Dataset organizado por carpetas (sin bounding boxes manuales)
* ✅ Etiqueta **“Desconocido”** para objetos fuera de las clases entrenadas
* ✅ Uso de webcam
* ✅ Arquitectura clara y modular

---

## 🧩 Tecnologías utilizadas

* **Python 3.10+**
* **OpenCV**
* **TensorFlow / Keras**
* **NumPy**
* **SSD MobileNet v3 (COCO – OpenCV DNN)**

---

## 📁 Estructura del proyecto

```
ClasificadorObjetos/
│
├── clasificador_detector.py
├── clasificador_objetos_cnn.keras
├── clases_objetos.npy
│
├── ObjetosTrainIA/
│   ├── Cup/
│   ├── Shoe/
│   ├── bottle/
│   ├── hat/
│   ├── smartphone/
│   ├── sunglasses/
│   └── watch/
│
├── detector/
│   ├── frozen_inference_graph.pb
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
│   └── coco.names
│
└── README.md
```

---

## 📦 Dataset

El dataset debe estar organizado en **carpetas por clase**, donde cada carpeta contiene aproximadamente **300 imágenes** del objeto correspondiente.

Ejemplo:

```
ObjetosTrainIA/
 ├── Cup/
 ├── Shoe/
 ├── bottle/
```

📌 **No se utiliza reconocimiento facial**.
📌 El proyecto está enfocado únicamente en **clasificación de objetos**.

---

## 🧠 Funcionamiento del sistema

El sistema opera en **dos etapas**:

### 1️⃣ Detección de objetos

* Se utiliza **SSD MobileNet v3** preentrenado en el dataset **COCO**
* Detecta regiones donde hay objetos (bounding boxes)
* Se ejecuta con **OpenCV DNN**

### 2️⃣ Clasificación de objetos

* Cada región detectada se recorta
* Se clasifica con una **CNN entrenada con imágenes propias**
* Si la probabilidad es baja → se muestra **“Desconocido”**

---

## ⚙️ Instalación

### 1️⃣ Crear entorno virtual (recomendado)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Instalar dependencias

```bash
pip install tensorflow opencv-python numpy
```

---

## 🧪 Entrenamiento del modelo

Antes de usar la webcam, es necesario entrenar la CNN:

```bash
python clasificador_detector.py --mode train
```

Esto generará:

* `clasificador_objetos_cnn.keras`
* `clases_objetos.npy`

---

## 🎥 Detección y clasificación en tiempo real

Para ejecutar el sistema con la cámara web:

```bash
python clasificador_detector.py --mode webcam
```

Presiona **ESC** para salir.

---

## 🟢 Ejemplo de salida

* 🟦 Rectángulo azul → objeto detectado (SSD)
* 🟩 Texto verde → clase predicha por la CNN
* ❓ **Desconocido** → objeto fuera de las clases entrenadas

---

## 📌 Notas importantes

* El detector **no es YOLO**
* El clasificador solo reconoce las clases entrenadas
* Se recomienda buena iluminación y fondo neutro
* El rendimiento mejora con mayor variedad de imágenes


---
