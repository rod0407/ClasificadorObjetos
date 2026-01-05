import os
import re
import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==========================
# CONFIGURACIÓN
# ==========================

# PON AQUÍ LA RUTA A TU CARPETA "ObjetosTrainIA"
dataPath   = r"C:\Users\rodri\Documents\ClasificadorObjetos\ObjetosTrainIA"
modeloPath = "modelo_objetos_svm.pkl"
clasesPath = "clases_objetos.npy"

IMG_SIZE = 64


#############################

def extraer_hog(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))

    feature = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return feature.astype(np.float32)


# ==========================
# CARGA DE DATOS
# ==========================

def cargar_dataset(ruta_base):
    imagenes = []
    etiquetas = []
    class_names = []

    for idx, nombre_clase in enumerate(sorted(os.listdir(ruta_base))):
        carpeta_clase = os.path.join(ruta_base, nombre_clase)
        if not os.path.isdir(carpeta_clase):
            continue

        print(f"[INFO] Leyendo clase {idx}: {nombre_clase}")
        class_names.append(nombre_clase)

        for filename in os.listdir(carpeta_clase):
            if not re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename, re.IGNORECASE):
                continue

            filepath = os.path.join(carpeta_clase, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"  [WARN] No se pudo leer: {filepath}")
                continue

            img_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
            feature = extraer_hog(img)


            imagenes.append(feature)
            etiquetas.append(idx)

    X = np.array(imagenes, dtype=np.float32)
    y = np.array(etiquetas, dtype=np.int32)

    print("[INFO] Total de imágenes:", X.shape[0])
    print("[INFO] Dimensión de feature:", X.shape[1])
    print("[INFO] Clases:", class_names)

    return X, y, class_names

# ==========================
# ENTRENAMIENTO
# ==========================

def entrenar_modelo():
    X, y, class_names = cargar_dataset(dataPath)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, gamma="scale", C=10.0)
    )

    print("\n[INFO] Entrenando modelo...")
    clf.fit(X_train, y_train)
    print("[INFO] Entrenamiento terminado.")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[INFO] Exactitud en test: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    joblib.dump(clf, modeloPath)
    np.save(clasesPath, np.array(class_names))
    print(f"[INFO] Modelo guardado en {modeloPath}")
    print(f"[INFO] Clases guardadas en {clasesPath}")

# ==========================
# PREDICCIÓN DE IMAGEN
# ==========================

def cargar_modelo_y_clases():
    if not os.path.exists(modeloPath):
        raise FileNotFoundError("Primero entrena el modelo (entrenar_modelo()).")
    clf = joblib.load(modeloPath)
    class_names = np.load(clasesPath, allow_pickle=True).tolist()
    return clf, class_names

def preprocesar_imagen(filepath):
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {filepath}")

    feature = extraer_hog(img)       
    return feature.reshape(1, -1)


def predecir_imagen(filepath):
    clf, class_names = cargar_modelo_y_clases()
    X = preprocesar_imagen(filepath)
    pred = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]
    prob = probs[pred]
    print(f"Imagen: {filepath}")
    print(f"Predicción: {class_names[pred]} (prob = {prob:.2f})")

# ==========================
# DEMO WEBCAM
# ==========================

def demo_webcam():
    clf, class_names = cargar_modelo_y_clases()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    print("[INFO] Presiona ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ==== ROI central (recorte cuadrado) ====
        h, w = frame.shape[:2]
        size = int(min(h, w) * 0.6)   # usa 60% del área menor
        cx, cy = w // 2, h // 2
        x1, y1 = cx - size//2, cy - size//2
        x2, y2 = cx + size//2, cy + size//2
        roi = frame[y1:y2, x1:x2]

        # feature HOG SOLO sobre ROI
        feature = extraer_hog(roi).reshape(1, -1)

        pred  = clf.predict(feature)[0]
        probs = clf.predict_proba(feature)[0]
        prob  = probs[pred]

        if prob < 0.60:
            label = "Desconocido"
        else:
            label = f"{class_names[pred]} ({prob:.2f})"

        # Dibuja el recuadro ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Clasificador de objetos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    # 1) ENTRENAR
    #entrenar_modelo()

    # 2) DEMO
    #predecir_imagen(r"C:/Users/TU_USUARIO/Documents/ObjetosTrainIA/Cup/alguna_imagen.jpg")
    demo_webcam()
