# 🖼️ Classification d'Images CIFAR-10 avec CNN

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Description

Projet du module  **Deep Learning** réalisé dans le cadre du Master IMSD à l'ENSA Khouribga. Ce projet implémente un **réseau de neurones convolutif (CNN)** pour la classification d'images du dataset CIFAR-10.

### 🎯 Objectif

Développer un modèle robuste capable de classifier automatiquement des images basse résolution (32x32 pixels) en 10 catégories distinctes avec une précision supérieure à 71%.

---

## 📊 Dataset CIFAR-10

<div align="center">

| Caractéristique | Valeur |
|----------------|---------|
| **Images Total** | 60 000 (RGB) |
| **Résolution** | 32×32 pixels |
| **Classes** | 10 classes distinctes |
| **Split** | 50k Train / 10k Test |

</div>

### 🏷️ Classes

- ✈️ Avion (Airplane)
- 🚗 Automobile
- 🐦 Oiseau (Bird)
- 🐱 Chat (Cat)
- 🦌 Cerf (Deer)
- 🐕 Chien (Dog)
- 🐸 Grenouille (Frog)
- 🐴 Cheval (Horse)
- 🚢 Bateau (Ship)
- 🚚 Camion (Truck)

### 🔍 Défis du Dataset

- **Faible résolution** : Perte de texture et détails fins
- **Arrière-plans complexes** : Éléments parasites dans l'image
- **Variabilité de posture** : Angles et positions variés

---

## 🏗️ Architecture du Modèle

### Structure CNN

```
┌─────────────────────────────────────────┐
│  Extraction des Caractéristiques        │
├─────────────────────────────────────────┤
│  Bloc 1: Détection Initiale             │
│  • Conv2D (32 filtres, 3×3, ReLU)       │
│  • Conv2D (32 filtres, 3×3, ReLU)       │
│  • MaxPooling2D (2×2)                   │
├─────────────────────────────────────────┤
│  Bloc 2: Complexité Accrue              │
│  • Conv2D (64 filtres, 3×3, ReLU)       │
│  • Conv2D (64 filtres, 3×3, ReLU)       │
│  • MaxPooling2D (2×2)                   │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Classification et Sortie               │
├─────────────────────────────────────────┤
│  • Flatten & Dense (64 unités)          │
│  • Dropout (0.5)                        │
│  • Softmax (10 classes)                 │
└─────────────────────────────────────────┘
```

**Total Paramètres:** ~328 000

---

## 🔧 Prétraitement & Pipeline

### 1. Normalisation
```python
# Conversion [0, 255] → [0, 1]
X_train = X_train.astype('float32') / 255.0
```

### 2. One-Hot Encoding
```python
# Labels catégoriels
y_train = to_categorical(y_train, 10)
```

### 3. Data Augmentation

Techniques appliquées pour améliorer la généralisation :

| Technique | Paramètre |
|-----------|-----------|
| **Rotation** | ±15° |
| **Translation** | Shift horizontal/vertical |
| **Flip Horizontal** | Oui |
| **Zoom** | Facteur aléatoire |

**Objectif:** Forcer l'apprentissage de caractéristiques invariantes.

---

## 🎓 Entraînement

### Hyperparamètres

| Paramètre | Valeur Testée | **Sélection Finale** |
|-----------|---------------|----------------------|
| Dropout (Conv) | 0.15, 0.3 | **0.3** |
| Dropout (Dense) | 0.3, 0.5 | **0.5** |
| Filtres (Couche 1) | 32 vs 64 | **32** |
| Learning Rate | N/A | **0.0005** (avec Scheduler) |

### Stratégies de Régularisation

- ✅ **Dropout** : Force la redondance des neurones
- ✅ **Early Stopping** : Arrêt au pic de performance validation
- ✅ **Optimiseur** : Adam (adaptatif)

**Méthode d'optimisation:** Grid/Random Search

---

## 📈 Résultats

### Performance Globale

<div align="center">

```
╔════════════════════════════════╗
║  Accuracy Globale : 71.1%      ║
╚════════════════════════════════╝
```

</div>

### F1-Scores par Classe

| Classe | F1-Score | Performance |
|--------|----------|-------------|
| 🚢 Bateau | 0.83 | ⭐⭐⭐ Excellent |
| 🚗 Automobile | 0.82 | ⭐⭐⭐ Excellent |
| 🚚 Camion | 0.76 | ⭐⭐ Bon |
| 🐦 Oiseau | 0.57 | ⭐ Moyen |
| 🐱 Chat | 0.50 | ⭐ Moyen |

**Observation:** Le modèle excelle sur les structures rigides (véhicules) mais peine sur les formes organiques (animaux).

### Courbes d'Apprentissage

- ✅ **Convergence saine** : Pas de surapprentissage majeur
- ✅ **Écart Train/Val stable** : Généralisation satisfaisante

---

## 🔬 Analyse des Erreurs

### Matrice de Confusion

**Principales confusions identifiées:**

1. **Sémantique (35%)** : Chat ↔ Chien, Cerf ↔ Cheval
2. **Silhouettes similaires** : Grenouille ↔ Cheval (formes proches à basse résolution)

### Explainability : Saliency Maps

Les cartes de saillance révèlent que le modèle :
- ✅ Se focalise sur l'**objet central**
- ✅ Ignore efficacement l'**arrière-plan**
- ⚠️ Peut se tromper sur des **patterns de texture** ambigus

---

## 📊 Comparaison Architecturale

### CNN vs MLP

| Architecture | Accuracy | Avantage |
|--------------|----------|----------|
| **MLP** | 53.7% | Perte de structure spatiale |
| **CNN** | **71.1%** | Préservation des motifs locaux (2D) |

**Gain relatif:** +32% grâce aux convolutions

---

## 🚀 Perspectives d'Amélioration

### 1. Architectures Profondes
- 🔹 **ResNet** : Connexions résiduelles pour capturer les détails fins
- 🔹 **VGG** : Profondeur accrue

### 2. Transfer Learning
- 🔹 Utilisation de poids pré-entraînés sur **ImageNet**
- 🔹 Fine-tuning des dernières couches

### 3. Super-Resolution
- 🔹 **Augmenter la netteté** avant classification
- 🔹 Techniques comme SRGAN ou ESRGAN

---

## 📁 Structure du Projet

```
projet-deep-learning/
│
├── 📓 ensa-master-imsd-dl-projet-3-cifar10__2_.ipynb   # Notebook principal
├── 📄 DL_-_Rapport.docx                                 # Rapport détaillé
├── 🎯 CNN_Classification_Low_Resolution_Imagery.pdf     # Présentation
├── 📖 README.md                                         # Ce fichier
│

```

---

## 🛠️ Installation & Utilisation

### Prérequis

```bash
Python >= 3.8
TensorFlow >= 2.8
Keras
NumPy
Matplotlib
Seaborn
```

### Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/cifar10-cnn-classification.git
cd cifar10-cnn-classification

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution

```bash
# Lancer Jupyter Notebook
jupyter notebook ensa-master-imsd-dl-projet-3-cifar10__2_.ipynb
```

Ou exécuter directement :

```python
# Charger et entraîner le modèle
python train_model.py

# Évaluer le modèle
python evaluate.py
```

---

## 📚 Références

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - Alex Krizhevsky
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow et al.
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)

---

## 👥 Auteurs

  
📍 SARA MAGGGAG & ACHRAF MASNSARI  


---

## 📄 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements


- 📚 Communauté TensorFlow/Keras
- 🌐 Dataset CIFAR-10 par Alex Krizhevsky

---

<div align="center">

**⭐ Si ce projet vous aide, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ and 🧠 

</div>
