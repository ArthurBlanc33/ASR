# Whisper Fine-Tuning FD

Ce projet permet de préparer un dataset audio/texte, d’entraîner un modèle Whisper sur vos propres données, et d’évaluer ses performances.

## 🧰 Prérequis

- Python 3.8+
- Créez un environnement virtuel (optionnel mais recommandé) :

```bash
python -m venv whisper_env
# Puis activez-le
whisper_env\Scripts\activate  # Sous Windows
source whisper_env/bin/activate  # Sous Linux/macOS
```

- Installez les dépendances nécessaires :

```bash
pip install torch pandas numpy librosa tqdm scikit-learn jiwer openai-whisper
```

## 🎧 Préparation des données

1. Téléchargez les données depuis Google Drive :
   👉 https://drive.google.com/file/d/1PtJS3BI5wW0N8z7lu7s_B5ZtsL5X9ZDS/view?usp=drive_link

2. Placez les dossiers `all_audio/` et `all_text/` à la **racine du projet** (même niveau que `appmodel.py`).

## 🚀 Split & entraînement

Lancez `appmodel.py` pour effectuer :

- Le **split du dataset** en `train.csv` / `test.csv`
- L’entraînement du modèle sur vos données personnalisées :

```python
from appmodel import WhisperFineTunerRobust, WhisperDataset
import torch

train_dataset = WhisperDataset("train.csv")
tuner = WhisperFineTunerRobust("tiny", language="EN")
tuner.train(train_dataset, epochs=2, batch_size=6, lr=1e-5)
tuner.save_model("whisper_finetuned_local.pth")
```

- Pour **réentraîner** un modèle déjà sauvegardé :

```python
tuner.model.load_state_dict(torch.load("whisper_finetuned_local.pth", map_location=tuner.device))
tuner.train(train_dataset, epochs=2, batch_size=6, lr=1e-5)
```

## ✅ Conseils

- Pour accélérer l'entraînement, commencez avec un sous-ensemble du dataset (e.g., 100 premiers exemples).
- Les tailles de modèles disponibles : `tiny`, `base`, `small`, `medium`, `large`.

---

