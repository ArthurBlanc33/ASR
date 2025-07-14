import whisper
import torch
import pandas as pd
from jiwer import wer  # pip install jiwer

# 1. Charger le modèle
model = whisper.load_model("small")
state_dict = torch.load("../whisper_finetuned_localupdate.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 2. Charger le test set
df = pd.read_csv("test.csv", header=None, names=["audio_path", "transcription"])

# 3. Boucle sur les N premiers exemples
N = 20  # nombre d'exemples à tester
refs, hyps = [], []
for i, row in df.head(N).iterrows():
    audio_path = row["audio_path"]
    ref = str(row["transcription"])
    try:
        result = model.transcribe(audio_path, language="en")
        hyp = result["text"].strip()
    except Exception as e:
        hyp = ""
        print(f"Erreur sur {audio_path} : {e}")
    refs.append(ref)
    hyps.append(hyp)
    print(f"Réel : {ref}")
    print(f"Prédit : {hyp}")
    print("---")

# 4. Calcul du WER global
score = wer(refs, hyps)
print(f"\nWER sur {N} exemples : {score:.2%}")
