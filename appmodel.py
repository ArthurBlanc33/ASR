# main.py

import torch
import whisper
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

class WhisperDataset(Dataset):
    def __init__(self, csv_path, max_length=480000):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            audio, sr = librosa.load(row['audio_path'], sr=16000)
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            return {
                'audio': torch.FloatTensor(audio),
                'transcription': str(row['transcription']).strip(),
                'valid': True
            }
        except:
            return {
                'audio': torch.zeros(self.max_length),
                'transcription': "",
                'valid': False
            }

class WhisperFineTunerRobust:
    def __init__(self, model_name="small", language="fr"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name).to(self.device)
        self.language = language
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=language, task="transcribe")
        self.setup_for_finetuning()

    def setup_for_finetuning(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.decoder.parameters():
            p.requires_grad = True

    def compute_loss(self, audio_batch, text_batch):
        valid_indices = [i for i, text in enumerate(text_batch) if text.strip()]
        if not valid_indices:
            return None

        valid_audio = audio_batch[valid_indices]
        valid_texts = [text_batch[i] for i in valid_indices]

        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(valid_audio).to(self.device)
            audio_features = self.model.encoder(mel)

        tokens = []
        for text in valid_texts:
            try:
                token = self.tokenizer.encode(text.strip())[:448]
                if len(token) > 0:
                    tokens.append(token)
                else:
                    print(f"⚠️ Token vide pour: '{text}'")
            except Exception as e:
                print(f"Erreur tokenization: {e} pour '{text}'")
                continue

        if not tokens:
            return None

        max_len = max(len(t) for t in tokens)
        eot_token = getattr(self.tokenizer, 'eot', 50257)
        padded_tokens = [t + [eot_token] * (max_len - len(t)) for t in tokens]
        tokens_tensor = torch.tensor(padded_tokens).to(self.device)

        # Ajout du contrôle pour éviter les tenseurs vides
        if tokens_tensor.shape[1] <= 1:
            print("⚠️ Batch ignoré : tous les tokens sont vides ou trop courts.")
            return None

        logits = self.model.decoder(tokens_tensor[:, :-1], audio_features)
        loss_fn = nn.CrossEntropyLoss(ignore_index=eot_token)
        loss = loss_fn(logits.transpose(1, 2), tokens_tensor[:, 1:])

        return loss

    def train(self, dataset, epochs=3, batch_size=2, lr=1e-5):
        valid_indices = [i for i in range(len(dataset)) if dataset[i]['valid'] and dataset[i]['transcription'].strip()]
        valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
        dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.decoder.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss, valid_batches = 0, 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                audio_batch = batch['audio'].to(self.device)
                text_batch = batch['transcription']
                loss = self.compute_loss(audio_batch, text_batch)
                if loss is not None and not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    valid_batches += 1
            print(f"Epoch {epoch+1} - Avg Loss: {total_loss / valid_batches:.4f} ({valid_batches} batches)")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    print("CUDA dispo :", torch.cuda.is_available())
    csv_path = "../whisper_dataset_clean.csv"  # À adapter

    # 1. Charger le fichier CSV complet
    df = pd.read_csv(csv_path)

    # 2. Split en train/test (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Sauvegarder les fichiers (facultatif mais utile)
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # 4. Créer les datasets sur TOUT le train
    train_dataset = WhisperDataset("train.csv")
    test_dataset = WhisperDataset("test.csv")

    # Entraîner sur tout le train
    tuner = WhisperFineTunerRobust("small", language="EN")
    # Améliorer le modèle
    # tuner.model.load_state_dict(torch.load("../whisper_finetuned_local.pth", map_location=tuner.device))
    
    # Puis tu peux relancer l'entraînement
    tuner.train(train_dataset, epochs=2, batch_size=6, lr=1e-5)
    tuner.save_model("whisper_finetuned_localupdtate.pth")
