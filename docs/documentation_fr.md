# whatsapp-analyzer — Documentation complète

> **Language / Langue :** [English](documentation_en.md) · Français

---

## Table des matières

1. [Présentation](#1-présentation)
2. [Installation](#2-installation)
3. [Démarrage rapide](#3-démarrage-rapide)
   - 3.1 [Interface en ligne de commande (CLI)](#31-interface-en-ligne-de-commande-cli)
   - 3.2 [API Python](#32-api-python)
   - 3.3 [Interface web](#33-interface-web)
4. [Formats d'entrée](#4-formats-dentrée)
5. [Architecture du pipeline](#5-architecture-du-pipeline)
6. [Référence des modules](#6-référence-des-modules)
   - 6.1 [Loader](#61-loader)
   - 6.2 [Parser](#62-parser)
   - 6.3 [Cleaner](#63-cleaner)
   - 6.4 [TopicClassifier](#64-topicclassifier)
   - 6.5 [SentimentAnalyzer](#65-sentimentanalyzer)
   - 6.6 [TemporalAnalyzer](#66-temporalanalyzer)
   - 6.7 [UserAnalyzer](#67-useranalyzer)
   - 6.8 [MediaAnalyzer](#68-mediaanalyzer)
   - 6.9 [GroupComparator](#69-groupcomparator)
   - 6.10 [Visualizer](#610-visualizer)
   - 6.11 [WhatsAppAnalyzer (core)](#611-whatsappanalyzer-core)
   - 6.12 [UserView](#612-userview)
   - 6.13 [Commandes CLI](#613-commandes-cli)
7. [Référence des sorties](#7-référence-des-sorties)
8. [Fonctionnalités optionnelles](#8-fonctionnalités-optionnelles)
9. [Confidentialité et aspects juridiques](#9-confidentialité-et-aspects-juridiques)
10. [Résolution des problèmes courants](#10-résolution-des-problèmes-courants)

---

## 1. Présentation

`whatsapp-analyzer` est un package Python **entièrement local** pour analyser les exports de groupes WhatsApp. Il traite une archive `.zip`, un fichier `_chat.txt` seul, ou un dossier décompressé, à travers un pipeline NLP multi-étapes :

| Étape | Ce qui se passe |
|-------|-----------------|
| **Chargement** | Détection du format, décompression si nécessaire |
| **Analyse syntaxique** | Extraction des messages dans un DataFrame structuré |
| **Nettoyage** | Suppression des emoji, détection de la langue, suppression des mots vides, lemmatisation |
| **Analyse** | Thèmes (LDA / BERTopic), sentiment (VADER / CamemBERT), profils utilisateurs, patterns temporels, statistiques médias |
| **Rapport** | HTML autonome avec graphiques, nuages de mots, fiches par utilisateur |

**Aucune donnée ne quitte votre machine.** Tout le traitement est local.

---

## 2. Installation

### Prérequis

- Python ≥ 3.12
- `pip` ou `uv`

### Installation avec pip

```
pip install wachat-analyzer
```

### Installation de base

```bash
git clone https://github.com/MendasD/whatsapp-analyzer.git
cd whatsapp-analyzer

# avec uv (recommandé)
uv sync

# ou avec pip
pip install -e .
```

### Modèles de langue spaCy

Les modèles spaCy sont téléchargés automatiquement au premier lancement. Pour les pré-installer manuellement :

```bash
python -m spacy download fr_core_news_sm   # Français
python -m spacy download en_core_web_sm   # Anglais
```

### Fonctionnalités optionnelles

Installez uniquement ce dont vous avez besoin :

```bash
pip install -e ".[bertopic]"   # Modélisation de thèmes BERTopic (GPU compatible)
pip install -e ".[camembert]"  # Sentiment français CamemBERT (nécessite PyTorch)
pip install -e ".[media]"      # Transcription audio/vidéo avec Whisper
```

---

## 3. Démarrage rapide

### 3.1 Interface en ligne de commande (CLI)

Toutes les commandes sont accessibles via le point d'entrée `whatsapp-analyzer`.

#### Analyser un groupe

```bash
whatsapp-analyzer analyze \
  --input data-example/_chat.txt \
  --topics 5 \
  --output reports/
```

Résultat attendu :

```
Analysing data-example/_chat.txt …
┌──────────────┬──────────────────────────────────────────────┐
│ Metric       │ Value                                        │
├──────────────┼──────────────────────────────────────────────┤
│ Group        │ _chat                                        │
│ Messages     │ 1 243                                        │
│ Participants │ 12                                           │
│ Period       │ 2024-01-01 → 2024-06-30                      │
│ Top topic    │ cours / td / examen / prof / notes           │
│ Top topic    │ sport / match / jouer / gagner / equipe      │
│ Top topic    │ sortie / soirée / vendredi / venir / ok      │
└──────────────┴──────────────────────────────────────────────┘
Report written to reports/report.html
```

#### Comparer plusieurs groupes

```bash
whatsapp-analyzer compare \
  --input data-example/_chat.txt \
  --input data-example/_chat0.txt \
  --output reports/
```

Chaque flag `--input` ajoute un groupe. Vous pouvez en passer autant que nécessaire.

#### Lancer l'interface web

```bash
whatsapp-analyzer serve
# Accessible sur http://localhost:8501
```

---

### 3.2 API Python

#### Analyse d'un groupe — style fluent

```python
from whatsapp_analyzer import WhatsAppAnalyzer

az = WhatsAppAnalyzer("data-example/_chat.txt", n_topics=5)
az.parse().clean().analyze()

chemin_rapport = az.report(output="reports/")  # → reports/report.html
chemin_csv     = az.to_csv(output="reports/")  # → reports/_chat.csv

print(f"Rapport : {chemin_rapport}")
```

#### Analyse d'un groupe — raccourci en une seule ligne

```python
from whatsapp_analyzer import WhatsAppAnalyzer

results = WhatsAppAnalyzer("data-example/_chat.txt").run()
# results["report_path"] contient le chemin vers le rapport HTML
```

#### Analyse détaillée par utilisateur

```python
from whatsapp_analyzer import WhatsAppAnalyzer

az = WhatsAppAnalyzer("data-example/_chat.txt")
az.parse().clean().analyze()

vue = az.user("Alice")
print(vue.summary())              # dict : nb_messages, sentiment_mean, …
print(vue.topics())               # DataFrame des affectations de thèmes
print(vue.sentiment_over_time())  # DataFrame avec timestamps et scores
print(vue.activity_heatmap())     # DataFrame 7×24 (jour × heure)
```

#### Comparer plusieurs groupes

```python
from pathlib import Path
from whatsapp_analyzer import WhatsAppAnalyzer
from whatsapp_analyzer.comparator import GroupComparator

az1 = WhatsAppAnalyzer("data-example/_chat.txt",  n_topics=5)
az2 = WhatsAppAnalyzer("data-example/_chat0.txt", n_topics=5)
az1.parse().clean().analyze()
az2.parse().clean().analyze()

comp = GroupComparator([az1, az2])
print(comp.compare_activity())   # DataFrame : une ligne par groupe
print(comp.compare_topics())     # DataFrame : poids des thèmes par groupe
print(comp.compare_sentiment())  # DataFrame : statistiques de sentiment par groupe
print(comp.common_users())       # DataFrame : auteurs présents dans plusieurs groupes

chemin_rapport = comp.report(Path("reports/"))
# → reports/comparison_report.html
```

#### Sélectionner les étapes d'analyse

```python
az = WhatsAppAnalyzer("data-example/_chat.txt")
az.parse().clean()

# Désactiver l'analyse de sentiment, activer la transcription media
az.analyze(topics=True, sentiment=False, temporal=True, media=True)
```

#### Forcer une langue

```python
az = WhatsAppAnalyzer("data-example/_chat.txt", lang="fr")
az.parse().clean().analyze()
```

#### Anonymiser les auteurs

```python
from whatsapp_analyzer.parser import Parser

df = Parser(anonymize=True).parse("data-example/_chat.txt")
# Les noms d'auteurs sont remplacés par des hachages SHA-256
```

---

### 3.3 Interface web

```bash
whatsapp-analyzer serve
```

L'interface Streamlit vous permet de :

- Importer un export `.zip` ou `.txt`
- Configurer le nombre de thèmes (2–15), le nombre minimal de mots, la langue
- Anonymiser optionnellement les noms d'auteurs
- Naviguer dans les résultats via cinq onglets : Vue d'ensemble, Thèmes, Sentiment, Activité, Rapport
- Télécharger le rapport HTML autonome

---

## 4. Formats d'entrée

Le package accepte trois types d'entrée, détectés automatiquement :

| Format | Exemple | Remarques |
|--------|---------|-----------|
| `.zip` | `WhatsApp Chat - Famille.zip` | Export natif WhatsApp, contient `_chat.txt` et les médias |
| `.txt` | `_chat.txt` | Fichier texte seul, sans médias |
| Dossier | `WhatsApp Chat - Famille/` | Dossier d'export décompressé |

### Formats de plateforme

Deux formats d'horodatage sont supportés :

**Android :**
```
12/01/2024, 08:15 - Alice: Bonjour tout le monde !
12/01/2024, 08:15 - Alice: Voici un message
sur plusieurs lignes.
```

**iOS :**
```
[12/01/2024 à 08:15:00] Alice : Bonjour tout le monde !
[12/01/2024, 08:15:00] Alice : Une autre variante
```

Les messages multi-lignes sont automatiquement fusionnés.

### Types de messages

Chaque message analysé est classé dans l'une des catégories suivantes :

| Type | Description |
|------|-------------|
| `text` | Message textuel normal d'un utilisateur |
| `media` | Message contenant un média omis |
| `system` | Événement système WhatsApp (membre ajouté, nom du groupe modifié, etc.) |

---

## 5. Architecture du pipeline

```
Entrée (ZIP / TXT / dossier)
         │
         ▼
    ┌─────────┐
    │ Loader  │  détecte le format, décompresse → LoadedGroup
    └────┬────┘
         │ chat_path, media_dir, group_name
         ▼
    ┌─────────┐
    │ Parser  │  regex → DataFrame [timestamp, author, message, msg_type, group_name]
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Cleaner │  suppression emoji, détection langue, mots vides, lemmatisation
    └────┬────┘  ajoute [cleaned_message, language, tokens]
         │
         ├─────────────────────────────────────────────────────────┐
         ▼                                                         ▼
  ┌──────────────────┐   ┌───────────────────┐   ┌───────────────────────┐
  │ TopicClassifier  │   │ SentimentAnalyzer │   │   TemporalAnalyzer    │
  │ LDA / BERTopic   │   │ VADER / CamemBERT │   │   heatmaps, timelines │
  └────────┬─────────┘   └────────┬──────────┘   └──────────┬────────────┘
           │                      │                          │
           └──────────────────────┼──────────────────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │ UserAnalyzer │  profils par utilisateur
                          └──────┬───────┘
                                  │
                   ┌──────────────┼──────────────┐
                   ▼              ▼               ▼
             ┌──────────┐  ┌───────────┐  ┌────────────┐
             │Visualizer│  │  CLI      │  │ Interface  │
             │rapport   │  │(click)   │  │ web        │
             │HTML      │  └───────────┘  │(Streamlit) │
             └──────────┘                └────────────┘
```

### Règles de dépendances

Les modules n'importent que des modules situés en amont dans le pipeline. Les bibliothèques lourdes (spaCy, transformers, sklearn, Whisper) sont importées **de façon paresseuse** à l'intérieur des corps de méthode, de sorte que le démarrage est instantané et que les dépendances optionnelles ne lèvent une `ImportError` que lorsque la fonctionnalité est réellement appelée.

---

## 6. Référence des modules

### 6.1 Loader

```python
from whatsapp_analyzer.loader import Loader, LoadedGroup
```

#### `Loader`

**`Loader().load(path)`**

Charge un groupe WhatsApp depuis n'importe quel format d'entrée supporté.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Chemin vers un `.zip`, `_chat.txt` ou dossier |

Retourne : `LoadedGroup`

Lève `FileNotFoundError` si aucun `_chat.txt` n'est trouvé dans le chemin.  
Lève `ValueError` si le fichier `.zip` est corrompu.

**`Loader().load_many(paths)`**

Charge plusieurs groupes en ignorant (avec avertissement) ceux qui échouent.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `paths` | `list[str \| Path]` | Un chemin par groupe |

Retourne : `list[LoadedGroup]`

Lève `RuntimeError` si **aucun** groupe n'a pu être chargé.

#### `LoadedGroup`

Attributs :

| Attribut | Type | Description |
|----------|------|-------------|
| `chat_path` | `Path` | Chemin vers le fichier `_chat.txt` |
| `media_dir` | `Path \| None` | Dossier médias, ou `None` si absent |
| `group_name` | `str` | Nom lisible du groupe |

**`loaded.cleanup()`** — Supprime le répertoire temporaire de décompression s'il en a été créé un (appelé automatiquement par `WhatsAppAnalyzer`).

---

### 6.2 Parser

```python
from whatsapp_analyzer.parser import Parser
```

#### `Parser`

**Paramètres du constructeur :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `anonymize` | `bool` | `False` | Remplace les noms d'auteurs par des hachages SHA-256 |
| `group_name` | `str \| None` | `None` | Remplace le nom de groupe stocké en sortie |

**`Parser().parse(chat_path)`**

Analyse un fichier `_chat.txt` et retourne un DataFrame.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `chat_path` | `Path` | Chemin vers le fichier d'export WhatsApp |

Retourne : `pd.DataFrame` avec les colonnes :

| Colonne | Type | Description |
|---------|------|-------------|
| `timestamp` | `datetime64[ns]` | Horodatage du message |
| `author` | `str` | Nom d'affichage de l'auteur |
| `message` | `str` | Texte brut du message |
| `msg_type` | `str` | `'text'`, `'media'` ou `'system'` |
| `group_name` | `str` | Nom du groupe |

Lève `ValueError` si aucun message n'a pu être analysé.

---

### 6.3 Cleaner

```python
from whatsapp_analyzer.cleaner import Cleaner
```

#### `Cleaner`

**Paramètres du constructeur :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `lang` | `str \| None` | `None` | Force le code de langue (`'fr'`, `'en'`). `None` = auto-détection |
| `remove_emoji` | `bool` | `True` | Supprime les caractères emoji |
| `min_words` | `int` | `3` | Supprime les messages nettoyés plus courts que ce nombre de mots |
| `use_lemma` | `bool` | `True` | Applique la lemmatisation spaCy quand un modèle est disponible |

**`Cleaner().clean(df)`**

Applique le pipeline complet de prétraitement NLP à un DataFrame analysé.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Sortie de `Parser.parse()` |

Retourne : `pd.DataFrame` filtré avec trois colonnes supplémentaires :

| Colonne | Type | Description |
|---------|------|-------------|
| `cleaned_message` | `str` | Chaîne de tokens nettoyés et lemmatisés |
| `language` | `str` | Code de langue détecté (ex. `'fr'`) |
| `tokens` | `list[str]` | Liste tokenisée des mots nettoyés |

**Étapes de traitement :**
1. Suppression des messages système et médias (conservation uniquement de `msg_type == 'text'`)
2. Normalisation Unicode et suppression des caractères de contrôle
3. Suppression des emoji (via la bibliothèque `emoji` si installée, repli ASCII sinon)
4. Détection de la langue dominante sur les 10 premiers messages
5. Mise en minuscules et suppression de la ponctuation
6. Suppression des mots vides (spaCy → NLTK → ensemble vide, par ordre de priorité)
7. Lemmatisation avec spaCy quand un modèle correspondant est disponible
8. Suppression des messages plus courts que `min_words`

---

### 6.4 TopicClassifier

```python
from whatsapp_analyzer.topic_classifier import TopicClassifier
```

#### `TopicClassifier`

**Paramètres du constructeur :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_topics` | `int` | `5` | Nombre de thèmes à extraire |
| `method` | `str` | `'lda'` | `'lda'` (toujours disponible) ou `'bertopic'` (nécessite l'extra `[bertopic]`) |

**`TopicClassifier().fit_transform(df)`**

Entraîne un modèle de thèmes et annote chaque message.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Sortie de `Cleaner.clean()`, doit contenir `cleaned_message` |

Retourne : `dict` avec les clés :

| Clé | Type | Description |
|-----|------|-------------|
| `df` | `pd.DataFrame` | DataFrame d'entrée enrichi avec `topic_id` (int) et `topic_score` (float) |
| `group_topics` | `pd.DataFrame` | Colonnes : `topic_id`, `topic_label`, `weight` |

`topic_label` est une chaîne de cinq mots les plus représentatifs séparés par des barres obliques, ex. `"cours / td / examen / prof / notes"`.

Lève `ValueError` si `df` est vide ou si `cleaned_message` est absent.  
Lève `RuntimeError` si `method='bertopic'` et que BERTopic n'est pas installé.

---

### 6.5 SentimentAnalyzer

```python
from whatsapp_analyzer.sentiment_analyzer import SentimentAnalyzer
```

#### `SentimentAnalyzer`

**Paramètres du constructeur :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `method` | `str` | `'vader'` | `'vader'` (toujours disponible) ou `'camembert'` (nécessite l'extra `[camembert]`) |

**`SentimentAnalyzer().analyze(df)`**

Attribue un score de sentiment à chaque message et calcule des statistiques agrégées.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Sortie de `Cleaner.clean()`, doit contenir `cleaned_message` |

Retourne : `dict` avec les clés :

| Clé | Type | Description |
|-----|------|-------------|
| `df` | `pd.DataFrame` | DataFrame d'entrée enrichi avec `sentiment_score` (float, de −1 à 1) et `sentiment_label` (str) |
| `by_user` | `pd.DataFrame` | Colonnes `author` et `sentiment_score` (moyenne par utilisateur) |
| `global` | `dict` | Clés : `mean` (float), `pos_pct` (float 0–1), `neg_pct` (float 0–1) |

Étiquettes de sentiment : `'positive'` (score ≥ 0,05), `'negative'` (score ≤ −0,05), `'neutral'` sinon.

Lève `ValueError` si `df` est vide ou si `cleaned_message` est absent.  
Lève `RuntimeError` si `method='camembert'` et que la bibliothèque n'est pas installée.

**Moteurs disponibles :**

- **VADER** — basé sur des règles, agnostique à la langue, rapide, sans GPU.
- **CamemBERT** (`cmarkea/distilcamembert-base-sentiment`) — modèle transformer fine-tuné pour le sentiment en français, retourne des notes de 1 à 5 étoiles converties en [−1, 1].

---

### 6.6 TemporalAnalyzer

```python
from whatsapp_analyzer.temporal_analyzer import TemporalAnalyzer
```

#### `TemporalAnalyzer`

Pas de paramètres de constructeur.

**`TemporalAnalyzer().analyze(df)`**

Calcule les métriques d'activité temporelle à partir du DataFrame nettoyé.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Doit contenir une colonne `timestamp` (datetime64[ns]) |

Retourne : `dict` avec les clés :

| Clé | Type | Description |
|-----|------|-------------|
| `timeline` | `pd.DataFrame` | Nombre de messages par jour ; index datetime, colonne `count` |
| `hourly_heatmap` | `pd.DataFrame` | 7 lignes (Lun–Dim) × 24 colonnes (0–23), valeurs = nombre de messages |
| `weekly_activity` | `pd.Series` | Nombre de messages par jour de la semaine (Lundi → Dimanche) |
| `monthly_activity` | `pd.Series` | Nombre de messages par mois calendaire (index Period) |
| `peak_hour` | `int` | Heure la plus active globalement (0–23) |
| `peak_day` | `str` | Jour de la semaine le plus actif (ex. `"Friday"`) |

Lève `ValueError` si `df` est vide ou si `timestamp` est absent.

---

### 6.7 UserAnalyzer

```python
from whatsapp_analyzer.user_analyzer import UserAnalyzer
```

#### `UserAnalyzer`

Pas de paramètres de constructeur.

**`UserAnalyzer().build_profiles(results)`**

Construit un profil pour chaque auteur présent dans les résultats du pipeline.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | Résultats du pipeline avec les clés `df_clean`, `topics`, `sentiment` |

Retourne : `dict[str, dict]` — correspondance nom d'auteur → dictionnaire de profil.

Chaque profil contient :

| Clé | Type | Description |
|-----|------|-------------|
| `message_count` | `int` | Nombre total de messages envoyés |
| `avg_message_length` | `float` | Longueur moyenne en nombre de tokens par message |
| `activity_hours` | `list[int]` | Les 3 heures les plus actives |
| `most_active_day` | `str` | Jour de la semaine le plus actif |
| `top_topics` | `list[str]` | Jusqu'à 3 thèmes les plus fréquents |
| `sentiment_mean` | `float \| None` | Score de sentiment moyen, ou `None` si non calculé |

**Méthodes statiques :**

**`UserAnalyzer.summary_for(author, results)`** — Retourne le dictionnaire de profil d'un auteur donné (dict vide si introuvable).

**`UserAnalyzer.topics_for(author, results)`** — Retourne un DataFrame des affectations de thèmes pour un auteur.

**`UserAnalyzer.sentiment_over_time_for(author, results)`** — Retourne un DataFrame avec les colonnes `timestamp` et `sentiment_score` pour un auteur.

**`UserAnalyzer.activity_heatmap_for(author, results)`** — Retourne un DataFrame de carte de chaleur 7×24 pour un auteur.

---

### 6.8 MediaAnalyzer

```python
from whatsapp_analyzer.media_analyzer import MediaAnalyzer
```

#### `MediaAnalyzer`

Pas de paramètres de constructeur.

**`MediaAnalyzer().analyze(media_dir)`**

Analyse un dossier de médias, calcule des statistiques de fichiers et transcrit optionnellement les audio/vidéo.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `media_dir` | `Path` | Dossier contenant les fichiers médias WhatsApp |

Retourne : `dict` avec les clés :

| Clé | Type | Description |
|-----|------|-------------|
| `stats` | `pd.DataFrame` | Colonnes : `file_type`, `count`, `total_size_mb` |
| `transcriptions` | `pd.DataFrame` | Colonnes : `file_path`, `text` (vide si Whisper n'est pas installé) |

**Extensions supportées :** `.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.opus`, `.ogg`, `.mp3`

La transcription nécessite l'extra `[media]` (`pip install -e ".[media]"`).

---

### 6.9 GroupComparator

```python
from whatsapp_analyzer.comparator import GroupComparator
```

#### `GroupComparator`

**Paramètres du constructeur :**

| Paramètre | Type | Description |
|-----------|------|-------------|
| `analyzers` | `list[WhatsAppAnalyzer]` | Groupes analysés à comparer |

**`GroupComparator().compare_activity()`**

Retourne une ligne de résumé d'activité par groupe.

Retourne : `pd.DataFrame` avec les noms de groupes comme index et les colonnes :
`nb_messages`, `nb_participants`, `msgs_per_day`, `period_start`, `period_end`.

**`GroupComparator().compare_topics()`**

Retourne un tableau croisé dynamique des poids de thèmes par groupe.

Retourne : `pd.DataFrame` avec les noms de groupes comme index, les libellés de thèmes comme colonnes, et les poids comme valeurs.

**`GroupComparator().compare_sentiment()`**

Retourne une ligne de résumé de sentiment par groupe.

Retourne : `pd.DataFrame` avec les noms de groupes comme index et les colonnes :
`sentiment_mean`, `pos_pct`, `neg_pct`.

**`GroupComparator().common_users()`**

Retourne les auteurs présents dans plus d'un groupe.

Retourne : `pd.DataFrame` avec les colonnes `author` et `groups` (liste des noms de groupes).

**`GroupComparator().report(output)`**

Génère un rapport HTML de comparaison multi-groupes.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `output` | `Path` | Dossier de destination |

Retourne : `Path` vers `comparison_report.html`.

---

### 6.10 Visualizer

```python
from whatsapp_analyzer.visualizer import Visualizer
```

#### `Visualizer`

Pas de paramètres de constructeur.

Toutes les méthodes `plot_*` retournent une `matplotlib.figure.Figure` et **n'écrivent pas de fichiers**. Passez la figure à `st.pyplot()` (Streamlit) ou sauvegardez-la avec `fig.savefig()`.

**`Visualizer().plot_topic_distribution(results)`** — Diagramme en barres des poids de thèmes.

**`Visualizer().plot_wordcloud(results, topic_id)`** — Nuage de mots pour un thème spécifique.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | Résultats du pipeline |
| `topic_id` | `int` | Index du thème à visualiser |

**`Visualizer().plot_sentiment_timeline(results)`** — Sentiment moyen glissant dans le temps.

**`Visualizer().plot_user_activity(results)`** — Diagramme en barres horizontal du nombre de messages par utilisateur.

**`Visualizer().plot_hourly_heatmap(results)`** — Carte de chaleur seaborn (jour × heure).

**`Visualizer().generate_report(results, output_dir)`**

Écrit un rapport HTML autonome.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | Dictionnaire des résultats du pipeline |
| `output_dir` | `Path` | Dossier où `report.html` sera écrit |

Retourne : `Path` vers `report.html`. Crée le dossier s'il n'existe pas.

Toutes les images sont encodées en base64 PNG — le fichier s'ouvre hors ligne sans CDN.

**`Visualizer().generate_comparison_report(comparison_data, output_dir)`**

Écrit un rapport HTML de comparaison multi-groupes.

Retourne : `Path` vers `comparison_report.html`.

---

### 6.11 WhatsAppAnalyzer (core)

```python
from whatsapp_analyzer import WhatsAppAnalyzer
```

L'orchestrateur principal. Tous les imports de sous-modules sont paresseux — aucun code NLP ne s'exécute à la construction.

**Paramètres du constructeur :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `input_path` | `str \| Path` | — | Chemin vers `.zip`, `_chat.txt` ou dossier |
| `n_topics` | `int` | `5` | Nombre de thèmes LDA |
| `lang` | `str \| None` | `None` | Forcer la langue (`'fr'`, `'en'`). `None` = auto-détection |
| `min_words` | `int` | `3` | Nombre minimal de tokens après nettoyage |
| `output_dir` | `str \| Path` | `'reports'` | Dossier de sortie par défaut |

**Méthodes fluentes — doivent être appelées dans l'ordre :**

| Méthode | Description | Retourne |
|---------|-------------|---------|
| `.parse()` | Chargement + analyse de l'export | `self` |
| `.clean(lang=None, min_words=None)` | Applique le prétraitement NLP | `self` |
| `.analyze(topics=True, sentiment=True, temporal=True, media=False)` | Lance tous les modules d'analyse | `self` |
| `.report(output=None)` | Génère le rapport HTML | `Path` |
| `.to_csv(output=None)` | Exporte le DataFrame enrichi en CSV | `Path` |

**Autres méthodes :**

| Méthode | Description | Retourne |
|---------|-------------|---------|
| `.run()` | Raccourci : parse → clean → analyze → report | `dict` (résultats) |
| `.user(author)` | Obtenir un `UserView` pour un auteur spécifique | `UserView` |

**Tolérance aux pannes :** chaque étape d'analyse (topics, sentiment, temporal, media, users) est enveloppée dans un try/except. Une étape qui échoue enregistre un avertissement et met sa clé de résultat à `None` — le reste du pipeline continue.

---

### 6.12 UserView

Retourné par `WhatsAppAnalyzer.user(author)`. Vue ciblée sur un seul auteur.

| Méthode | Retourne | Description |
|---------|---------|-------------|
| `.summary()` | `dict` | Dictionnaire de profil complet pour cet auteur |
| `.topics()` | `pd.DataFrame` | DataFrame des affectations de thèmes |
| `.sentiment_over_time()` | `pd.DataFrame` | Scores de sentiment avec horodatages |
| `.activity_heatmap()` | `pd.DataFrame` | Carte de chaleur d'activité 7×24 |

---

### 6.13 Commandes CLI

Le point d'entrée est `whatsapp-analyzer` (installé avec le package).

#### `analyze`

```
whatsapp-analyzer analyze [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--input CHEMIN` | obligatoire | — | Chemin vers `.zip`, `_chat.txt` ou dossier |
| `--topics INT` | optionnel | `5` | Nombre de thèmes LDA |
| `--output DOSSIER` | optionnel | `reports` | Dossier de destination pour `report.html` |

#### `compare`

```
whatsapp-analyzer compare [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--input CHEMIN` | répétable | — | Un chemin par groupe (répéter le flag pour chaque groupe) |
| `--output DOSSIER` | optionnel | `reports` | Dossier de destination pour `comparison_report.html` |

#### `serve`

```
whatsapp-analyzer serve
```

Lance l'interface web Streamlit sur `http://localhost:8501`. Pas d'options.

---

## 7. Référence des sorties

### Clés du dictionnaire `results`

Le dictionnaire `results` retourné par `WhatsAppAnalyzer.run()` ou rempli via l'API fluente :

| Clé | Type | Remplie par |
|-----|------|-------------|
| `group_name` | `str` | `parse()` |
| `df_raw` | `pd.DataFrame` | `parse()` |
| `df_clean` | `pd.DataFrame` | `clean()` |
| `topics` | `dict \| None` | `analyze(topics=True)` |
| `sentiment` | `dict \| None` | `analyze(sentiment=True)` |
| `temporal` | `dict \| None` | `analyze(temporal=True)` |
| `media` | `dict \| None` | `analyze(media=True)` |
| `users` | `dict[str, dict] \| None` | `analyze()` (toujours) |
| `report_path` | `Path` | `run()` uniquement |

### Résumé des colonnes du DataFrame

**Après `parse()` :**
`timestamp`, `author`, `message`, `msg_type`, `group_name`

**Après `clean()` :**
tout ce qui précède + `cleaned_message`, `language`, `tokens`

**Après `analyze(topics=True)` :**
tout ce qui précède + `topic_id`, `topic_score`

**Après `analyze(sentiment=True)` :**
tout ce qui précède + `sentiment_score`, `sentiment_label`

---

## 8. Fonctionnalités optionnelles

| Extra | Package(s) installé(s) | Fonctionnalité débloquée |
|-------|------------------------|--------------------------|
| `[bertopic]` | `bertopic`, `sentence-transformers` | Modélisation de thèmes BERTopic |
| `[camembert]` | `transformers`, `torch` | Analyse de sentiment CamemBERT en français |
| `[media]` | `openai-whisper` | Transcription audio/vidéo |

Installer plusieurs extras simultanément :

```bash
pip install -e ".[bertopic,camembert,media]"
```

---

## 9. Confidentialité et aspects juridiques

- **100 % local.** Aucun contenu de message n'est envoyé à un serveur externe.
- **Pas d'automatisation de WhatsApp.** Les exports sont générés manuellement par l'utilisateur depuis l'application WhatsApp. Ce package n'automatise pas l'interface WhatsApp.
- **Anonymisation** disponible via `Parser(anonymize=True)` ou la case à cocher dans l'interface web. Les noms d'auteurs sont remplacés par des hachages SHA-256.
- **Stockage des données.** Les exports réels doivent être placés dans `data/raw/` (gitignored). Ne jamais commiter de vraies données WhatsApp.
- **Usage prévu.** Analyse personnelle ou académique de conversations auxquelles vous participez. Les données brutes de conversation ne doivent pas être redistribuées.
- **Conditions d'utilisation.** Conforme aux Conditions d'utilisation de WhatsApp (Meta).

---

## 10. Résolution des problèmes courants

### Modèle spaCy introuvable

```
OSError: [E050] Can't find model 'fr_core_news_sm'.
```

**Solution :** Le modèle est téléchargé automatiquement au premier lancement. En cas d'échec :
```bash
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

### `vader_lexicon` NLTK introuvable

Le lexique est téléchargé automatiquement lors du premier appel à `SentimentAnalyzer` avec `method='vader'`. En cas d'absence de connexion internet :

```python
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
```

### BERTopic ou CamemBERT non installé

```
RuntimeError: CamemBERT requires the [camembert] extra.
```

**Solution :**
```bash
pip install -e ".[camembert]"
```

### `No messages could be parsed`

Le fichier n'est pas un export WhatsApp valide, ou il utilise un format de date non supporté. Vérifiez que le fichier a bien été exporté directement depuis WhatsApp et qu'il contient des lignes correspondant au format Android ou iOS décrit dans la [section 4](#4-formats-dentrée).

### Rapport vide / graphiques manquants

Une ou plusieurs étapes d'analyse ont peut-être échoué silencieusement. Consultez la sortie des logs (avec `logging.basicConfig(level=logging.WARNING)`) pour voir quelles étapes ont été ignorées et pourquoi.
