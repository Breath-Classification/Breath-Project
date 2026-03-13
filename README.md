# Breath-Project
xhost +local:docker
pip install -e .  -uzywane aby byly zaleznosci miedzy plikami


## Tworzenie venva graficznego
python -m venv venv_streamlit --system-site-packages
source venv_streamlit/bin/activate
pip install --upgrade pip
pip install streamlit
streamlit run local_app.py --server.address=0.0.0.0 --server.port=8501
ctr+c 
deactivate

1) GRU
2) LSTM
3) stacked LSTM 2 warstwy
4) CNN+LSTM
5) Attention LSTM
6) Bidirectional LSTM

pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

✓ sliding window feature extraction
✓ temporal windowing
✓ frame stacking
✓ block-wise representation
✓ context windows
✓ local temporal embedding
✓ multi-step input for RNN

# Konwersja pliku `timestamp,value` do formatu akceptowanego przez model

Ten projekt używa **dwóch formatów pośrednich** zanim dane trafią do modelu:

1. surowy CSV: `seconds,data`
2. plik „pretrained” (punktowy): `value,label,time`
3. plik sekwencyjny do treningu: `x1,x2,...,feature,label`

Poniżej masz kroki bez zmieniania kodu funkcji.

---

## 1) Przygotuj wejściowy plik TXT

Masz już format:

```text
2026-03-12T18:01:19.319Z,604679
2026-03-12T18:01:19.437Z,601156
...
```

To jest poprawny input dla skryptu `scripts/convert_timestamps.py`, który:
- parsuje ISO timestamp,
- liczy czas od startu nagrania,
- zapisuje wynik jako CSV z kolumnami `seconds,data`.

**Dlaczego?**
Model i dalsze skrypty (`load_raw_data`) oczekują czasu liczbowego (sekundy), a nie daty ISO.

---

## 2) Umieść plik w odpowiednim folderze `data/raw`

Skrypt `convert_timestamps.py` przetwarza pliki `.txt` z:
- `data/raw/tens/`
- `data/raw/acc/`

Czyli:
- jeśli to tensometr: wrzuć np. `tens_mojpomiar.txt` do `data/raw/tens/`
- jeśli akcelerometr: wrzuć np. `acc_mojpomiar.txt` do `data/raw/acc/`

**Ważne nazewnictwo:**
Dalsze skrypty rozpoznają sensor po nazwie pliku (`"tens"` vs `"acc"`).

---

## 3) Uruchom konwersję timestampów

Z katalogu:

```bash
cd Breath_dataset_and_apps/code/brp-ml-model-main/scripts
python convert_timestamps.py
```

Efekt:
- z `*.txt` zrobi `*.csv` z nagłówkiem `seconds,data`
- usunie oryginalny `*.txt`

**Dlaczego?**
`scripts/load_data.py` (`load_raw_data`) czyta właśnie taki CSV (pierwsza kolumna = czas, druga = sygnał).

---

## 4) Wygeneruj plik „pretrained” (`value,label,time`)

Uruchom:

```bash
cd Breath_dataset_and_apps/code/brp-ml-model-main/scripts
python categorise_automatically.py
```

Co robi skrypt:
- czyta `data/raw/<sensor>/*.csv`,
- nakłada filtr (`moving_average`) i normalizację (`normalize`),
- generuje etykiety monotoniczności,
- zapisuje do `data/pretrained/<sensor>/<plik>.txt` w formacie:

```text
<znormalizowana_wartosc>,<etykieta>,<czas_w_s>
```

**Dlaczego?**
To format używany przez skrypty etykietowania/manualnej korekty (`labelling.py`) oraz późniejsze tworzenie sekwencji.

---

## 5) (Opcjonalnie, ale zalecane) Popraw etykiety ręcznie

Jeśli etykiety automatyczne są niedokładne:

```bash
cd Breath_dataset_and_apps/code/brp-ml-model-main/scripts
python labelling.py
```

W skrypcie ustawiasz `FILENAME`, np. `"normal"`, żeby otworzyć:
- `../data/pretrained/tens/tens_<FILENAME>.txt`
- `../data/pretrained/acc/acc_<FILENAME>.txt`

**Dlaczego?**
Jakość etykiet bezpośrednio wpływa na jakość modelu.

---

## 6) Zbuduj finalny plik sekwencji dla modelu

Uruchom:

```bash
cd Breath_dataset_and_apps/code/brp-ml-model-main
python scripts/load_data.py
```

Domyślnie skrypt wywoła:
- `prepare_data_for_training(SensorType.TENSOMETER)`

To tworzy pliki w:
- `data/pretrained/tens_sequence/`

Najważniejszy plik treningowy:
- `data/pretrained/tens_sequence/tens_concatenated.txt`

Format jednej linii:

```text
x1,x2,...,xN,rozstep,label
```

gdzie `label` jest na końcu.

**Dlaczego?**
`models/Keras/Models/AbstractModel.py` podczas `load_data(...)` zakłada, że **ostatnia kolumna to etykieta** i taki plik jest bezpośrednio ładowany do treningu.

---

## 7) Sprawdź zgodność przed treningiem

Szybki check:

```bash
head -n 3 Breath_dataset_and_apps/code/brp-ml-model-main/data/pretrained/tens_sequence/tens_concatenated.txt
```

Oczekujesz:
- wartości rozdzielone przecinkami,
- ostatnia wartość = klasa/etykieta,
- brak timestampów ISO.

---

## Minimalny pipeline (skrót)

1. `raw txt (ISO,value)` → do `data/raw/<sensor>/`
2. `python scripts/convert_timestamps.py`
3. `python scripts/categorise_automatically.py`
4. (opcjonalnie) `python scripts/labelling.py`
5. `python scripts/load_data.py`
6. trenuj model na `data/pretrained/<sensor>_sequence/<sensor>_concatenated.txt`

---

## Uwaga praktyczna

W `categorise_automatically.py` jest stałe `SENSOR_NAME = "acc"`. Jeśli pracujesz na tensometrze, uruchom analogicznie dla `tens` (bez zmiany logiki funkcji — tylko odpowiednia konfiguracja uruchomienia/pliku).
