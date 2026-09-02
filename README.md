# Progetto Algoritmi — Needleman-Wunsch

## Descrizione del Progetto

Implementazione e analisi sperimentale dell'algoritmo di **Needleman-Wunsch** per
l'allineamento globale di sequenze biologiche (DNA), con studio di tre varianti:

| Algoritmo             | Tempo  | Spazio      | Output               |
| --------------------- | ------ | ----------- | -------------------- |
| **NW Base**           | O(n·m) | O(n·m)      | Score + Allineamento |
| **NW Spazio Lineare** | O(n·m) | O(min(n,m)) | Solo Score           |
| **Hirschberg**        | O(n·m) | O(min(n,m)) | Score + Allineamento |

Le tre varianti sono implementate sia in **Python** sia in **Cython** (compilato in C)
per ottenere uno speed-up significativo (~50-100×) nella fase di sperimentazione.

---

## Struttura dei File

```
.
├── needleman_wunsch.py              # Implementazione base NW (Python puro)
├── needleman_wunsch_ottimizzato.py   # Varianti ottimizzate: Spazio Lineare + Hirschberg (Python puro)
├── nw_core.pyx                      # Stesse implementazioni in Cython (compilate in C)
├── setup_cython.py                  # Script di build per compilare il modulo Cython
├── sperimentazione.py               # Analisi empirica: benchmark di tempo e memoria + grafici
├── requirements.txt                 # Dipendenze Python del progetto
├── nw_core.cpython-*.so             # Modulo Cython compilato (eseguibile)
├── grafico_tempo_tutti.png          # Grafico: confronto tempi (3 algoritmi)
├── grafico_memoria_tutti.png        # Grafico: confronto memoria (3 algoritmi)
├── grafico_rapporto_asintotico.png  # Grafico: verifica T(n)/n² → costante
└── grafico_scalabilita_ottimizzati.png  # Grafico: scalabilità varianti ottimizzate
```

---

## Ordine di Esecuzione

### Prerequisiti

- Python 3.10+
- Compilatore C (gcc)
- Cython

### Passo 1 — Installare le dipendenze

```bash
pip install -r requirements.txt
pip install cython
```

### Passo 2 — Compilare il modulo Cython

```bash
python setup_cython.py build_ext --inplace
```

Questo genera il file `nw_core.cpython-*.so`, ovvero il modulo Cython compilato
che viene poi importato dallo script di sperimentazione.

### Passo 3 — Eseguire l'algoritmo base (Python puro)

```bash
python needleman_wunsch.py
```

Esegue l'algoritmo NW base su due coppie di sequenze di esempio,
mostrando la matrice di scoring, la sottostruttura ottima e l'allineamento.

### Passo 4 — Eseguire le varianti ottimizzate (Python puro)

```bash
python needleman_wunsch_ottimizzato.py
```

Esegue le varianti NW Spazio Lineare e Hirschberg, verificando la correttezza
rispetto alla versione base e confrontando le prestazioni.

### Passo 5 — Eseguire la sperimentazione completa

```bash
python sperimentazione.py
```

Esegue il benchmark sistematico di tutti e tre gli algoritmi (versione Cython)
su dimensioni crescenti delle sequenze, misurando tempo e memoria.
Genera automaticamente i quattro grafici `.png` presenti nella cartella.

> **Nota:** la sperimentazione può richiedere diversi minuti, soprattutto
> per le istanze di dimensione 100.000.

---
