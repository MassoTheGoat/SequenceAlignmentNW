"""
================================================================================
ALGORITMO DI NEEDLEMAN-WUNSCH — Allineamento Globale di Sequenze
================================================================================

Paradigma: Programmazione Dinamica
Complessità temporale: O(n * m)   dove n = |seq1|, m = |seq2|
Complessità spaziale: O(n * m)

================================================================================
PSEUDOCODICE
================================================================================

NEEDLEMAN-WUNSCH(seq1, seq2, match, mismatch, gap)

  Input:
    - seq1, seq2: le due sequenze da allineare
    - match:      punteggio per caratteri uguali      (es. +1)
    - mismatch:   penalità per caratteri diversi       (es. -1)
    - gap:        penalità per inserzione/cancellazione (es. -2)

  Output:
    - score:   valore della soluzione ottima
    - align1:  prima sequenza allineata (con eventuali gap '-')
    - align2:  seconda sequenza allineata (con eventuali gap '-')

  --- FASE 1: Inizializzazione ---

  n ← lunghezza(seq1)
  m ← lunghezza(seq2)

  // Creare matrice di scoring F di dimensione (n+1) x (m+1)
  F[0][0] ← 0

  PER i DA 1 A n:
      F[i][0] ← i * gap          // allineare seq1[1..i] con stringa vuota

  PER j DA 1 A m:
      F[0][j] ← j * gap          // allineare stringa vuota con seq2[1..j]

  --- FASE 2: Riempimento (Fill) ---

  PER i DA 1 A n:
      PER j DA 1 A m:

          SE seq1[i] == seq2[j]:
              diag ← F[i-1][j-1] + match
          ALTRIMENTI:
              diag ← F[i-1][j-1] + mismatch

          su   ← F[i-1][j] + gap      // gap in seq2 (cancellazione)
          sx   ← F[i][j-1] + gap      // gap in seq1 (inserzione)

          F[i][j] ← max(diag, su, sx)

  score ← F[n][m]   // valore della soluzione ottima

  --- FASE 3: Traceback ---

  // Ricostruire l'allineamento ottimo partendo da F[n][m]
  // e risalendo fino a F[0][0]

  align1 ← ""
  align2 ← ""
  i ← n
  j ← m

  MENTRE i > 0 E j > 0:

      SE seq1[i] == seq2[j]:
          s ← match
      ALTRIMENTI:
          s ← mismatch

      SE F[i][j] == F[i-1][j-1] + s:        // mossa diagonale
          align1 ← seq1[i] + align1
          align2 ← seq2[j] + align2
          i ← i - 1
          j ← j - 1

      ALTRIMENTI SE F[i][j] == F[i-1][j] + gap:  // mossa verso l'alto
          align1 ← seq1[i] + align1
          align2 ← '-' + align2
          i ← i - 1

      ALTRIMENTI:                                  // mossa verso sinistra
          align1 ← '-' + align1
          align2 ← seq2[j] + align2
          j ← j - 1

  // Gestire i caratteri rimanenti
  MENTRE i > 0:
      align1 ← seq1[i] + align1
      align2 ← '-' + align2
      i ← i - 1

  MENTRE j > 0:
      align1 ← '-' + align1
      align2 ← seq2[j] + align2
      j ← j - 1

  RESTITUIRE score, align1, align2

================================================================================
"""

import numpy as np
from tabulate import tabulate


# ──────────────────────────────────────────────────────────────────────────────
# FUNZIONE PRINCIPALE: Needleman-Wunsch
# ──────────────────────────────────────────────────────────────────────────────

def needleman_wunsch(seq1: str, seq2: str,
                     match: int = 1,
                     mismatch: int = -1,
                     gap: int = -2) -> tuple:
    """
    Implementa l'algoritmo di Needleman-Wunsch per l'allineamento globale
    di due sequenze usando il paradigma della programmazione dinamica.

    Parametri
    ----------
    seq1 : str
        Prima sequenza da allineare.
    seq2 : str
        Seconda sequenza da allineare.
    match : int
        Punteggio assegnato quando i caratteri corrispondono (default: +1).
    mismatch : int
        Penalità per caratteri diversi (default: -1).
    gap : int
        Penalità per gap / indel (default: -2).

    Restituisce
    -----------
    tuple : (score, align1, align2, F)
        score  - valore ottimo dell'allineamento
        align1 - prima sequenza allineata
        align2 - seconda sequenza allineata
        F      - matrice di scoring completa
    """

    n = len(seq1)
    m = len(seq2)

    # ── FASE 1: Inizializzazione della matrice di scoring ──────────────────
    # F[i][j] rappresenta il punteggio ottimo per allineare
    # seq1[0..i-1] con seq2[0..j-1]
    # Nota: si usano liste Python anziché np.array per prestazioni migliori
    # (l'accesso scalare a liste è ~5-10× più veloce di numpy in loop Python)

    F = [[0] * (m + 1) for _ in range(n + 1)]

    # Prima colonna: allineare seq1[0..i-1] con stringa vuota → solo gap
    for i in range(1, n + 1):
        F[i][0] = i * gap

    # Prima riga: allineare stringa vuota con seq2[0..j-1] → solo gap
    for j in range(1, m + 1):
        F[0][j] = j * gap

    # ── FASE 2: Riempimento (Fill) ────────────────────────────────────────
    # Per ogni cella F[i][j], calcolare il massimo tra:
    #   - diag: proviene da F[i-1][j-1] + score(seq1[i-1], seq2[j-1])
    #   - su:   proviene da F[i-1][j]   + gap  (gap in seq2)
    #   - sx:   proviene da F[i][j-1]   + gap  (gap in seq1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = F[i - 1][j - 1] + match
            else:
                diag = F[i - 1][j - 1] + mismatch

            su = F[i - 1][j] + gap    # gap in seq2 (cancellazione)
            sx = F[i][j - 1] + gap    # gap in seq1 (inserzione)

            F[i][j] = max(diag, su, sx)

    # Il valore della soluzione ottima si trova in F[n][m]
    score = F[n][m]

    # ── FASE 3: Traceback ─────────────────────────────────────────────────
    # Ricostruire l'allineamento ottimo risalendo dalla cella F[n][m]
    # fino alla cella F[0][0], seguendo le scelte ottimali.

    align1 = ""
    align2 = ""
    i, j = n, m

    while i > 0 and j > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch

        if F[i][j] == F[i - 1][j - 1] + s:
            # Mossa diagonale: i caratteri sono allineati tra loro
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
        elif F[i][j] == F[i - 1][j] + gap:
            # Mossa verso l'alto: gap in seq2
            align1 = seq1[i - 1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            # Mossa verso sinistra: gap in seq1
            align1 = "-" + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    # Gestire i caratteri rimanenti (bordi della matrice)
    while i > 0:
        align1 = seq1[i - 1] + align1
        align2 = "-" + align2
        i -= 1
    while j > 0:
        align1 = "-" + align1
        align2 = seq2[j - 1] + align2
        j -= 1

    return score, align1, align2, F


# ──────────────────────────────────────────────────────────────────────────────
# FUNZIONI DI VISUALIZZAZIONE
# ──────────────────────────────────────────────────────────────────────────────

def stampa_matrice(F, seq1: str, seq2: str) -> None:
    """Stampa la matrice di scoring in formato tabella leggibile."""

    # Intestazioni colonna: ε (stringa vuota) + caratteri di seq2
    col_headers = ["", "ε"] + list(seq2)

    # Costruire le righe con intestazione riga
    righe = []
    row_labels = ["ε"] + list(seq1)

    n_righe = len(F)
    n_colonne = len(F[0]) if n_righe > 0 else 0

    for i in range(n_righe):
        riga = [row_labels[i]] + [int(F[i][j]) for j in range(n_colonne)]
        righe.append(riga)

    print("\n📊 Matrice di Scoring F:")
    print(tabulate(righe, headers=col_headers, tablefmt="fancy_grid",
                   stralign="center", numalign="center"))


def stampa_allineamento(align1: str, align2: str, score: int) -> None:
    """Stampa l'allineamento risultante in formato leggibile."""

    # Costruire la riga di matching: | per match, x per mismatch, spazio per gap
    match_line = ""
    for a, b in zip(align1, align2):
        if a == b:
            match_line += "|"
        elif a == "-" or b == "-":
            match_line += " "
        else:
            match_line += "x"

    print(f"\n🧬 Allineamento Globale Ottimo (score = {score}):\n")
    print(f"  Seq1: {align1}")
    print(f"        {match_line}")
    print(f"  Seq2: {align2}")


def stampa_sottostruttura_ottima(F, seq1: str, seq2: str) -> None:
    """
    Mostra la proprietà di sottostruttura ottima della programmazione dinamica:
    il valore ottimo F[i][j] dipende solo dai sottoproblemi già risolti
    F[i-1][j-1], F[i-1][j], F[i][j-1].
    """
    print("\n📐 Proprietà di Sottostruttura Ottima:")
    print("   F[i][j] = max(")
    print("       F[i-1][j-1] + score(seq1[i], seq2[j]),  // diagonale")
    print("       F[i-1][j]   + gap,                       // dall'alto")
    print("       F[i][j-1]   + gap                        // da sinistra")
    print("   )")
    print(f"\n   Valore soluzione ottima: F[{len(seq1)}][{len(seq2)}] = {F[len(seq1)][len(seq2)]}")


# ──────────────────────────────────────────────────────────────────────────────
# ESECUZIONE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("  NEEDLEMAN-WUNSCH — Allineamento Globale di Sequenze")
    print("  Paradigma: Programmazione Dinamica")
    print("=" * 65)

    # ── Esempio 1: Sequenze biologiche (DNA) ──────────────────────────────
    seq1 = "GCATGCG"
    seq2 = "GATTACA"

    # Parametri di scoring
    MATCH    =  1    # premio per caratteri uguali
    MISMATCH = -1    # penalità per caratteri diversi
    GAP      = -2    # penalità per gap (indel)

    print(f"\n  Sequenza 1: {seq1}")
    print(f"  Sequenza 2: {seq2}")
    print(f"  Match: {MATCH}  |  Mismatch: {MISMATCH}  |  Gap: {GAP}")

    # Eseguire l'algoritmo
    score, align1, align2, F = needleman_wunsch(
        seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )

    # Visualizzare i risultati
    stampa_matrice(F, seq1, seq2)
    stampa_sottostruttura_ottima(F, seq1, seq2)
    stampa_allineamento(align1, align2, score)

    # ── Esempio 2: Sequenze più corte per chiarezza ───────────────────────
    print("\n" + "=" * 65)
    print("  Esempio 2: Sequenze brevi")
    print("=" * 65)

    seq1_b = "AGTC"
    seq2_b = "ATC"

    print(f"\n  Sequenza 1: {seq1_b}")
    print(f"  Sequenza 2: {seq2_b}")
    print(f"  Match: {MATCH}  |  Mismatch: {MISMATCH}  |  Gap: {GAP}")

    score_b, align1_b, align2_b, F_b = needleman_wunsch(
        seq1_b, seq2_b, match=MATCH, mismatch=MISMATCH, gap=GAP
    )

    stampa_matrice(F_b, seq1_b, seq2_b)
    stampa_sottostruttura_ottima(F_b, seq1_b, seq2_b)
    stampa_allineamento(align1_b, align2_b, score_b)

    print("\n" + "=" * 65)
    print("  ✅ Algoritmo completato con successo")
    print("=" * 65)
