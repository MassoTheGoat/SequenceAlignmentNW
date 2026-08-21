"""
Algoritmo di Needleman-Wunsch — Allineamento Globale di Sequenze

Paradigma: Programmazione Dinamica
Complessità temporale: O(n·m)   dove n = |seq1|, m = |seq2|
Complessità spaziale: O(n·m)

L'algoritmo costruisce una matrice di scoring F di dimensione (n+1)×(m+1),
dove F[i][j] = score ottimo per allineare seq1[0..i-1] con seq2[0..j-1].

Ricorrenza:
  F[i][j] = max(
      F[i-1][j-1] + score(seq1[i], seq2[j]),   (diagonale)
      F[i-1][j]   + gap,                        (dall'alto)
      F[i][j-1]   + gap                         (da sinistra)
  )

L'allineamento si ricostruisce con traceback da F[n][m] a F[0][0].
"""

import numpy as np
from tabulate import tabulate


def needleman_wunsch(seq1: str, seq2: str,
                     match: int = 1, mismatch: int = -1,
                     gap: int = -2) -> tuple:
    """
    Allineamento globale di due sequenze con Needleman-Wunsch.

    Restituisce (score, align1, align2, F).
    """
    n = len(seq1)
    m = len(seq2)

    # FASE 1: Inizializzazione
    F = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        F[i][0] = i * gap
    for j in range(1, m + 1):
        F[0][j] = j * gap

    # FASE 2: Riempimento (Fill)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = F[i - 1][j - 1] + match
            else:
                diag = F[i - 1][j - 1] + mismatch

            su = F[i - 1][j] + gap    # gap in seq2
            sx = F[i][j - 1] + gap    # gap in seq1

            F[i][j] = max(diag, su, sx)

    score = F[n][m]

    # FASE 3: Traceback — ricostruzione dell'allineamento ottimo
    align1 = ""
    align2 = ""
    i, j = n, m

    while i > 0 and j > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch

        if F[i][j] == F[i - 1][j - 1] + s:       # diagonale
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
        elif F[i][j] == F[i - 1][j] + gap:        # dall'alto
            align1 = seq1[i - 1] + align1
            align2 = "-" + align2
            i -= 1
        else:                                       # da sinistra
            align1 = "-" + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    # Caratteri rimanenti (bordi della matrice)
    while i > 0:
        align1 = seq1[i - 1] + align1
        align2 = "-" + align2
        i -= 1
    while j > 0:
        align1 = "-" + align1
        align2 = seq2[j - 1] + align2
        j -= 1

    return score, align1, align2, F


# ── Funzioni di visualizzazione ──────────────────────────────────────────────

def stampa_matrice(F, seq1: str, seq2: str) -> None:
    """Stampa la matrice di scoring F in formato tabella."""
    col_headers = ["", "ε"] + list(seq2)
    row_labels = ["ε"] + list(seq1)
    n_righe = len(F)
    n_colonne = len(F[0]) if n_righe > 0 else 0

    righe = []
    for i in range(n_righe):
        riga = [row_labels[i]] + [int(F[i][j]) for j in range(n_colonne)]
        righe.append(riga)

    print("\n📊 Matrice di Scoring F:")
    print(tabulate(righe, headers=col_headers, tablefmt="fancy_grid",
                   stralign="center", numalign="center"))


def stampa_allineamento(align1: str, align2: str, score: int) -> None:
    """Stampa l'allineamento in formato leggibile."""
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
    """Mostra la ricorrenza di programmazione dinamica."""
    print("\n📐 Proprietà di Sottostruttura Ottima:")
    print("   F[i][j] = max(")
    print("       F[i-1][j-1] + score(seq1[i], seq2[j]),  // diagonale")
    print("       F[i-1][j]   + gap,                       // dall'alto")
    print("       F[i][j-1]   + gap                        // da sinistra")
    print("   )")
    print(f"\n   Valore soluzione ottima: F[{len(seq1)}][{len(seq2)}] = {F[len(seq1)][len(seq2)]}")


# ── Esecuzione principale ───────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("  NEEDLEMAN-WUNSCH — Allineamento Globale di Sequenze")
    print("  Paradigma: Programmazione Dinamica")
    print("=" * 65)

    # Esempio 1
    seq1 = "GCATGCG"
    seq2 = "GATTACA"
    MATCH, MISMATCH, GAP = 1, -1, -2

    print(f"\n  Sequenza 1: {seq1}")
    print(f"  Sequenza 2: {seq2}")
    print(f"  Match: {MATCH}  |  Mismatch: {MISMATCH}  |  Gap: {GAP}")

    score, align1, align2, F = needleman_wunsch(
        seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    stampa_matrice(F, seq1, seq2)
    stampa_sottostruttura_ottima(F, seq1, seq2)
    stampa_allineamento(align1, align2, score)

    # Esempio 2
    print("\n" + "=" * 65)
    print("  Esempio 2: Sequenze brevi")
    print("=" * 65)

    seq1_b, seq2_b = "AGTC", "ATC"
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
