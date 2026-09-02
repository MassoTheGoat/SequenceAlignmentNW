import numpy as np
from tabulate import tabulate
from needleman_wunsch import needleman_wunsch, stampa_allineamento


# ── Variante 1: Score in Spazio Lineare ──────────────────────────────────────

def nw_score_lineare(seq1: str, seq2: str,
                     match: int = 1, mismatch: int = -1,
                     gap: int = -2) -> int:
   
    # Righe lungo la sequenza più corta → spazio O(min(n,m))
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    n, m = len(seq1), len(seq2)

    prev_row = [j * gap for j in range(m + 1)]
    curr_row = [0] * (m + 1)

    for i in range(1, n + 1):
        curr_row[0] = i * gap
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = prev_row[j - 1] + match
            else:
                diag = prev_row[j - 1] + mismatch
            su = prev_row[j] + gap
            sx = curr_row[j - 1] + gap
            curr_row[j] = max(diag, su, sx)
        prev_row, curr_row = curr_row, prev_row

    return prev_row[m]


# ── Funzione ausiliaria: ultima riga della matrice NW ────────────────────────

def _nw_ultima_riga(seq1: str, seq2: str,
                    match: int, mismatch: int, gap: int) -> list:
    """Calcola l'ultima riga della matrice NW in spazio O(m)."""
    n, m = len(seq1), len(seq2)

    prev_row = [j * gap for j in range(m + 1)]
    curr_row = [0] * (m + 1)

    for i in range(1, n + 1):
        curr_row[0] = i * gap
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = prev_row[j - 1] + match
            else:
                diag = prev_row[j - 1] + mismatch
            su = prev_row[j] + gap
            sx = curr_row[j - 1] + gap
            curr_row[j] = max(diag, su, sx)
        prev_row, curr_row = curr_row, prev_row

    return prev_row


# ── Variante 2: Hirschberg ───────────────────────────────────────────────────

def hirschberg(seq1: str, seq2: str,
               match: int = 1, mismatch: int = -1,
               gap: int = -2) -> tuple:

    def _hirschberg_ricorsivo(x: str, y: str) -> tuple:
        n, m = len(x), len(y)

        # Casi base
        if n == 0:
            return "-" * m, y
        if m == 0:
            return x, "-" * n

        # Per matrici 1 x m o n x 1 la matrice DP completa è minuscola (O(m) o O(n))
        # e non viola lo spazio lineare.
        if n == 1 or m == 1:
            _, a1, a2, _ = needleman_wunsch(x, y, match, mismatch, gap)
            return a1, a2

        # Divide: split di x a metà
        i_mid = n // 2
        score_sx = _nw_ultima_riga(x[:i_mid], y, match, mismatch, gap)
        score_dx = _nw_ultima_riga(x[i_mid:][::-1], y[::-1], match, mismatch, gap)

        # Trova j* = argmax { score_sx[j] + score_dx[m-j] }
        j_taglio = 0
        miglior_somma = score_sx[0] + score_dx[m]
        for j in range(1, m + 1):
            s = score_sx[j] + score_dx[m - j]
            if s > miglior_somma:
                miglior_somma = s
                j_taglio = j

        # Impera: ricorsione sui due sottoproblemi
        align1_sx, align2_sx = _hirschberg_ricorsivo(x[:i_mid], y[:j_taglio])
        align1_dx, align2_dx = _hirschberg_ricorsivo(x[i_mid:], y[j_taglio:])

        return align1_sx + align1_dx, align2_sx + align2_dx

    score = nw_score_lineare(seq1, seq2, match, mismatch, gap)
    align1, align2 = _hirschberg_ricorsivo(seq1, seq2)
    return score, align1, align2


# ── Tabella comparativa ──────────────────────────────────────────────────────

def stampa_confronto_complessita() -> None:
    print("\n" + "=" * 75)
    print("CONFRONTO DELLE COMPLESSITÀ")
    print("=" * 75)

    headers = ["Algoritmo", "Tempo", "Spazio", "Output"]
    dati = [
        ["NW Base",           "O(n·m)", "O(n·m)",      "Score + Allineamento"],
        ["NW Spazio Lineare", "O(n·m)", "O(min(n,m))", "Solo Score"],
        ["Hirschberg",        "O(n·m)", "O(min(n,m))", "Score + Allineamento"],
    ]
    print(tabulate(dati, headers=headers, tablefmt="fancy_grid",
                   stralign="center", numalign="center"))
    print("\nHirschberg ha costante ~2× nel tempo (ricorsione).\n")


def stampa_confronto_memoria(n: int, m: int) -> None:
    mem_base = (n + 1) * (m + 1) * 8
    mem_lineare = 2 * (min(n, m) + 1) * 8

    def formatta_byte(b: int) -> str:
        if b < 1024:       return f"{b} B"
        elif b < 1024**2:  return f"{b / 1024:.1f} KB"
        elif b < 1024**3:  return f"{b / (1024**2):.1f} MB"
        else:              return f"{b / (1024**3):.1f} GB"

    rapporto = mem_base / mem_lineare if mem_lineare > 0 else float('inf')
    print(f"\nConfronto memoria per n={n}, m={m}:")
    print(f"     • NW Base:        {formatta_byte(mem_base)}")
    print(f"     • Spazio lineare: {formatta_byte(mem_lineare)}")
    print(f"     • Riduzione:      {rapporto:.0f}×\n")


# ── Esecuzione principale ───────────────────────────────────────────────────

if __name__ == "__main__":

    import time

    print("=" * 75)
    print("  VARIANTI OTTIMIZZATE — NEEDLEMAN-WUNSCH")
    print("=" * 75)

    MATCH, MISMATCH, GAP = 1, -1, -2

    # TEST 1: Correttezza
    print("\n" + "─" * 75)
    print("  TEST 1: Verifica di correttezza")
    print("─" * 75)

    seq1, seq2 = "GCATGCG", "GATTACA"
    print(f"\n  Seq1: {seq1}  |  Seq2: {seq2}")
    print(f"  Match: {MATCH}  |  Mismatch: {MISMATCH}  |  Gap: {GAP}")

    score_base, a1_base, a2_base, _ = needleman_wunsch(seq1, seq2, MATCH, MISMATCH, GAP)
    score_lin = nw_score_lineare(seq1, seq2, MATCH, MISMATCH, GAP)
    score_hir, a1_hir, a2_hir = hirschberg(seq1, seq2, MATCH, MISMATCH, GAP)

    assert score_base == score_lin == score_hir
    print(f"\nScore: {score_base} (tutti concordi)")
    print(f"  NW Base:     {a1_base} / {a2_base}")
    print(f"  Hirschberg:  {a1_hir} / {a2_hir}")

    # TEST 2: Secondo esempio
    print("\n" + "─" * 75)
    print("  TEST 2: Secondo esempio")
    print("─" * 75)

    seq1_b, seq2_b = "AGTC", "ATC"
    sb, _, _, _ = needleman_wunsch(seq1_b, seq2_b, MATCH, MISMATCH, GAP)
    sl = nw_score_lineare(seq1_b, seq2_b, MATCH, MISMATCH, GAP)
    sh, ah1, ah2 = hirschberg(seq1_b, seq2_b, MATCH, MISMATCH, GAP)
    assert sb == sl == sh
    print(f"Score: {sb} (tutti concordi)")
    print(f"  Hirschberg: {ah1} / {ah2}")

    # TEST 3: Prestazioni
    print("\n" + "─" * 75)
    print("  TEST 3: Confronto prestazionale (n=500)")
    print("─" * 75)

    import random
    random.seed(42)
    L = 500
    s1 = "".join(random.choice("ACGT") for _ in range(L))
    s2 = "".join(random.choice("ACGT") for _ in range(L))

    t0 = time.perf_counter()
    sb_l, _, _, _ = needleman_wunsch(s1, s2, MATCH, MISMATCH, GAP)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    nw_score_lineare(s1, s2, MATCH, MISMATCH, GAP)
    t_lin = time.perf_counter() - t0

    t0 = time.perf_counter()
    hirschberg(s1, s2, MATCH, MISMATCH, GAP)
    t_hir = time.perf_counter() - t0

    print(f"\n  NW Base:     {t_base:.4f}s")
    print(f"  NW Lineare:  {t_lin:.4f}s")
    print(f"  Hirschberg:  {t_hir:.4f}s")

    stampa_confronto_complessita()
    stampa_confronto_memoria(L, L)
    stampa_confronto_memoria(10_000, 10_000)
    stampa_confronto_memoria(100_000, 100_000)

    # TEST 4: Grandi dimensioni (solo ottimizzati)
    print("─" * 75)
    print("  TEST 4: Grandi dimensioni (n=5000, solo ottimizzati)")
    print("─" * 75)

    L2 = 5_000
    s1g = "".join(random.choice("ACGT") for _ in range(L2))
    s2g = "".join(random.choice("ACGT") for _ in range(L2))

    print(f"\n  Base richiederebbe ~{L2**2 * 8 / (1024**2):.0f} MB")
    print(f"  Ottimizzati usano   ~{2 * L2 * 8 / 1024:.0f} KB")

    t0 = time.perf_counter()
    sg = nw_score_lineare(s1g, s2g, MATCH, MISMATCH, GAP)
    print(f"\nNW Lineare: score={sg}, tempo={time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    sgh, _, _ = hirschberg(s1g, s2g, MATCH, MISMATCH, GAP)
    print(f"Hirschberg: score={sgh}, tempo={time.perf_counter()-t0:.2f}s")

    assert sg == sgh
    print("\n" + "=" * 75)
    print("Tutti i test completati con successo")
    print("=" * 75)
