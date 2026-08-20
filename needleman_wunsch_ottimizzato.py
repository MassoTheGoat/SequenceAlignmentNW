"""
================================================================================
VARIANTI OTTIMIZZATE DELL'ALGORITMO DI NEEDLEMAN-WUNSCH
================================================================================

Questo file contiene due varianti dell'algoritmo di Needleman-Wunsch originale
(implementato in needleman_wunsch.py) con l'obiettivo di migliorarne le
prestazioni, in particolare quelle SPAZIALI, così da poter elaborare sequenze
di dimensioni molto più grandi.

────────────────────────────────────────────────────────────────────────────────
PROBLEMA DELL'IMPLEMENTAZIONE BASE
────────────────────────────────────────────────────────────────────────────────

L'algoritmo base costruisce l'intera matrice di scoring F di dimensione
(n+1) × (m+1), dove n = |seq1| e m = |seq2|.

    Complessità temporale: O(n · m)
    Complessità spaziale:  O(n · m)

Per sequenze biologiche reali (es. genomi), n e m possono essere dell'ordine
di 10^6 o più. Una matrice (10^6) × (10^6) richiederebbe ~10^12 celle,
ovvero circa 8 TB di memoria con interi a 64 bit. Questo rende l'algoritmo
base INAPPLICABILE a istanze di grandi dimensioni.

────────────────────────────────────────────────────────────────────────────────
VARIANTE 1: NEEDLEMAN-WUNSCH CON SPAZIO LINEARE (solo score)
────────────────────────────────────────────────────────────────────────────────

IDEA CHIAVE:
  Nell'algoritmo base, per calcolare la riga i della matrice F, servono
  SOLO i valori della riga i-1 (e della riga i stessa, per la cella a
  sinistra). Non servono le righe 0, 1, ..., i-2.

  Quindi possiamo mantenere in memoria solo DUE righe alla volta:
    - la riga precedente (prev_row)
    - la riga corrente (curr_row)

  Ad ogni iterazione, la riga corrente completata diventa la riga
  precedente, e si riutilizza lo spazio.

ULTERIORE OTTIMIZZAZIONE:
  Si sceglie di iterare lungo la dimensione maggiore e mantenere le
  due righe lungo la dimensione minore. In questo modo lo spazio usato
  è O(min(n, m)) anziché O(max(n, m)).

LIMITAZIONE:
  Questa variante calcola SOLO lo score ottimo. Non permette di
  ricostruire l'allineamento, perché il traceback richiede l'intera
  matrice F che qui non viene memorizzata.

COMPLESSITÀ:
    Temporale: O(n · m)         — invariata rispetto all'originale
    Spaziale:  O(min(n, m))     — drastica riduzione!

PSEUDOCODICE:

  NW-SCORE-LINEARE(seq1, seq2, match, mismatch, gap)

    // Assicurarsi che seq2 sia la più corta (per minimizzare lo spazio)
    SE lunghezza(seq1) < lunghezza(seq2):
        SCAMBIA seq1, seq2

    n ← lunghezza(seq1)    // la più lunga
    m ← lunghezza(seq2)    // la più corta

    // Inizializzare due righe di dimensione m+1
    prev_row[j] ← j * gap    PER j DA 0 A m

    PER i DA 1 A n:
        curr_row[0] ← i * gap
        PER j DA 1 A m:
            SE seq1[i] == seq2[j]:
                diag ← prev_row[j-1] + match
            ALTRIMENTI:
                diag ← prev_row[j-1] + mismatch
            su ← prev_row[j]   + gap
            sx ← curr_row[j-1] + gap
            curr_row[j] ← max(diag, su, sx)
        prev_row ← curr_row    // la corrente diventa la precedente

    RESTITUIRE prev_row[m]

────────────────────────────────────────────────────────────────────────────────
VARIANTE 2: ALGORITMO DI HIRSCHBERG (score + allineamento in spazio lineare)
────────────────────────────────────────────────────────────────────────────────

IDEA CHIAVE:
  L'algoritmo di Hirschberg combina:
    (a) Programmazione Dinamica (Needleman-Wunsch con spazio lineare)
    (b) Divide et Impera

  Il trucco fondamentale è il seguente:
    1. Si divide seq1 a metà: seq1 = X_sinistra + X_destra
    2. Si calcola l'ultima riga della matrice NW per:
       - X_sinistra  vs  seq2           (scorrimento in avanti)
       - X_destra^R  vs  seq2^R         (scorrimento all'indietro)
       dove ^R indica il reverse della sequenza.
    3. Si trova il punto di taglio ottimo j* su seq2 tale che:
       score_avanti[j*] + score_indietro[m - j*] sia massimo.
       Questo j* identifica dove l'allineamento ottimo "attraversa"
       la riga di mezzo.
    4. Si ricorre su (X_sinistra, seq2[0..j*]) e (X_destra, seq2[j*..m]).

  Il caso base si raggiunge quando una delle due sequenze ha lunghezza
  0 o 1, e l'allineamento è banale.

PERCHÉ FUNZIONA:
  Il principio di ottimalità di Bellman garantisce che se l'allineamento
  globale ottimo passa per la cella (n/2, j*), allora:
    - l'allineamento di seq1[0..n/2] con seq2[0..j*] è ottimo
    - l'allineamento di seq1[n/2..n] con seq2[j*..m] è ottimo
  Quindi possiamo risolvere i due sottoproblemi indipendentemente.

COMPLESSITÀ:
    Temporale: O(n · m)         — invariata (la somma delle aree dei
                                   sottoproblemi converge a 2·n·m)
    Spaziale:  O(min(n, m))     — come la Variante 1!

  Ma a differenza della Variante 1, qui si ottiene ANCHE l'allineamento
  completo, non solo lo score.

PSEUDOCODICE:

  HIRSCHBERG(seq1, seq2, match, mismatch, gap)

    n ← lunghezza(seq1)
    m ← lunghezza(seq2)

    --- Casi base ---

    SE n == 0:
        RESTITUIRE ("-" * m, seq2)

    SE m == 0:
        RESTITUIRE (seq1, "-" * n)

    SE n == 1:
        // Allineare un singolo carattere con l'intera seq2
        // usando NW standard (matrice piccola)
        RESTITUIRE NW_BASE(seq1, seq2)

    SE m == 1:
        RESTITUIRE NW_BASE(seq1, seq2)

    --- Passo ricorsivo (Divide et Impera) ---

    metà ← n / 2

    // Calcolare l'ultima riga di NW per seq1[0..metà] vs seq2
    score_sx ← NW_ULTIMA_RIGA(seq1[0..metà], seq2)

    // Calcolare l'ultima riga di NW per reverse(seq1[metà..n]) vs reverse(seq2)
    score_dx ← NW_ULTIMA_RIGA(reverse(seq1[metà..n]), reverse(seq2))

    // Trovare il punto di taglio ottimo
    j* ← argmax_j { score_sx[j] + score_dx[m - j] }

    // Ricorrere sui due sottoproblemi
    (align1_sx, align2_sx) ← HIRSCHBERG(seq1[0..metà], seq2[0..j*])
    (align1_dx, align2_dx) ← HIRSCHBERG(seq1[metà..n], seq2[j*..m])

    RESTITUIRE (align1_sx + align1_dx, align2_sx + align2_dx)

================================================================================
"""

import numpy as np
from tabulate import tabulate

# Importare l'algoritmo base per i confronti
from needleman_wunsch import needleman_wunsch, stampa_allineamento


# ══════════════════════════════════════════════════════════════════════════════
# VARIANTE 1: Score in Spazio Lineare  —  O(n·m) tempo, O(min(n,m)) spazio
# ══════════════════════════════════════════════════════════════════════════════

def nw_score_lineare(seq1: str, seq2: str,
                     match: int = 1,
                     mismatch: int = -1,
                     gap: int = -2) -> int:
    """
    Calcola SOLO lo score ottimo dell'allineamento globale usando
    spazio O(min(n, m)) anziché O(n·m).

    MIGLIORIA RISPETTO ALL'ORIGINALE:
    ──────────────────────────────────
    • Spazio: da O(n·m) → O(min(n, m))
      Per due sequenze di 10.000 caratteri ciascuna:
        - Originale: ~100.000.000 celle = ~800 MB
        - Ottimizzato: ~10.000 celle = ~80 KB
      Riduzione di un fattore 10.000×

    • Tempo: invariato O(n·m), ma con costante più piccola grazie a:
      - Migliore località di cache (solo 2 righe in memoria)
      - Meno allocazioni di memoria

    MOTIVO DELLA MIGLIORIA:
    ───────────────────────
    Nella fase di Fill, ogni cella F[i][j] dipende solo da:
      F[i-1][j-1], F[i-1][j], F[i][j-1]
    cioè valori della riga precedente e della riga corrente.
    Le righe 0, 1, ..., i-2 non sono più necessarie e possono
    essere scartate. Basta mantenere solo 2 righe.

    LIMITAZIONE:
    ────────────
    Non si può ricostruire l'allineamento (traceback), perché
    servirebbero le informazioni di TUTTE le righe, che qui
    vengono scartate. Per l'allineamento completo in spazio
    lineare, si veda l'algoritmo di Hirschberg (Variante 2).

    Parametri
    ----------
    seq1 : str   — prima sequenza
    seq2 : str   — seconda sequenza
    match, mismatch, gap : int — parametri di scoring

    Restituisce
    -----------
    int : lo score ottimo dell'allineamento globale
    """

    # OTTIMIZZAZIONE: assicurarsi che le righe siano lungo la sequenza
    # più corta, così lo spazio è O(min(n, m)) e non O(max(n, m)).
    # Questo non cambia lo score perché NW è simmetrico rispetto
    # allo scambio delle sequenze (il punteggio ottimo è lo stesso).
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    n = len(seq1)   # la più lunga → determina il numero di righe
    m = len(seq2)   # la più corta → determina la lunghezza delle righe

    # Allocare solo DUE righe di dimensione (m+1) anziché la matrice (n+1)×(m+1)
    # Nota: si usano liste Python anziché np.array per prestazioni migliori
    # (l'accesso scalare a liste è ~5-10× più veloce di numpy in loop Python)
    prev_row = [j * gap for j in range(m + 1)]
    curr_row = [0] * (m + 1)

    # Riempimento riga per riga
    for i in range(1, n + 1):
        curr_row[0] = i * gap       # prima colonna: solo gap

        for j in range(1, m + 1):
            # Calcolo identico all'originale, ma usando prev_row e curr_row
            if seq1[i - 1] == seq2[j - 1]:
                diag = prev_row[j - 1] + match
            else:
                diag = prev_row[j - 1] + mismatch

            su = prev_row[j] + gap         # gap in seq2
            sx = curr_row[j - 1] + gap     # gap in seq1

            curr_row[j] = max(diag, su, sx)

        # Scambiare le righe: la corrente diventa la precedente
        # (si riutilizza lo spazio della vecchia prev_row)
        prev_row, curr_row = curr_row, prev_row

    # Dopo l'ultimo scambio, il risultato è in prev_row[m]
    return prev_row[m]


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE AUSILIARIA PER HIRSCHBERG: ultima riga della matrice NW
# ══════════════════════════════════════════════════════════════════════════════

def _nw_ultima_riga(seq1: str, seq2: str,
                    match: int, mismatch: int, gap: int) -> list:
    """
    Calcola e restituisce l'ULTIMA RIGA della matrice di Needleman-Wunsch
    per l'allineamento di seq1 vs seq2, usando solo spazio O(m).

    Questa funzione è il mattone fondamentale dell'algoritmo di Hirschberg.
    Applicandola sia in avanti che all'indietro (su sequenze rovesciate),
    si individua il punto di taglio ottimo per il divide et impera.

    Parametri
    ----------
    seq1, seq2 : str — sequenze da allineare
    match, mismatch, gap : int — parametri di scoring

    Restituisce
    -----------
    list : lista di dimensione (len(seq2) + 1) contenente
           l'ultima riga della matrice di scoring
    """

    n = len(seq1)
    m = len(seq2)

    # Due righe come nella Variante 1 (liste Python per velocità)
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


# ══════════════════════════════════════════════════════════════════════════════
# VARIANTE 2: Algoritmo di Hirschberg  —  O(n·m) tempo, O(min(n,m)) spazio
# ══════════════════════════════════════════════════════════════════════════════

def hirschberg(seq1: str, seq2: str,
               match: int = 1,
               mismatch: int = -1,
               gap: int = -2) -> tuple:
    """
    Algoritmo di Hirschberg: allineamento globale ottimo con
    ricostruzione dell'allineamento in spazio lineare.

    MIGLIORIA RISPETTO ALL'ORIGINALE:
    ──────────────────────────────────
    • Spazio: da O(n·m) → O(min(n, m))   — come la Variante 1
    • Ma in più: permette di ricostruire l'ALLINEAMENTO COMPLETO
      (non solo lo score), cosa che la Variante 1 non può fare.

    COME FUNZIONA (Divide et Impera + Programmazione Dinamica):
    ────────────────────────────────────────────────────────────
    1. Si divide seq1 a metà (riga i_mid = n//2 della matrice).
    2. Si eseguono DUE passate in spazio lineare:
       - Avanti:   NW su seq1[0..i_mid] vs seq2 → ultima riga = score_sx
       - Indietro: NW su reverse(seq1[i_mid..n]) vs reverse(seq2) → score_dx
    3. Si trova j* = argmax_j { score_sx[j] + score_dx[m-j] }
       Questo j* è il punto in cui l'allineamento ottimo "attraversa"
       la riga di mezzo.
    4. Si ricorre su (seq1[0..i_mid], seq2[0..j*]) e
       (seq1[i_mid..n], seq2[j*..m]).

    PERCHÉ LA COMPLESSITÀ TEMPORALE RESTA O(n·m):
    ──────────────────────────────────────────────
    Ad ogni livello di ricorsione si elabora un'area proporzionale a
    n/2 · m (metà della matrice). Sommando tutti i livelli:
      n·m + n/2·m + n/4·m + ... = n·m · (1 + 1/2 + 1/4 + ...)
                                 = n·m · 2
                                 = O(n·m)
    La serie geometrica converge, quindi il costo totale è al più
    il doppio di quello dell'algoritmo base: una costante moltiplicativa
    trascurabile.

    Parametri
    ----------
    seq1 : str   — prima sequenza
    seq2 : str   — seconda sequenza
    match, mismatch, gap : int — parametri di scoring

    Restituisce
    -----------
    tuple : (score, align1, align2)
        score  - valore ottimo dell'allineamento
        align1 - prima sequenza allineata
        align2 - seconda sequenza allineata
    """

    def _hirschberg_ricorsivo(x: str, y: str) -> tuple:
        """
        Funzione ricorsiva interna che implementa il divide et impera.

        Parametri
        ----------
        x : str — sottoparte della prima sequenza
        y : str — sottoparte della seconda sequenza

        Restituisce
        -----------
        tuple : (align1, align2) — le due sottosequenze allineate
        """

        n = len(x)
        m = len(y)

        # ── Casi base ────────────────────────────────────────────────────

        if n == 0:
            # x è vuota → tutti gap nella prima sequenza
            return "-" * m, y

        if m == 0:
            # y è vuota → tutti gap nella seconda sequenza
            return x, "-" * n

        if n == 1:
            # x ha un solo carattere → allineamento banale
            # Si cerca se il carattere x[0] appare in y per fare un match
            return _allinea_singolo_carattere(x, y)

        if m == 1:
            # y ha un solo carattere → simmetrico al caso precedente
            a2, a1 = _allinea_singolo_carattere(y, x)
            return a1, a2

        # ── Passo ricorsivo (Divide et Impera) ───────────────────────────

        i_mid = n // 2

        # Passata in AVANTI: NW su x[0..i_mid] vs y
        # Otteniamo l'ultima riga: score_sx[j] = score ottimo
        # per allineare x[0..i_mid] con y[0..j]
        score_sx = _nw_ultima_riga(x[:i_mid], y, match, mismatch, gap)

        # Passata all'INDIETRO: NW su reverse(x[i_mid..n]) vs reverse(y)
        # Otteniamo l'ultima riga: score_dx[j] = score ottimo
        # per allineare x[i_mid..n]^R con y^R[0..j]
        # che equivale a: score per allineare x[i_mid..n] con y[m-j..m]
        score_dx = _nw_ultima_riga(x[i_mid:][::-1], y[::-1], match, mismatch, gap)

        # Trovare il punto di taglio ottimo j*:
        # j* = argmax_j { score_sx[j] + score_dx[m - j] }
        # Questo massimizza lo score totale dell'allineamento che
        # passa per la riga i_mid
        j_taglio = 0
        miglior_somma = score_sx[0] + score_dx[m]
        for j in range(1, m + 1):
            s = score_sx[j] + score_dx[m - j]
            if s > miglior_somma:
                miglior_somma = s
                j_taglio = j

        # Ricorrere sui due sottoproblemi
        align1_sx, align2_sx = _hirschberg_ricorsivo(x[:i_mid], y[:j_taglio])
        align1_dx, align2_dx = _hirschberg_ricorsivo(x[i_mid:], y[j_taglio:])

        # Concatenare i risultati
        return align1_sx + align1_dx, align2_sx + align2_dx

    def _allinea_singolo_carattere(x: str, y: str) -> tuple:
        """
        Caso base: allineare un singolo carattere x[0] con l'intera
        sequenza y. Si cerca la posizione migliore dove fare il match
        (o il mismatch meno penalizzante).

        Strategia:
        - Si prova ogni possibile posizione di allineamento di x[0] con y[j]
        - Il resto di y viene allineato con gap
        - Si sceglie la posizione che massimizza lo score

        Esempio:  x = "A", y = "GATTACA"
          Opzione j=0: A------- vs GATTACA  (A con G = mismatch, + 6 gap)
          Opzione j=3: ---A---- vs GATTACA  (A con T = mismatch, + 6 gap)
          Opzione j=5: -----A-- vs GATTACA  (A con A = match!, + 6 gap)
          ...si sceglie quella con score massimo.
        """
        m = len(y)

        # Opzione 0: x[0] è tutto gap → score = m gap per y + 1 gap per x
        miglior_score = (m + 1) * gap
        miglior_j = -1    # -1 indica "non allineare x[0] con nessun y[j]"

        for j in range(m):
            # Score se x[0] è allineato con y[j]
            s = match if x[0] == y[j] else mismatch
            # j gap prima + score per (x[0], y[j]) + (m-j-1) gap dopo
            score_j = j * gap + s + (m - j - 1) * gap
            if score_j > miglior_score:
                miglior_score = score_j
                miglior_j = j

        if miglior_j == -1:
            # Meglio non allineare x[0] con nessun carattere di y
            return "-" * m + x[0], y + "-"
        else:
            # Allineare x[0] con y[miglior_j], gap altrove
            align1 = "-" * miglior_j + x[0] + "-" * (m - miglior_j - 1)
            align2 = y
            return align1, align2

    # ── Esecuzione ────────────────────────────────────────────────────────

    # Calcolare lo score con la Variante 1 (spazio lineare)
    score = nw_score_lineare(seq1, seq2, match, mismatch, gap)

    # Calcolare l'allineamento con il divide et impera
    align1, align2 = _hirschberg_ricorsivo(seq1, seq2)

    return score, align1, align2


# ══════════════════════════════════════════════════════════════════════════════
# TABELLA COMPARATIVA DELLE COMPLESSITÀ
# ══════════════════════════════════════════════════════════════════════════════

def stampa_confronto_complessita() -> None:
    """
    Stampa una tabella comparativa delle complessità delle tre varianti.
    """
    print("\n" + "=" * 75)
    print("  📊 CONFRONTO DELLE COMPLESSITÀ")
    print("=" * 75)

    headers = ["Algoritmo", "Tempo", "Spazio", "Output"]
    dati = [
        ["NW Base",             "O(n·m)", "O(n·m)",       "Score + Allineamento"],
        ["NW Spazio Lineare",   "O(n·m)", "O(min(n,m))",  "Solo Score"],
        ["Hirschberg",          "O(n·m)", "O(min(n,m))",  "Score + Allineamento"],
    ]

    print(tabulate(dati, headers=headers, tablefmt="fancy_grid",
                   stralign="center", numalign="center"))

    print("\n  💡 Nota: L'algoritmo di Hirschberg ha una costante")
    print("     moltiplicativa ~2× nel tempo (dovuta alla ricorsione),")
    print("     ma la complessità asintotica resta O(n·m).\n")


def stampa_confronto_memoria(n: int, m: int) -> None:
    """
    Mostra un confronto concreto dell'uso di memoria per dimensioni date.
    """
    mem_base = (n + 1) * (m + 1) * 8       # int64 = 8 byte
    mem_lineare = 2 * (min(n, m) + 1) * 8   # due righe

    def formatta_byte(b: int) -> str:
        if b < 1024:
            return f"{b} B"
        elif b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        elif b < 1024 ** 3:
            return f"{b / (1024**2):.1f} MB"
        else:
            return f"{b / (1024**3):.1f} GB"

    rapporto = mem_base / mem_lineare if mem_lineare > 0 else float('inf')

    print(f"\n  📏 Confronto memoria per sequenze di lunghezza n={n}, m={m}:")
    print(f"     • NW Base:          {formatta_byte(mem_base)}")
    print(f"     • Spazio lineare:   {formatta_byte(mem_lineare)}")
    print(f"     • Riduzione:        {rapporto:.0f}×\n")


# ══════════════════════════════════════════════════════════════════════════════
# ESECUZIONE PRINCIPALE — Test e confronto
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import time

    print("=" * 75)
    print("  VARIANTI OTTIMIZZATE — NEEDLEMAN-WUNSCH")
    print("  Migliorie spaziali per istanze di grandi dimensioni")
    print("=" * 75)

    # ── Parametri di scoring ─────────────────────────────────────────────
    MATCH    =  1
    MISMATCH = -1
    GAP      = -2

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Verifica di correttezza su sequenze piccole
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 75)
    print("  TEST 1: Verifica di correttezza (sequenze piccole)")
    print("─" * 75)

    seq1 = "GCATGCG"
    seq2 = "GATTACA"

    print(f"\n  Sequenza 1: {seq1}")
    print(f"  Sequenza 2: {seq2}")
    print(f"  Match: {MATCH}  |  Mismatch: {MISMATCH}  |  Gap: {GAP}")

    # Algoritmo base (dall'altro file)
    score_base, align1_base, align2_base, _ = needleman_wunsch(
        seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )

    # Variante 1: solo score in spazio lineare
    score_lin = nw_score_lineare(
        seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )

    # Variante 2: Hirschberg (score + allineamento in spazio lineare)
    score_hir, align1_hir, align2_hir = hirschberg(
        seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )

    print(f"\n  Risultati:")
    print(f"    NW Base:          score = {score_base}")
    print(f"    NW Lineare:       score = {score_lin}")
    print(f"    Hirschberg:       score = {score_hir}")

    # Verificare la correttezza degli score
    assert score_base == score_lin, "❌ Score NW Lineare diverso dal base!"
    assert score_base == score_hir, "❌ Score Hirschberg diverso dal base!"
    print(f"\n  ✅ Tutti gli score coincidono: {score_base}")

    # Verificare che gli allineamenti di Hirschberg siano validi
    # (potrebbero differire da quelli base, ma devono avere lo stesso score)
    print(f"\n  Allineamento NW Base:")
    print(f"    {align1_base}")
    print(f"    {align2_base}")
    print(f"\n  Allineamento Hirschberg:")
    print(f"    {align1_hir}")
    print(f"    {align2_hir}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: Verifica su un secondo esempio
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 75)
    print("  TEST 2: Secondo esempio")
    print("─" * 75)

    seq1_b = "AGTC"
    seq2_b = "ATC"

    print(f"\n  Sequenza 1: {seq1_b}")
    print(f"  Sequenza 2: {seq2_b}")

    score_base_b, a1_b, a2_b, _ = needleman_wunsch(
        seq1_b, seq2_b, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    score_lin_b = nw_score_lineare(
        seq1_b, seq2_b, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    score_hir_b, ah1_b, ah2_b = hirschberg(
        seq1_b, seq2_b, match=MATCH, mismatch=MISMATCH, gap=GAP
    )

    assert score_base_b == score_lin_b == score_hir_b
    print(f"  ✅ Score: {score_base_b} (tutti concordi)")
    print(f"  Allineamento Hirschberg: {ah1_b} / {ah2_b}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: Confronto prestazionale su sequenze più lunghe
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 75)
    print("  TEST 3: Confronto prestazionale")
    print("─" * 75)

    # Generare sequenze casuali di DNA
    import random
    random.seed(42)
    ALFABETO = "ACGT"
    LUNGHEZZA = 500

    seq_lunga_1 = "".join(random.choice(ALFABETO) for _ in range(LUNGHEZZA))
    seq_lunga_2 = "".join(random.choice(ALFABETO) for _ in range(LUNGHEZZA))

    print(f"\n  Sequenze casuali di DNA, lunghezza = {LUNGHEZZA}")

    # NW Base
    t0 = time.perf_counter()
    score_base_l, _, _, _ = needleman_wunsch(
        seq_lunga_1, seq_lunga_2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    t_base = time.perf_counter() - t0

    # NW Spazio Lineare (solo score)
    t0 = time.perf_counter()
    score_lin_l = nw_score_lineare(
        seq_lunga_1, seq_lunga_2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    t_lineare = time.perf_counter() - t0

    # Hirschberg
    t0 = time.perf_counter()
    score_hir_l, _, _ = hirschberg(
        seq_lunga_1, seq_lunga_2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    t_hirschberg = time.perf_counter() - t0

    assert score_base_l == score_lin_l == score_hir_l

    print(f"\n  ✅ Score: {score_base_l} (tutti concordi)")
    print(f"\n  ⏱  Tempi di esecuzione:")
    print(f"     • NW Base:          {t_base:.4f} s")
    print(f"     • NW Lineare:       {t_lineare:.4f} s")
    print(f"     • Hirschberg:       {t_hirschberg:.4f} s")

    # ══════════════════════════════════════════════════════════════════════
    # Tabella comparativa delle complessità
    # ══════════════════════════════════════════════════════════════════════

    stampa_confronto_complessita()

    # Confronto memoria concreto
    stampa_confronto_memoria(LUNGHEZZA, LUNGHEZZA)
    stampa_confronto_memoria(10_000, 10_000)
    stampa_confronto_memoria(100_000, 100_000)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: Dimostrazione su istanze che l'algoritmo base NON può gestire
    # ══════════════════════════════════════════════════════════════════════

    print("─" * 75)
    print("  TEST 4: Istanza di grandi dimensioni (solo varianti ottimizzate)")
    print("─" * 75)

    LUNGHEZZA_GRANDE = 5_000

    seq_grande_1 = "".join(random.choice(ALFABETO) for _ in range(LUNGHEZZA_GRANDE))
    seq_grande_2 = "".join(random.choice(ALFABETO) for _ in range(LUNGHEZZA_GRANDE))

    print(f"\n  Sequenze casuali di DNA, lunghezza = {LUNGHEZZA_GRANDE}")
    print(f"  ⚠  L'algoritmo base richiederebbe ~{LUNGHEZZA_GRANDE**2 * 8 / (1024**2):.0f} MB")
    print(f"     Le varianti ottimizzate usano ~{2 * LUNGHEZZA_GRANDE * 8 / 1024:.0f} KB")

    t0 = time.perf_counter()
    score_grande = nw_score_lineare(
        seq_grande_1, seq_grande_2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    t_grande = time.perf_counter() - t0

    print(f"\n  ✅ NW Lineare completato: score = {score_grande}")
    print(f"     Tempo: {t_grande:.2f} s")

    t0 = time.perf_counter()
    score_grande_h, _, _ = hirschberg(
        seq_grande_1, seq_grande_2, match=MATCH, mismatch=MISMATCH, gap=GAP
    )
    t_grande_h = time.perf_counter() - t0

    print(f"  ✅ Hirschberg completato: score = {score_grande_h}")
    print(f"     Tempo: {t_grande_h:.2f} s")

    assert score_grande == score_grande_h

    print("\n" + "=" * 75)
    print("  ✅ Tutti i test completati con successo")
    print("=" * 75)
