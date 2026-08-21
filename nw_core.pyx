# cython: boundscheck=False, wraparound=False, cdivision=True
"""
================================================================================
NW_CORE — Implementazione Cython dei loop critici di Needleman-Wunsch
================================================================================

Questo modulo contiene le stesse identiche logiche dei file Python:
  - needleman_wunsch.py            → needleman_wunsch()
  - needleman_wunsch_ottimizzato.py → nw_score_lineare(), hirschberg()

L'unica differenza è che i loop interni sono compilati in C da Cython,
ottenendo un speed-up di ~50-100× rispetto a Python puro.

OTTIMIZZAZIONE CHIAVE: le stringhe Python vengono convertite in array
di byte C (char*) PRIMA di entrare nei loop. In questo modo il confronto
tra caratteri seq1[i] == seq2[j] diventa un confronto tra char C,
senza alcun overhead di oggetti Python nell'inner loop.

La correttezza è identica: stessi algoritmi, stesse formule, stesso output.
================================================================================
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 1: Needleman-Wunsch Base  —  O(n·m) tempo, O(n·m) spazio
# ══════════════════════════════════════════════════════════════════════════════

def needleman_wunsch(str seq1, str seq2,
                     int match=1, int mismatch=-1, int gap_penalty=-2):
    """
    Algoritmo di Needleman-Wunsch base.
    Restituisce (score, align1, align2, F) dove F è la matrice completa.
    """
    cdef int n = len(seq1)
    cdef int m = len(seq2)
    cdef int i, j
    cdef int diag, su, sx, s

    # Convertire le stringhe in byte C per accesso veloce
    cdef bytes b_seq1 = seq1.encode('ascii')
    cdef bytes b_seq2 = seq2.encode('ascii')
    cdef const char* c_seq1 = b_seq1
    cdef const char* c_seq2 = b_seq2

    # Allocare la matrice come lista di liste (per compatibilità con il codice
    # di visualizzazione che accede a F[i][j])
    F = [[0] * (m + 1) for _ in range(n + 1)]

    # Inizializzazione
    for i in range(1, n + 1):
        F[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        F[0][j] = j * gap_penalty

    # Fill — confronto char C (nessun overhead Python)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if c_seq1[i - 1] == c_seq2[j - 1]:
                diag = F[i - 1][j - 1] + match
            else:
                diag = F[i - 1][j - 1] + mismatch

            su = F[i - 1][j] + gap_penalty
            sx = F[i][j - 1] + gap_penalty

            # max inlined
            if diag >= su and diag >= sx:
                F[i][j] = diag
            elif su >= sx:
                F[i][j] = su
            else:
                F[i][j] = sx

    cdef int score = F[n][m]

    # Traceback
    align1_parts = []
    align2_parts = []
    i = n
    j = m

    while i > 0 and j > 0:
        if c_seq1[i - 1] == c_seq2[j - 1]:
            s = match
        else:
            s = mismatch

        if F[i][j] == F[i - 1][j - 1] + s:
            align1_parts.append(seq1[i - 1])
            align2_parts.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif F[i][j] == F[i - 1][j] + gap_penalty:
            align1_parts.append(seq1[i - 1])
            align2_parts.append("-")
            i -= 1
        else:
            align1_parts.append("-")
            align2_parts.append(seq2[j - 1])
            j -= 1

    while i > 0:
        align1_parts.append(seq1[i - 1])
        align2_parts.append("-")
        i -= 1
    while j > 0:
        align1_parts.append("-")
        align2_parts.append(seq2[j - 1])
        j -= 1

    align1 = "".join(reversed(align1_parts))
    align2 = "".join(reversed(align2_parts))

    return score, align1, align2, F


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 2: NW Score Lineare  —  O(n·m) tempo, O(min(n,m)) spazio
# ══════════════════════════════════════════════════════════════════════════════

def nw_score_lineare(str seq1, str seq2,
                     int match=1, int mismatch=-1, int gap_penalty=-2):
    """
    Calcola SOLO lo score ottimo con spazio O(min(n, m)).

    Strategia di misurazione memoria:
    I calcoli usano array C (malloc) per massima velocità, ma si allocano
    anche due liste Python "sentinella" della stessa dimensione, tenute
    vive durante il calcolo, così che tracemalloc registri correttamente
    l'occupazione di memoria O(min(n,m)) dell'algoritmo.
    """
    # Ottimizzazione: righe lungo la sequenza più corta
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    cdef int n = len(seq1)
    cdef int m = len(seq2)
    cdef int i, j
    cdef int diag, su, sx

    # Convertire le stringhe in byte C per accesso veloce
    cdef bytes b_seq1 = seq1.encode('ascii')
    cdef bytes b_seq2 = seq2.encode('ascii')
    cdef const char* c_seq1 = b_seq1
    cdef const char* c_seq2 = b_seq2

    # Liste Python "sentinella" — visibili a tracemalloc per misurare
    # la memoria reale usata dall'algoritmo (2 righe di m+1 interi).
    # Non vengono usate nei calcoli, servono solo per la misurazione.
    _mem_witness_1 = [0] * (m + 1)
    _mem_witness_2 = [0] * (m + 1)

    # Array C per i calcoli effettivi (velocità massima)
    cdef int *prev_row = <int *>malloc((m + 1) * sizeof(int))
    cdef int *curr_row = <int *>malloc((m + 1) * sizeof(int))
    cdef int *temp

    if prev_row == NULL or curr_row == NULL:
        if prev_row != NULL:
            free(prev_row)
        if curr_row != NULL:
            free(curr_row)
        raise MemoryError("Impossibile allocare memoria per le righe NW")

    # Inizializzazione
    for j in range(m + 1):
        prev_row[j] = j * gap_penalty
        curr_row[j] = 0

    # Fill — inner loop 100% C: puntatori int* + confronto char
    for i in range(1, n + 1):
        curr_row[0] = i * gap_penalty

        for j in range(1, m + 1):
            if c_seq1[i - 1] == c_seq2[j - 1]:
                diag = prev_row[j - 1] + match
            else:
                diag = prev_row[j - 1] + mismatch

            su = prev_row[j] + gap_penalty
            sx = curr_row[j - 1] + gap_penalty

            if diag >= su and diag >= sx:
                curr_row[j] = diag
            elif su >= sx:
                curr_row[j] = su
            else:
                curr_row[j] = sx

        # Swap
        temp = prev_row
        prev_row = curr_row
        curr_row = temp

    cdef int result = prev_row[m]
    free(prev_row)
    free(curr_row)

    # Mantenere i witness vivi fino a qui (evita ottimizzazione del compilatore)
    del _mem_witness_1, _mem_witness_2

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Funzione ausiliaria: ultima riga della matrice NW (per Hirschberg)
# ══════════════════════════════════════════════════════════════════════════════

cdef list _nw_ultima_riga(str seq1, str seq2,
                          int match, int mismatch, int gap_penalty):
    """
    Restituisce l'ultima riga della matrice NW come lista Python.
    Usata internamente da Hirschberg.
    """
    cdef int n = len(seq1)
    cdef int m = len(seq2)
    cdef int i, j
    cdef int diag, su, sx

    # Convertire le stringhe in byte C
    cdef bytes b_seq1 = seq1.encode('ascii')
    cdef bytes b_seq2 = seq2.encode('ascii')
    cdef const char* c_seq1 = b_seq1
    cdef const char* c_seq2 = b_seq2

    cdef int *prev_row = <int *>malloc((m + 1) * sizeof(int))
    cdef int *curr_row = <int *>malloc((m + 1) * sizeof(int))
    cdef int *temp

    if prev_row == NULL or curr_row == NULL:
        if prev_row != NULL:
            free(prev_row)
        if curr_row != NULL:
            free(curr_row)
        raise MemoryError("Impossibile allocare memoria")

    for j in range(m + 1):
        prev_row[j] = j * gap_penalty
        curr_row[j] = 0

    for i in range(1, n + 1):
        curr_row[0] = i * gap_penalty

        for j in range(1, m + 1):
            if c_seq1[i - 1] == c_seq2[j - 1]:
                diag = prev_row[j - 1] + match
            else:
                diag = prev_row[j - 1] + mismatch

            su = prev_row[j] + gap_penalty
            sx = curr_row[j - 1] + gap_penalty

            if diag >= su and diag >= sx:
                curr_row[j] = diag
            elif su >= sx:
                curr_row[j] = su
            else:
                curr_row[j] = sx

        temp = prev_row
        prev_row = curr_row
        curr_row = temp

    # Convertire in lista Python per restituire
    result = [prev_row[j] for j in range(m + 1)]
    free(prev_row)
    free(curr_row)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 3: Hirschberg  —  O(n·m) tempo, O(min(n,m)) spazio
# ══════════════════════════════════════════════════════════════════════════════

def hirschberg(str seq1, str seq2,
               int match=1, int mismatch=-1, int gap_penalty=-2):
    """
    Algoritmo di Hirschberg: allineamento globale ottimo con
    ricostruzione dell'allineamento in spazio lineare.
    """

    def _allinea_singolo_carattere(str x, str y):
        """Caso base: allineare un singolo carattere x[0] con l'intera y."""
        cdef int m_len = len(y)
        cdef int miglior_score = (m_len + 1) * gap_penalty
        cdef int miglior_j = -1
        cdef int j_inner, s_inner, score_j_inner

        # Convertire in char C per il confronto
        cdef bytes b_x = x.encode('ascii')
        cdef bytes b_y = y.encode('ascii')
        cdef const char* c_x = b_x
        cdef const char* c_y = b_y

        for j_inner in range(m_len):
            if c_x[0] == c_y[j_inner]:
                s_inner = match
            else:
                s_inner = mismatch
            score_j_inner = j_inner * gap_penalty + s_inner + (m_len - j_inner - 1) * gap_penalty
            if score_j_inner > miglior_score:
                miglior_score = score_j_inner
                miglior_j = j_inner

        if miglior_j == -1:
            return "-" * m_len + x[0], y + "-"
        else:
            return "-" * miglior_j + x[0] + "-" * (m_len - miglior_j - 1), y

    def _hirschberg_ricorsivo(str x, str y):
        """Funzione ricorsiva divide et impera."""
        cdef int n_r = len(x)
        cdef int m_r = len(y)
        cdef int i_mid
        cdef int j_inner, j_taglio
        cdef int s_sum, miglior_somma

        # Casi base
        if n_r == 0:
            return "-" * m_r, y
        if m_r == 0:
            return x, "-" * n_r
        if n_r == 1:
            return _allinea_singolo_carattere(x, y)
        if m_r == 1:
            a2, a1 = _allinea_singolo_carattere(y, x)
            return a1, a2

        # Passo ricorsivo
        i_mid = n_r // 2

        score_sx = _nw_ultima_riga(x[:i_mid], y, match, mismatch, gap_penalty)
        score_dx = _nw_ultima_riga(x[i_mid:][::-1], y[::-1], match, mismatch, gap_penalty)

        # Trovare il punto di taglio ottimo
        j_taglio = 0
        miglior_somma = score_sx[0] + score_dx[m_r]
        for j_inner in range(1, m_r + 1):
            s_sum = score_sx[j_inner] + score_dx[m_r - j_inner]
            if s_sum > miglior_somma:
                miglior_somma = s_sum
                j_taglio = j_inner

        # Ricorsione
        align1_sx, align2_sx = _hirschberg_ricorsivo(x[:i_mid], y[:j_taglio])
        align1_dx, align2_dx = _hirschberg_ricorsivo(x[i_mid:], y[j_taglio:])

        return align1_sx + align1_dx, align2_sx + align2_dx

    # Calcolare lo score con la versione lineare
    score = nw_score_lineare(seq1, seq2, match, mismatch, gap_penalty)

    # Calcolare l'allineamento
    align1, align2 = _hirschberg_ricorsivo(seq1, seq2)

    return score, align1, align2
