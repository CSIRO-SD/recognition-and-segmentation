"""Split ``weak_context_text`` into SigLIP2-safe chunks (≤64 positions each for almost all rows)."""

from __future__ import annotations

M_LATITUDE = "Latitude:"
M_SAMING = "Sampling event:"
M_NOTES = "Notes:"


def split_weak_context_four(s: str) -> tuple[str, str, str, str]:
    """Segments (order in template must hold: Latitude < Sampling event < Notes):

    1. before ``Latitude:``
    2. ``Latitude:`` … before ``Sampling event:``
    3. ``Sampling event:`` … before ``Notes:``
    4. ``Notes:`` … end
    """
    s = (s or "").strip()
    if not s:
        return "", "", "", ""
    if M_LATITUDE not in s or M_SAMING not in s or M_NOTES not in s:
        raise ValueError(
            f"weak_context_text missing one of required markers {M_LATITUDE!r}, {M_SAMING!r}, {M_NOTES!r}",
        )
    i_lat, i_se, i_nt = s.index(M_LATITUDE), s.index(M_SAMING), s.index(M_NOTES)
    if not (i_lat < i_se < i_nt):
        raise ValueError(f"marker order invalid: lat={i_lat} se={i_se} nt={i_nt}")
    return (
        s[:i_lat].strip(),
        s[i_lat:i_se].strip(),
        s[i_se:i_nt].strip(),
        s[i_nt:].strip(),
    )
