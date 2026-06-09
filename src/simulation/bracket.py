"""2026 World Cup structure: groups, bracket template, third-place allocation.

Sources (fetched 2026-06-09):
- Groups: FIFA Final Draw results / Wikipedia 2026 FIFA World Cup
- Bracket template (matches 73-104): Wikipedia, 2026 FIFA World Cup
  knockout stage. The exact third-place allocation table is Annex C of the
  FIFA regulations (495 combinations); we implement it as bipartite
  matching against the published slot constraints.

Team names use OUR dataset's naming (e.g. "South Korea", not "Korea
Republic"); the simulation validates them against data/processed files.
"""

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOSTS = {"United States", "Mexico", "Canada"}

# Round of 32: (match_no, slot_a, slot_b). Slots: ("W", g) group winner,
# ("R", g) runner-up, ("T", allowed) a third from one of `allowed` groups.
R32 = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("T", "ABCDF")),
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("T", "CDFGH")),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("T", "CEFHI")),
    (80, ("W", "L"), ("T", "EHIJK")),
    (81, ("W", "D"), ("T", "BEFIJ")),
    (82, ("W", "G"), ("T", "AEHIJ")),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("T", "EFGIJ")),
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("T", "DEIJL")),
    (88, ("R", "D"), ("R", "G")),
]

# Later rounds: match_no -> (feeder_match_a, feeder_match_b)
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}

THIRD_SLOTS: dict[int, frozenset] = {
    m: frozenset(slot_b[1]) for m, _, slot_b in R32 if slot_b[0] == "T"
}


def allocate_thirds(qualified: set[str]) -> dict[int, str]:
    """Assign the 8 qualified third-place groups to the 8 constrained R32
    slots — a bipartite perfect matching, found by backtracking with the
    most-constrained slot first. FIFA's Annex C guarantees a solution
    exists for every C(12,8)=495 combination (verified in tests).

    Deterministic for a given input set. The official Annex C may pick a
    different *valid* assignment; slot constraints are honored either way.
    """
    assert len(qualified) == 8, f"need exactly 8 third-place groups, got {qualified}"
    slots = sorted(THIRD_SLOTS, key=lambda m: (len(THIRD_SLOTS[m] & qualified), m))

    def backtrack(i: int, remaining: frozenset, acc: dict) -> dict | None:
        if i == len(slots):
            return acc
        m = slots[i]
        for g in sorted(THIRD_SLOTS[m] & remaining):
            result = backtrack(i + 1, remaining - {g}, {**acc, m: g})
            if result is not None:
                return result
        return None

    result = backtrack(0, frozenset(qualified), {})
    assert result is not None, f"no valid allocation for {sorted(qualified)}"
    return result
