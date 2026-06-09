"""Bracket structure tests — esp. the third-place allocation matcher."""

from itertools import combinations

from src.simulation.bracket import (FINAL, GROUPS, QF, R16, R32, SF,
                                    THIRD_SLOTS, allocate_thirds)


def test_structure_counts():
    assert len(GROUPS) == 12 and all(len(t) == 4 for t in GROUPS.values())
    teams = [t for g in GROUPS.values() for t in g]
    assert len(teams) == len(set(teams)) == 48
    assert len(R32) == 16 and len(THIRD_SLOTS) == 8
    assert len(R16) == 8 and len(QF) == 4 and len(SF) == 2 and len(FINAL) == 1
    # Every R32 winner feeds exactly one R16 match.
    fed = sorted(m for pair in R16.values() for m in pair)
    assert fed == sorted(m for m, _, _ in R32)


def test_no_group_meets_own_third():
    """A group winner can never face the third from its own group."""
    for match_no, slot_a, slot_b in R32:
        if slot_b[0] == "T":
            assert slot_a[1] not in slot_b[1], f"match {match_no}"


def test_all_495_combinations_allocate():
    """FIFA designed the slot constraints so EVERY possible set of 8
    qualified thirds has a perfect matching. If our transcription of the
    slot constraints were wrong, some combination would fail."""
    for combo in combinations("ABCDEFGHIJKL", 8):
        alloc = allocate_thirds(set(combo))
        assert sorted(alloc.values()) == sorted(combo)          # all used once
        for match_no, group in alloc.items():
            assert group in THIRD_SLOTS[match_no]               # constraint held


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All bracket tests passed.")
