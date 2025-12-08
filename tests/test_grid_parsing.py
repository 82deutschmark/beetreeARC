import sys
import os
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import parse_grid_from_text

GRID_CASES_DIR = Path(__file__).parent / "grid_parsing_cases"

def get_test_cases():
    case_dir = GRID_CASES_DIR / "cases"
    cases = []
    if case_dir.exists():
        for f in case_dir.iterdir():
            if f.suffix == ".txt":
                cases.append(f)
    return sorted(cases)

@pytest.mark.parametrize("case_file", get_test_cases())
def test_parsing_all_cases(case_file):
    """Ensure all collected test cases can be parsed successfully."""
    with open(case_file, "r") as f:
        text = f.read()
    
    try:
        grid = parse_grid_from_text(text)
        assert grid is not None
        assert isinstance(grid, list)
        assert len(grid) >= 2, f"Grid height {len(grid)} is too small (min 2)"
        assert isinstance(grid[0], list)
        # Basic validity check
        width = len(grid[0])
        assert width >= 2, f"Grid width {width} is too small (min 2)"
        for row in grid:
            assert len(row) == width
            assert all(isinstance(cell, int) for cell in row)
    except ValueError as e:
        pytest.fail(f"Failed to parse case {case_file.name}: {e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
