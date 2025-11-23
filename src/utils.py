from typing import List

Grid = List[List[int]]

def parse_grid_from_text(raw_text: str) -> Grid:
    candidate_rows = []
    for line in raw_text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens and all(token.isdigit() for token in tokens):
            candidate_rows.append([int(token) for token in tokens])
        else:
            candidate_rows.append(None)

    blocks = []
    current_block = []

    for row in candidate_rows:
        if row is None:
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            if current_block and len(row) != len(current_block[0]):
                blocks.append(current_block)
                current_block = [row]
            else:
                current_block.append(row)

    if current_block:
        blocks.append(current_block)

    if not blocks:
        raise ValueError("Could not parse any numeric rows from OpenAI response.")

    return blocks[-1]

def format_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def verify_prediction(predicted: Grid, expected: Grid) -> bool:
    return predicted == expected
