"""Export each shipped supplementary .xlsx to CSV.

The published supplements are Excel workbooks whose first row is a long title
banner and whose second row is the real header. Converting once here keeps every
other script on plain CSV. The CSV outputs are already committed, so you normally
do not need to run this; it is here for provenance.

Requires openpyxl (the only script that does).

    python code/convert_supplementary_to_csv.py
"""

import argparse
import csv
import re
from pathlib import Path

import openpyxl


def table_number(filename: str) -> int | None:
    """The 'N' in 'Supplementary Table N…', or None for other files."""
    match = re.match(r"Supplementary Table (\d+)", filename)
    return int(match.group(1)) if match else None


def is_banner_row(row: tuple) -> bool:
    """True if a row is a single long title spanning an otherwise empty line."""
    filled = [cell for cell in row if cell not in (None, "")]
    return len(filled) == 1 and isinstance(filled[0], str) and len(filled[0]) > 40


def used_width(rows: list[tuple]) -> int:
    """Number of columns up to the last non-empty cell across all rows."""
    width = 0
    for row in rows:
        for index, cell in enumerate(row):
            if cell not in (None, ""):
                width = max(width, index + 1)
    return width


def export_worksheet(worksheet, output_path: Path) -> tuple[int, int]:
    """Write one worksheet to CSV, dropping a leading banner. Returns (rows, cols)."""
    rows = [row for row in worksheet.iter_rows(values_only=True)
            if any(cell not in (None, "") for cell in row)]
    if not rows:
        return 0, 0
    if is_banner_row(rows[0]):
        rows = rows[1:]
    width = used_width(rows)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(list(row[:width]))
    return len(rows) - 1, width


def main() -> int:
    release_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=str(release_root / "data/published_tables/supplementary"))
    parser.add_argument("--dst", default=str(release_root / "data/published_tables/supplementary_csv"))
    args = parser.parse_args()

    source_dir = Path(args.src)
    destination_dir = Path(args.dst)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for workbook_path in sorted(source_dir.glob("*.xlsx")):
        number = table_number(workbook_path.name)
        if number is None:
            continue
        workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
        output_name = f"supp_table_{number}.csv"
        n_rows, n_cols = export_worksheet(workbook.active, destination_dir / output_name)
        workbook.close()
        print(f"{workbook_path.name[:58]:<60} -> {output_name}  ({n_rows} rows x {n_cols} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
