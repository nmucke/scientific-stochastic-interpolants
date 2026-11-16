"""Add \\underline{} to the second-best value per column of LaTeX results tables.

Rule (same as the bold rule of common/latex_tables.py): lower is better; values are
compared rounded to 2 decimals in UNSCALED units (a "(\\times 10^{-1})" column group
is divided by 10 first); rows whose label contains "reference" and cells carrying a
superscript (div k/n, partial) or no number do not compete. The best rounded value
must already be bold (checked: a column whose bold set disagrees is left untouched
and reported); every competing cell equal to the next distinct rounded value is
underlined.

usage: underline_second.py <file.tex> <label> [<label> ...]   (edits in place)
"""
import re
import sys

NUM = re.compile(r"^\s*(\\textbf\{|\\underline\{)?\s*(-?\d+(?:\.\d+)?)\s*\}?\s*$")


def split_rows(body):
    """Logical rows of a tabular body: (start, end) spans ending at '\\\\'."""
    rows, pos = [], 0
    for m in re.finditer(r"\\\\", body):
        rows.append((pos, m.end()))
        pos = m.end()
    return rows


def group_scales(header_text, ncols):
    """Per data column: divisor to unscaled units, from the \\multicolumn header."""
    scales = []
    for span, txt in re.findall(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*?)\}\s*(?=&|\\\\|$)", header_text, re.S):
        m = re.search(r"\\times\s*10\^\{-(\d+)\}", txt)
        scales += [10 ** int(m.group(1)) if m else 1] * int(span)
    return scales if len(scales) == ncols else [1] * ncols


def process_table(tab):
    b0 = tab.index(r"\begin{tabular}")
    b0 = tab.index("}", tab.index("}", b0) + 1) + 1  # after \begin{tabular}{spec}
    b1 = tab.index(r"\end{tabular}")
    body = tab[b0:b1]
    spans = split_rows(body)
    cells = []  # (row_idx, col_idx, abs_start, abs_end, text)
    rows_cells = []
    header = ""
    for ri, (s, e) in enumerate(spans):
        row = body[s:e - 2]
        row_clean = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", row)
        if "\\multicolumn" in row_clean and "&" in row_clean and not rows_cells:
            header += row_clean
        parts, idx, offs = [], 0, []
        for m in re.finditer(r"(?<!\\)&", row):
            parts.append((idx, m.start()))
            idx = m.end()
        parts.append((idx, len(row)))
        label = row[parts[0][0]:parts[0][1]]
        data = parts[1:]
        if not data or "\\multicolumn" in row_clean and len(data) < 2:
            continue
        rows_cells.append((ri, s, label, data))
    ncols = max((len(d) for _, _, _, d in rows_cells), default=0)
    scales = group_scales(header, ncols)
    edits = []  # (abs_start, abs_end, new_text)
    report = []
    for c in range(ncols):
        comp = []
        for ri, s, label, data in rows_cells:
            if c >= len(data) or "reference" in label.lower():
                continue
            a, z = data[c]
            txt = body[s + a:s + z]
            if "textsuperscript" in txt:
                continue
            m = NUM.match(txt)
            if not m:
                continue
            val = float(m.group(2))
            comp.append((round(val / scales[c] + 1e-12, 2), bool(m.group(1) and "textbf" in m.group(1)), s + a, s + z, txt, m.group(2)))
        vals = sorted({v for v, *_ in comp})
        if len(vals) < 2:
            continue
        best, second = vals[0], vals[1]
        bold_set = {(st) for v, b, st, *_ in comp if b}
        want_bold = {(st) for v, b, st, *_ in comp if v == best}
        if bold_set != want_bold:
            report.append(f"col {c}: existing bold differs from rule -> left untouched")
            continue
        for v, b, st, en, txt, num in comp:
            if v == second and not b:
                lead = txt[: len(txt) - len(txt.lstrip())]
                trail = txt[len(txt.rstrip()):]
                edits.append((st, en, f"{lead}\\underline{{{num}}}{trail}"))
    for st, en, new in sorted(edits, reverse=True):
        body = body[:st] + new + body[en:]
    return tab[:b0] + body + tab[b1:], len(edits), report


def main():
    path, labels = sys.argv[1], sys.argv[2:]
    src = open(path).read()
    for lab in labels:
        li = src.index(f"\\label{{{lab}}}")
        env_start = max(src.rfind("\\begin{table", 0, li), src.rfind("\\begin{wraptable", 0, li))
        env_end = min(i for i in (src.find("\\end{table*}", li), src.find("\\end{wraptable}", li), src.find("\\end{table}", li)) if i >= 0)
        tab = src[env_start:env_end]
        if "\\underline{" in tab.split("\\begin{tabular}")[1]:
            print(f"{lab}: already has underlines, skipped"); continue
        new, n, rep = process_table(tab)
        new = new.replace(
            r"\textbf{Bold} marks the best value in each column, together with every value equal to it to 2 decimal places in unscaled units.",
            r"\textbf{Bold} marks the best value in each column and \underline{underline} the second best, each together with every value equal to it to 2 decimal places in unscaled units.")
        new = new.replace(
            "\\textbf{Bold} marks the best value in each column,\n    including ties to 2 decimal places.",
            "\\textbf{Bold} marks the best value in each column and\n    \\underline{underline} the second best, including ties\n    to 2 decimal places.")
        new = new.replace(
            r"\textbf{Bold} marks the best value in each column, together with every value equal to it to 2 decimal places (on the unscaled values).",
            r"\textbf{Bold} marks the best value in each column and \underline{underline} the second best, each together with every value equal to it to 2 decimal places (on the unscaled values).")
        src = src[:env_start] + new + src[env_end:]
        print(f"{lab}: {n} cells underlined", *rep, sep="\n  ")
    open(path, "w").write(src)


if __name__ == "__main__":
    main()
