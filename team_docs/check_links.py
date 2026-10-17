#!/usr/bin/env python3
"""Check (and optionally repair) the links in our Markdown documentation.

Our docs link straight to the code, like this:

    [`pure_pursuit()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L58)

When someone edits the code, line 58 may stop being where `pure_pursuit` is
defined. This script finds every such link, looks the name in the backticks
up in the linked file, and reports (or, with --fix, rewrites) the line number.

Names it understands, inside the backticks of the link text:
    `pure_pursuit()`        a function or class:  def pure_pursuit / class ...
    `Driver.plan()`         a method: def plan, searched after class Driver
    `make_update.update()`  a nested function, the same way
    `LIMITS`                a module constant:    LIMITS = ...
    `DEFAULTS['mu']`        a key inside a dict:  'mu': ...  after DEFAULTS
Other links are checked too: the file must exist, and a #heading anchor in a
Markdown file must match one of its headings (GitHub's rules for anchors).

    python3 team_docs/check_links.py          # report problems, exit 1 if any
    python3 team_docs/check_links.py --fix    # also rewrite stale line numbers
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / 'SUBMISSION.md', REPO / 'training' / 'README.md',
        *sorted((REPO / 'team_docs').glob('*.md'))]

LINK = re.compile(r'\[([^\]]*)\]\(([^)\s]+)\)')
SYMBOL = re.compile(r'^`([A-Za-z_][\w.]*?)(?:\(\))?(?:\[\'(\w+)\'\])?`$')


def find_symbol(lines, parts):
    """Line number (1-based) where the dotted name `parts` is defined, or None.

    Each part is searched for after the line where the previous part was found,
    so `Driver.plan` finds the `def plan` inside `class Driver`.
    """
    start = 0
    for part in parts:
        pattern = re.compile(rf'^\s*(?:(?:async\s+)?def|class)\s+{re.escape(part)}\b'
                             rf'|^\s*{re.escape(part)}\s*(?::[^=]*)?=(?!=)'
                             rf'|^\s*[\'"]{re.escape(part)}[\'"]\s*:')
        for i in range(start, len(lines)):
            if pattern.search(lines[i]):
                start = i
                break
        else:
            return None
    return start + 1


def slug(heading):
    """GitHub's anchor for a Markdown heading."""
    text = re.sub(r'[`*]|<[^>]+>', '', heading.strip().lower())
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'[^\w\- ]', '', text)
    return text.replace(' ', '-')


def anchors(md_file):
    found, seen = set(), {}
    in_code = False
    for line in md_file.read_text().splitlines():
        if line.lstrip().startswith('```'):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r'^#{1,6}\s+(.*)', line)
        if m:
            base = slug(m.group(1))
            n = seen.get(base, 0)
            found.add(base if n == 0 else f'{base}-{n}')
            seen[base] = n + 1
    return found


def check(doc, fix):
    problems, text = [], doc.read_text()

    def repair(m):
        label, target = m.group(1), m.group(2)
        if re.match(r'^[a-z]+:', target):                     # http:, https:, mailto:
            return m.group(0)
        path, _, anchor = target.partition('#')
        file = (doc.parent / path).resolve() if path else doc
        where = f'{doc.relative_to(REPO)}: [{label}]({target})'
        if not file.exists():
            problems.append(f'{where}: no such file')
            return m.group(0)
        if file.suffix == '.py':
            sym = SYMBOL.match(label.strip())
            if not sym or label.strip('`').endswith('.py'):     # a file name, not a symbol
                return m.group(0)
            parts = sym.group(1).split('.') + ([sym.group(2)] if sym.group(2) else [])
            line = find_symbol(file.read_text().splitlines(), parts)
            if line is None:
                problems.append(f'{where}: `{".".join(parts)}` not found in {path}')
                return m.group(0)
            if anchor != f'L{line}':
                if not fix:
                    problems.append(f'{where}: should be #L{line}')
                return f'[{label}]({path}#L{line})'
        elif file.suffix == '.md' and anchor and anchor not in anchors(file):
            problems.append(f'{where}: no heading #{anchor}')
        return m.group(0)

    new = LINK.sub(repair, text)
    if fix and new != text:
        doc.write_text(new)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--fix', action='store_true', help='rewrite stale #L line numbers')
    args = parser.parse_args()
    problems = [p for doc in DOCS if doc.exists() for p in check(doc, args.fix)]
    for p in problems:
        print(p)
    print(f'{len(problems)} problem(s) in {sum(d.exists() for d in DOCS)} files')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
