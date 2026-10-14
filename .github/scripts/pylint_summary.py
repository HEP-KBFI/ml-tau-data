"""Render pylint `--output-format=json2` output as a Markdown summary.

Usage: python pylint_summary.py pylint_output.json > lint_summary.md
"""

import json
import sys
from collections import Counter

# GitHub rejects comments longer than 65536 characters.
MAX_CHARS = 60000
EMOJI = {
    "fatal": "💀",
    "error": "❌",
    "warning": "⚠️",
    "refactor": "🛠️",
    "convention": "📏",
    "info": "ℹ️",
}
ORDER = list(EMOJI)


def main(path):
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    messages = report["messages"]
    score = report["statistics"]["score"]
    counts = Counter(m["type"] for m in messages)

    header = [
        "<!-- pylint-summary -->",
        "### 🧹 Pylint summary",
        "",
        f"🧾 **Score:** {score:.2f}/10",
        "",
        " · ".join(f"{EMOJI[t]} **{t}:** {counts.get(t, 0)}" for t in ORDER if t != "info"),
        "",
        "_Ignored checks are configured in `.pylintrc`._",
        "",
    ]
    if not messages:
        print("\n".join(header + ["No issues found 🎉"]))
        return

    messages.sort(key=lambda m: (ORDER.index(m["type"]), m["path"], m["line"]))
    body = []
    for m in messages:
        first_line = m["message"].splitlines()[0]
        body.append(
            f"- {EMOJI[m['type']]} `{m['path']}:{m['line']}:{m['column']}` "
            f"**{m['messageId']}** ({m['symbol']}): {first_line}"
        )

    out = "\n".join(header) + "\n<details>\n<summary>Show all {} issues</summary>\n\n".format(len(messages))
    footer = "\n</details>\n"
    shown = 0
    for line in body:
        if len(out) + len(line) + len(footer) + 200 > MAX_CHARS:
            out += f"\n- … {len(body) - shown} more, see the **pylint-report** artifact.\n"
            break
        out += line + "\n"
        shown += 1
    print(out + footer)


if __name__ == "__main__":
    main(sys.argv[1])
