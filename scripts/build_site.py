"""Build the GitHub Pages site from the notes and the real result files.

Every step's write-up lives in notes/*.md in plain language.  Numbers that
appear on the page are pulled from outputs/*.json where possible, so the site
cannot drift from what the code actually produced.  Run this after each step
and commit docs/.
"""

import json
import os
import re
import subprocess
import sys

import markdown

ROOT = os.path.join(os.path.dirname(__file__), "..")
NOTES = os.path.join(ROOT, "notes")
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(ROOT, "outputs")

STEPS = [
    ("00-what-this-is.md", "The idea", None),
    ("01-build-the-input.md", "Step 1 — build the input", "done"),
    ("02-does-it-work.md", "Step 2 — does it work?", "done"),
    ("03-mutation-scan.md", "Step 3 — break every letter", "done"),
    ("04-validation.md", "Step 4 — check against real disease", "done"),
]

BADGE = {"done": ("done", "#1a7f5a"), "running": ("in progress", "#b8860b"),
         "todo": ("not started", "#8a8a8a"), None: ("", "")}


def git(*args, default=""):
    try:
        return subprocess.check_output(["git", "-C", ROOT, *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def commit_rows(n=25):
    log = git("log", f"-{n}", "--date=short", "--pretty=%h%x1f%ad%x1f%s")
    rows = []
    for line in log.splitlines():
        h, d, s = line.split("\x1f")
        rows.append(f"<tr><td class='mono'>{h}</td><td class='mono'>{d}</td><td>{esc(s)}</td></tr>")
    return "\n".join(rows)


def demote(html):
    """Push note headings one level down so section titles stay dominant."""
    for lvl in (4, 3, 2):
        html = re.sub(rf"<(/?)h{lvl}>", rf"<\g<1>h{lvl + 1}>", html)
    return html


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def figures_block():
    """Link any figures that exist, copied into docs/."""
    out = []
    for fn, cap in (
        ("step2_trna_maps_isolated.png",
         "Predicted folding for all 22 tRNAs, each folded on its own. "
         "Dark pixels are predicted contacts; blue and green circles mark where "
         "the acceptor and anticodon stems should be."),
        ("step2_trna_maps_genome.png",
         "The same tRNAs read out of the whole-genome pass. Most of the signal "
         "is gone."),
    ):
        src = os.path.join(OUT, fn)
        if os.path.exists(src):
            subprocess.run(["cp", src, os.path.join(DOCS, fn)], check=True)
            out.append(f"<figure><img src='{fn}' alt='{esc(cap)}'>"
                       f"<figcaption>{esc(cap)}</figcaption></figure>")
    return "\n".join(out)


def step3_block():
    recs = load("step3_mutations.json")
    if not recs:
        return ""
    n = len(recs)
    trnas = len({r["trna"] for r in recs})
    hits = [r for r in recs if r["rcrs_pos"] == 3243 and r["mut"] == "G"]
    rows = ""
    if hits:
        h = hits[0]
        worse = sum(1 for r in recs if r["pairs_lost"] > h["pairs_lost"])
        pct = 100 * (1 - worse / n)
        rows = (f"<p>The best-known mitochondrial disease mutation, "
                f"<span class='mono'>m.3243A&gt;G</span>, scores in the "
                f"<strong>{pct:.0f}th percentile</strong> for shape damage out of all "
                f"{n:,} mutations tested.</p>")
    top = sorted(recs, key=lambda r: -r["pairs_lost"])[:10]
    tbl = "\n".join(
        f"<tr><td class='mono'>m.{r['rcrs_pos']}{r['wt']}&gt;{r['mut']}</td>"
        f"<td>{r['trna']}</td><td>{r['pairs_lost']:.0%}</td>"
        f"<td>{r['llr']:.1f}</td></tr>" for r in top)
    return (f"<p><strong>{n:,} mutations</strong> scored across {trnas} tRNAs.</p>"
            f"{rows}"
            f"<h4>Most shape-damaging mutations found</h4>"
            f"<table><thead><tr><th>mutation</th><th>tRNA</th>"
            f"<th>folding lost</th><th>surprise</th></tr></thead>"
            f"<tbody>{tbl}</tbody></table>"
            f"<p class='note'>Surprise is a log-ratio: more negative means the model "
            f"is more confident the original letter belongs there.</p>")


def main():
    os.makedirs(DOCS, exist_ok=True)
    md = markdown.Markdown(extensions=["tables", "fenced_code"])

    sections = []
    for fn, title, status in STEPS:
        path = os.path.join(NOTES, fn)
        if not os.path.exists(path):
            continue
        md.reset()
        body = demote(md.convert(open(path).read()))
        label, colour = BADGE[status]
        badge = (f"<span class='badge' style='background:{colour}'>{label}</span>"
                 if label else "")
        extra = ""
        if fn.startswith("02"):
            extra = figures_block()
        if fn.startswith("03"):
            extra = step3_block()
        anchor = fn[:2]
        sections.append(
            f"<section id='s{anchor}'><h2>{esc(title)} {badge}</h2>{body}{extra}</section>")

    nav = "\n".join(
        f"<a href='#s{fn[:2]}'>{esc(title.split(' — ')[0])}</a>"
        for fn, title, _ in STEPS if os.path.exists(os.path.join(NOTES, fn)))

    repo = git("remote", "get-url", "origin").replace(".git", "")
    repo_link = (f"<a href='{repo}'>source on GitHub</a>" if repo.startswith("http") else "")
    html = TEMPLATE.format(
        nav=nav, sections="\n".join(sections), commits=commit_rows(),
        updated=git("log", "-1", "--date=format:%d %B %Y", "--pretty=%ad", default=""),
        repo_link=repo_link,
    )
    with open(os.path.join(DOCS, "index.html"), "w") as fh:
        fh.write(html)
    print(f"wrote {os.path.join(DOCS, 'index.html')} ({len(html):,} bytes)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mitochondrial tRNA lab notebook</title>
<meta name="description" content="Using a bacterial DNA language model to predict which mutations break human mitochondrial tRNAs.">
<style>
  :root {{
    --bg: #fbfaf7; --fg: #1c1b19; --muted: #5f5c57; --rule: #e0ddd6;
    --accent: #1a5f7a; --card: #ffffff; --code: #f2f0eb;
  }}
  :root:not([data-theme="light"]) {{
    @media (prefers-color-scheme: dark) {{
      --bg: #16181a; --fg: #e8e6e3; --muted: #a09d98; --rule: #2e3133;
      --accent: #6fb3d0; --card: #1d2022; --code: #24272a;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #16181a; --fg: #e8e6e3; --muted: #a09d98; --rule: #2e3133;
    --accent: #6fb3d0; --card: #1d2022; --code: #24272a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 17px/1.65 ui-serif, Georgia, "Times New Roman", serif;
  }}
  .wrap {{ max-width: 760px; margin: 0 auto; padding: 0 16px 80px; }}
  header {{ padding: 56px 0 8px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; }}
  h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); margin: 0; font-size: 1.02rem; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 14px; padding: 14px 0 6px;
         border-bottom: 1px solid var(--rule); margin-bottom: 28px;
         font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.86rem; }}
  nav a {{ color: var(--accent); text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
  section {{ margin: 0 0 44px; }}
  h2 {{ font-size: 1.38rem; margin: 36px 0 12px; letter-spacing: -0.01em; }}
  h3, h4 {{ font-size: 1.05rem; margin: 26px 0 8px;
            font-family: ui-sans-serif, system-ui, sans-serif; }}
  p {{ margin: 0 0 14px; }}
  a {{ color: var(--accent); }}
  .badge {{ display: inline-block; vertical-align: middle; color: #fff;
            font: 600 11px/1 ui-sans-serif, system-ui, sans-serif;
            padding: 5px 9px; border-radius: 999px; margin-left: 6px;
            text-transform: uppercase; letter-spacing: 0.06em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 18px 0;
           font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); }}
  th {{ font-weight: 600; color: var(--muted); font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.05em; }}
  .mono, code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.88em; }}
  code {{ background: var(--code); padding: 1px 5px; border-radius: 4px; }}
  pre {{ background: var(--code); padding: 14px; border-radius: 8px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  figure {{ margin: 22px 0; }}
  figure img {{ width: 100%; height: auto; border: 1px solid var(--rule);
                border-radius: 8px; background: #fff; }}
  figcaption {{ color: var(--muted); font-size: 0.86rem; margin-top: 8px;
                font-family: ui-sans-serif, system-ui, sans-serif; }}
  .note {{ color: var(--muted); font-size: 0.9rem; }}
  blockquote {{ margin: 0 0 14px; padding-left: 16px; border-left: 3px solid var(--rule);
                color: var(--muted); }}
  footer {{ border-top: 1px solid var(--rule); padding-top: 18px; color: var(--muted);
            font: 14px/1.6 ui-sans-serif, system-ui, sans-serif; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Can an AI trained on bacteria predict mitochondrial disease?</h1>
  <p class="sub">A working lab notebook. Updated as we go.</p>
</header>
<nav>{nav}</nav>
{sections}
<section id="log">
  <h2>Every step, as it happened</h2>
  <p class="note">Each row is one commit. The notebook above is rebuilt from the
  repository after every step, so it never drifts from the code.</p>
  <table><thead><tr><th>commit</th><th>date</th><th>what changed</th></tr></thead>
  <tbody>{commits}</tbody></table>
</section>
<footer>Last updated {updated}. {repo_link}</footer>
</div>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
