"""17_update_map_page.py: updates docs/index.html for the revised paper (title, CPMI line, findings)."""
import re
PATH = "docs/index.html"
NEW_T = "Grey-Listed but Not Pricier? De-risking, Business Models and the Hidden Cost of Sending Money to Africa"
html = open(PATH, encoding="utf-8").read()
problems = []
html, n = re.subn(r"<h1>.*?</h1>", lambda _: f"<h1>{NEW_T}</h1>", html, count=1, flags=re.S)
if n == 0: problems.append("<h1> heading")
html, n = re.subn(r"(<meta name=\"description\" content=\"[^\"]*?the paper )'[^']*'", lambda m: m.group(1) + f"'{NEW_T}'", html, count=1)
if n == 0: print("Note: description tag not changed (wording differs)")
if "correspondents_index" not in html:
    m = re.search(r'\{label:"Balances abroad"\}\);', html)
    if m:
        add = ('\n    if (c.correspondents_index) h += `<p>Correspondent banking relationships (CPMI, 2011 = 100): '
               '${Object.entries(c.correspondents_index).map(([y, v]) => `${y}: <b>${v}</b>`).join(" · ")}</p>`;')
        html = html[:m.end()] + add + html[m.end():]
    else:
        problems.append("balances-abroad chart line")
new_li = ("<li>If a country's correspondent banking relationships (CPMI) halve, the quoted cost of sending USD 200 "
          "changes by <b>${f.correspondents_halve_service}</b> at service level: inconclusive, with moderate increases "
          "not ruled out.</li><li>If a country's banks' balances abroad (BIS) halve, the quoted cost changes by "
          "<b>${f.deposits_abroad_halve}</b>: no meaningful effect, but these balances do not track correspondent "
          "relationships.</li>")
if "correspondents_halve_service" not in html:
    starts = [m.start() for m in re.finditer(r"<li>", html)]
    pos = html.find("${f.deposits_abroad_halve}")
    if pos > 0 and starts:
        s = max(x for x in starts if x < pos); e = html.find("</li>", pos) + len("</li>")
        html = html[:s] + new_li + html[e:]
    else:
        problems.append("deposits-abroad findings bullet")
if problems:
    raise SystemExit("Not found, nothing changed: " + ", ".join(problems))
open(PATH, "w", encoding="utf-8").write(html)
print("docs/index.html updated")
