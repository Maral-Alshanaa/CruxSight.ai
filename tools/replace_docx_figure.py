# tools/replace_docx_figure.py
"""
Swap the picture belonging to one figure caption inside a .docx without touching layout.

Only the bytes of the embedded media part are replaced. The <wp:extent>/<a:ext> size, anchoring,
wrapping, cropping, caption, tracked changes and comments in document.xml are copied unchanged.
Never overwrites the input; always writes a new file.

  python tools/replace_docx_figure.py IN.docx --probe
  python tools/replace_docx_figure.py IN.docx OUT.docx --figure 3 --png results/figures/fig3_cp_recall.png
"""
import argparse
import re
import struct
import sys
import zipfile

from lxml import etree

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
R_EMBED = "{%s}embed" % NS["r"]
EMU = 914400


def paras(root):
    return root.findall(".//w:body//w:p", NS)


def ptext(p):  # includes text inside tracked insertions, excludes deleted text
    return "".join(t.text or "" for t in p.iter("{%s}t" % NS["w"]))


def rels(z):
    x = etree.fromstring(z.read("word/_rels/document.xml.rels"))
    return {e.get("Id"): e.get("Target") for e in x}


def pics(root):
    out = []
    ps = paras(root)
    for i, p in enumerate(ps):
        for blip in p.iterfind(".//a:blip", NS):
            ext = p.find(".//wp:extent", NS)
            cap = next((ptext(q) for q in (ps[i + 1: i + 4] + ps[max(0, i - 2): i]) if re.match(
                r"^\s*(fig\.?|figure)\s*\d+", ptext(q), re.I)), "")
            out.append(dict(rid=blip.get(R_EMBED), cx=int(ext.get("cx")), cy=int(ext.get("cy")),
                            caption=cap.strip()))
    return out


def png_size(b):
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", b[16:24])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--figure", type=int, default=3)
    ap.add_argument("--png")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.docx)
    root = etree.fromstring(z.read("word/document.xml"))
    rl = rels(z)
    P = pics(root)
    if a.probe:
        for p in P:
            print(f'{rl.get(p["rid"])}: {p["cx"]/EMU:.3f} x {p["cy"]/EMU:.3f} in | {p["caption"][:70]!r}')
        return
    hit = [p for p in P if re.match(rf"^\s*(fig\.?|figure)\s*{a.figure}\b", p["caption"], re.I)]
    if len(hit) != 1:
        sys.exit(f"ABORT: {len(hit)} pictures matched caption 'Figure {a.figure}'; run --probe")
    p = hit[0]
    target = "word/" + rl[p["rid"]].lstrip("/").removeprefix("word/")
    if not target.lower().endswith(".png"):
        sys.exit(f"ABORT: existing media is {target}, not .png; replace manually")
    new = open(a.png, "rb").read()
    w, h = png_size(new)
    r_old, r_new = p["cx"] / p["cy"], w / h
    print(f"target part: {target}\nframe: {p['cx']/EMU:.3f} x {p['cy']/EMU:.3f} in  (aspect {r_old:.4f})")
    print(f"new png: {w} x {h} px (aspect {r_new:.4f}; {w/(p['cx']/EMU):.0f} dpi at frame width)")
    if abs(r_old - r_new) / r_old > 0.01:
        sys.exit("ABORT: aspect ratio differs >1%; regenerate with --width/--height of the frame "
                 "or the picture will be stretched")
    users = [k for k, v in rl.items() if v.endswith(target.removeprefix("word/"))]
    n_use = sum(1 for q in P if rl.get(q["rid"]) == rl[p["rid"]])
    if len(users) > 1 or n_use > 1:
        sys.exit(f"ABORT: media part {target} is used by {max(len(users), n_use)} pictures; "
                 "replacing it would change the other figure(s) too")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as o:
        for it in z.infolist():
            o.writestr(it, new if it.filename == target else z.read(it.filename))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
