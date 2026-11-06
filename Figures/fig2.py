import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

W, H = 1120, 820
fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_facecolor("white")

FS, FSH, FSL = 7.4, 8.0, 8.6
LH, PAD = 22.0, 13.0
boxes, texts, names = [], [], []

def box(cx, ytop, width, lines, name, shaded=False, dashed=False, rule=False, minh=0):
    n = len(lines); h = max((n-1)*LH + 2*PAD + 6, minh)
    x0, x1, y0, y1 = cx-width/2, cx+width/2, ytop, ytop+h
    ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
                 facecolor="#efefef" if shaded else "white", edgecolor="black",
                 linewidth=0.9, linestyle=(0,(4,2.5)) if dashed else "solid", zorder=2))
    boxes.append((x0,y0,x1,y1)); names.append(name)
    ystart = (y0+y1)/2 - (n-1)*LH/2
    for i,(txt,bold) in enumerate(lines):
        yy = ystart + i*LH
        t = ax.text(cx, yy, txt, ha="center", va="center", zorder=3,
                    fontsize=FSH if bold else FS,
                    fontweight="bold" if bold else "normal", color="black")
        texts.append((t, len(boxes)-1))
        if rule and i == 0:
            ax.plot([cx-70, cx+70], [yy+6.0, yy+6.0], lw=0.7, color="#9a9a9a", zorder=3)
    return (x0,y0,x1,y1)

def arrow(x0,y0,x1,y1):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0), zorder=1,
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=0.9,
                        shrinkA=0, shrinkB=0, mutation_scale=9))
def line(x0,y0,x1,y1): ax.plot([x0,x1],[y0,y1], color="black", lw=0.9, zorder=1)

CX, LCX, RCX = 560, 300, 820
ax.plot([112,112],[22,800], color="#cccccc", lw=1.0, zorder=0)

b1 = box(CX, 22, 260, [("Submissions received",False),("n = 41",True)], "submissions")
bx = box(915, 22, 380, [("Excluded (n = 2, both Group A):",False),
                        ("1 code + role, no scale/case data",False),
                        ("1 complete response, duplicate participant code",False)],
         "excluded", dashed=True)
arrow(b1[2], (b1[1]+b1[3])/2, bx[0]-2, (b1[1]+b1[3])/2)

b2 = box(CX, 140, 260, [("Analysed sample",False),("n = 39",True)], "analysed")
arrow(CX, b1[3], CX, b2[1]-2)

line(CX, b2[3], CX, b2[3]+28); line(LCX, b2[3]+28, RCX, b2[3]+28)
arrow(LCX, b2[3]+28, LCX, b2[3]+56); arrow(RCX, b2[3]+28, RCX, b2[3]+56)
bA = box(LCX, b2[3]+56, 320, [("Group A — control",True),("risk band only",False),
                              ("21 allocated → 19 analysed",False)], "groupA")
bB = box(RCX, b2[3]+56, 320, [("Group B — explanation",True),("risk band + SHAP panel",False),
                              ("20 allocated → 20 analysed",False)], "groupB")
line(LCX,bA[3],LCX,bA[3]+30); line(RCX,bB[3],RCX,bB[3]+22); line(LCX,bA[3]+30,RCX,bA[3]+30)
arrow(CX, bA[3]+30, CX, bA[3]+58)
bT = box(CX, bA[3]+58, 400, [("Per-case task loop — 5 cases, fixed order",True),
                             ("1 low · 2 moderate · 2 high predicted risk",False),
                             ("Action checklist · free-text rationale",False),
                             ("Perceived accuracy, trust, confidence (5-pt)",False)],
         "taskloop", shaded=True)
arrow(CX, bT[3], CX, bT[3]+32)
bP = box(CX, bT[3]+32, 360, [("Post-task scales (Section 4.9)",False),
                             ("trust · competence · autonomy · usefulness · ease",False)], "scales")
line(CX,bP[3],CX,bP[3]+28); line(LCX,bP[3]+28,RCX,bP[3]+28)
arrow(LCX,bP[3]+28,LCX,bP[3]+56); arrow(RCX,bP[3]+28,RCX,bP[3]+56)
bQ = box(LCX, bP[3]+56, 320, [("Quantitative stream",True),("205 case-responses",False),
    ("− 6 blank, − 5 excluded duplicate",False),("→ 194 scored items",False),
    ("Intervention quality 0–2, blind",False),("Mann–Whitney U · Bonferroni",False)],
    "quant", rule=True)
bR = box(RCX, bP[3]+56, 320, [("Qualitative stream",True),("191 open-ended responses",False),
    ("(pooled case + survey fields)",False),("− 16 boilerplate → 175 analysed",False),
    ("Reflexive thematic analysis · 6 themes",False)], "qual", rule=True, minh=5*LH + 2*PAD + 6)

for lab, bb in (("Enrolment",b1),("Allocation",bA),("Task loop",bT),("Analysis",bQ)):
    ax.text(96, (bb[1]+bb[3])/2, lab, ha="right", va="center", style="italic",
            fontsize=FSL, color="#333333")

# ---------------- checks ----------------
fig.canvas.draw(); inv = ax.transData.inverted(); bad = 0
for t, bi in texts:
    e = t.get_window_extent(fig.canvas.get_renderer())
    (tx0,ty0) = inv.transform((e.x0,e.y0)); (tx1,ty1) = inv.transform((e.x1,e.y1))
    bx0,by0,bx1,by1 = boxes[bi]
    if tx0 < bx0+2 or tx1 > bx1-2 or min(ty0,ty1) < by0+1 or max(ty0,ty1) > by1-1:
        bad += 1; print(f"  TEXT OVERFLOW in {names[bi]}: {t.get_text()[:46]!r}")
for i in range(len(boxes)):
    for j in range(i+1, len(boxes)):
        a,b = boxes[i], boxes[j]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            bad += 1; print(f"  BOX OVERLAP: {names[i]} and {names[j]}")
for i,(x0,y0,x1,y1) in enumerate(boxes):
    if x0 < 118: bad += 1; print(f"  {names[i]} crosses the stage rail (x0={x0:.0f})")
    if x1 > W-6: bad += 1; print(f"  {names[i]} runs off the canvas (x1={x1:.0f})")
print(f"{len(texts)} labels and {len(boxes)} boxes checked — {bad} problem(s)")
fig.savefig("Figure2_participant_flow.png", dpi=300, bbox_inches="tight",
            pad_inches=0.07, facecolor="white")
