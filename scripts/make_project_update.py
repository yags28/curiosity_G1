#!/usr/bin/env python3
"""Generate project_updatev1.pdf — a plain-language project update for a senior."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, PageBreak, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---- colors ----
NAVY = HexColor("#1a2a4f")
BLUE = HexColor("#2a5db0")
GREEN = HexColor("#1e7a44")
RED = HexColor("#b02a2a")
GREY = HexColor("#555555")
LIGHT = HexColor("#eef2f8")

styles = getSampleStyleSheet()

def style(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

title_st   = style("title", parent=styles["Title"], fontSize=24, textColor=NAVY, spaceAfter=4, leading=28)
sub_st     = style("sub", fontSize=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=2)
h1_st      = style("h1", fontSize=15, textColor=NAVY, spaceBefore=16, spaceAfter=6, leading=18, fontName="Helvetica-Bold")
h2_st      = style("h2", fontSize=12, textColor=BLUE, spaceBefore=10, spaceAfter=4, leading=15, fontName="Helvetica-Bold")
body_st    = style("body", fontSize=10, leading=15, spaceAfter=6, alignment=TA_LEFT)
bullet_st  = style("bullet", fontSize=10, leading=14, spaceAfter=2)
small_st   = style("small", fontSize=9, textColor=GREY, leading=12)
note_st    = style("note", fontSize=9.5, leading=13, textColor=GREY)

def H1(t): return Paragraph(t, h1_st)
def H2(t): return Paragraph(t, h2_st)
def P(t):  return Paragraph(t, body_st)
def rule(): return HRFlowable(width="100%", thickness=0.8, color=HexColor("#c3cee0"), spaceBefore=2, spaceAfter=8)

def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, bullet_st), leftIndent=10, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletColor=BLUE,
    )

def callout(text, color=LIGHT, txt_color=None):
    st = style("callout", fontSize=9.5, leading=13, textColor=txt_color or HexColor("#222222"))
    tbl = Table([[Paragraph(text, st)]], colWidths=[6.6*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBEFORE", (0,0), (0,-1), 3, BLUE),
    ]))
    return tbl

def data_table(rows, header=True, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    ts = [
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, HexColor("#c3cee0")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [None, LIGHT]),
    ]
    if header:
        ts += [
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("TEXTCOLOR", (0,0), (-1,0), HexColor("#ffffff")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(ts))
    return t

# ---- page furniture ----
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.9*inch, 0.5*inch, "CDL — Curiosity-Driven Learning for Unitree G1")
    canvas.drawRightString(7.6*inch, 0.5*inch, "Page %d" % doc.page)
    canvas.setStrokeColor(HexColor("#c3cee0"))
    canvas.line(0.9*inch, 0.65*inch, 7.6*inch, 0.65*inch)
    canvas.restoreState()

story = []

# ===================== TITLE =====================
story += [
    Spacer(1, 0.5*inch),
    Paragraph("Project Update — v1", title_st),
    Paragraph("Curiosity-Driven Learning for Emergent Tool Use", sub_st),
    Paragraph("Unitree G1 Humanoid Robot &middot; Purdue", sub_st),
    Spacer(1, 0.15*inch),
    Paragraph("Prepared by Yagnesh Dawankar &middot; June 15, 2026", sub_st),
    Spacer(1, 0.2*inch),
    rule(),
]

story += [
    H1("1. The Big Picture (Plain English)"),
    P("We are teaching a humanoid robot (the Unitree G1) to <b>figure out how to use tools on its own</b> — "
      "for example, using a stick to reach a far-away object, or standing on a box to press a high button. "
      "Instead of telling the robot exactly what to do, we reward it for being <b>curious</b>: it gets a bonus "
      "for trying things it hasn't seen before. Over millions of practice attempts in simulation, this curiosity "
      "pushes it to discover clever solutions by itself."),
    P("The ultimate goal is <b>sim-to-real transfer</b>: train entirely in a fast physics simulator, then run the "
      "learned behavior on the real robot. The project runs over 16 weeks and covers 5 increasingly hard tasks."),
    callout("<b>Where we are now:</b> Phase 2 (training) is underway. We have built all 5 task environments, "
            "trained working policies on Tasks 1 and 4, compressed them into small deployable networks "
            "(distillation), and run a deep investigation into why one task does not yet transfer between two "
            "different simulators. That investigation is the most important finding so far."),
]

# ===================== RESEARCH =====================
story += [
    H1("2. What We Researched — The Three Curiosity Methods"),
    P("\"Curiosity\" is a bonus reward that encourages exploration. The core idea: the robot keeps a small neural "
      "network that tries to <i>predict</i> something about each state it visits. When it visits a <b>new</b> "
      "situation, the prediction is bad — and that prediction error becomes a reward. So the robot is naturally "
      "drawn toward the unfamiliar. We compared three ways of doing this:"),
    H2("RND — Random Network Distillation"),
    bullets([
        "A fixed, random \"target\" network and a second \"predictor\" network that learns to copy it.",
        "Reward = how badly the predictor copies the target on the current state. New states = high error = high reward.",
        "The classic, simplest baseline.",
    ]),
    H2("DRND — Distributional RND"),
    bullets([
        "Uses an <b>ensemble</b> of several independent (target, predictor) pairs instead of one.",
        "Reward = the average prediction error across the whole ensemble.",
        "More stable and informative signal because several predictors must all agree a state is familiar.",
    ]),
    H2("RDD — Random Disagreement Distillation"),
    bullets([
        "One fixed target but <b>several predictors</b>; reward = how much the predictors <i>disagree</i> with each other.",
        "High disagreement means the state is novel. A bandwidth parameter (sigma) controls sensitivity.",
        "In theory elegant, but in our tests the disagreement signal faded too quickly (more below).",
    ]),
    callout("All three share the same plumbing in our code: a single factory "
            "<font face='Courier'>make_curiosity(cfg, obs_dim, device)</font> builds whichever method we ask for, "
            "and the PPO trainer talks to all of them through one common interface. This made head-to-head "
            "comparison clean and fair."),
]

# ===================== ARCHITECTURE =====================
story += [
    PageBreak(),
    H1("3. The Architecture & Models"),
    P("We use <b>PPO</b> (Proximal Policy Optimization), a standard and reliable reinforcement-learning algorithm, "
      "as the engine. The curiosity bonus is added on top of the task reward. The brain of the robot is an "
      "<b>Actor-Critic</b> network with an important twist:"),
    H2("Asymmetric Actor-Critic"),
    bullets([
        "<b>Actor (the policy that acts):</b> sees only what a real robot could sense — its own joints and body "
        "(proprioception), plus, in the newest version, the position of nearby objects. This keeps it deployable "
        "on real hardware.",
        "<b>Critic (the value estimator, used only in training):</b> is allowed to \"cheat\" and see privileged "
        "simulator-only information (exact object states, etc.). This makes training more stable. The critic is "
        "thrown away at deployment.",
        "We actually run <b>two critics</b> — one for the external task reward, one for the internal curiosity "
        "reward — so the two signals don't get tangled.",
        "The output is a 43-dimensional Gaussian — i.e. the robot controls 43 joint targets each step.",
    ]),
    data_table([
        ["Component", "Inputs (observation size)", "Notes"],
        ["Actor (policy)", "109 dims (original) -> 115 (closed-loop)", "Proprioception; +object pose in new version"],
        ["Critic (value)", "132 dims (original) -> 138 (closed-loop)", "Privileged sim-only info; training only"],
        ["Action output", "43-dim Gaussian", "Joint position targets"],
    ], col_widths=[1.5*inch, 2.7*inch, 2.4*inch]),
    Spacer(1, 6),
    callout("<b>Key file:</b> <font face='Courier'>src/utils/networks.py</font> defines the ActorCritic. The "
            "curiosity modules live in <font face='Courier'>src/curiosity/</font> (rnd.py, drnd.py, rdd.py), and "
            "the PPO training loop is in <font face='Courier'>src/agents/ppo.py</font>."),
    H2("How it runs"),
    P("Training spins up many copies of the robot in parallel inside <b>NVIDIA Isaac Lab / Isaac Sim 4.5</b> "
      "(a GPU physics simulator). On the big cluster we use 4096 robots at once for 200M steps; locally on an "
      "RTX 3050 (8 GB) we use a smaller 32-robot, 10M-step config to develop and debug."),
]

# ===================== TASKS =====================
story += [
    H1("4. The Five Tasks We Built"),
    P("Each task is a self-contained simulated scene with its own objects, success condition, and randomization. "
      "All five are implemented and pass their unit tests."),
    data_table([
        ["#", "Task", "What the robot must do", "Success condition", "Status"],
        ["1", "Distant Target", "Use a 90 cm stick to reach a far object", "Target moved >= 0.5 m", "Trained"],
        ["2", "Elevated Button", "Stand on a 20 kg box to reach a button at 1.8 m", "Button pressed", "Built"],
        ["3", "Occluded Retrieval", "Push a stick through a slot to fetch a hidden box", "Target moved >= 0.5 m", "Built"],
        ["4", "Weight Lever", "Use a plank + fulcrum to lever up a heavy box", "Heavy box lifted >= 0.10 m", "Trained"],
        ["5", "Composite", "Go through a gap, move a box, climb it, press button", "Button contact >= 2 N", "Built"],
    ], col_widths=[0.25*inch, 1.2*inch, 2.3*inch, 1.7*inch, 0.85*inch]),
    Spacer(1, 6),
    P("All tasks include <b>domain randomization</b> — we randomly vary friction, mass, and other physics "
      "properties every reset so the policy doesn't overfit to one exact setting (this is what helps real-world "
      "transfer). Tasks 2, 3, and 5 are built and unit-tested but await cluster compute to train at full scale."),
    callout("<b>Engineering fixes along the way:</b> PhysX requires CPU tensors and all-environment indices to set "
            "masses/friction; and the Isaac Lab URDF importer was missing a function in version 4.5, which we "
            "guarded so the robot model loads correctly.", color=HexColor("#fff6e6")),
]

# ===================== RESULTS =====================
story += [
    PageBreak(),
    H1("5. Training Results So Far"),
    H2("Task 1 — Comparing the three curiosity methods (10M steps each)"),
    data_table([
        ["Method", "Final success", "What happened"],
        ["RND", "~0% (likely)", "Policy never settled; no useful learning signal recorded"],
        ["DRND", "100%", "Sudden 'phase transition' at ~4.8M steps: 0% -> 44% -> 100%"],
        ["RDD", "0%", "Episodes collapsed to ~5 steps; disagreement signal decayed too fast"],
    ], col_widths=[1.1*inch, 1.3*inch, 4.0*inch]),
    Spacer(1, 6),
    callout("<b>Headline result:</b> <b>DRND is the clear winner.</b> It reached 100% success on Task 1, while RND "
            "and RDD failed. The DRND robot learned to solve the task in about 12 steps with a solid positive "
            "reward. This validates DRND as our default curiosity method going forward.", color=HexColor("#e8f5ec")),
    P("Task 4 (Weight Lever) was also trained successfully with DRND, reaching high success rates and producing a "
      "usable teacher policy."),
    H1("6. Distillation (Phase 3) — Shrinking the Policy for Deployment"),
    P("The training-time policy is big and relies on the privileged critic. For deployment we use <b>DAgger</b> "
      "(Dataset Aggregation), a method where a small <b>student</b> network watches the <b>teacher</b> and learns "
      "to imitate it, repeatedly correcting its own mistakes over 20 rounds."),
    bullets([
        "<b>Task 1 student:</b> hit 100% success by round 3 and held it through round 20. Training loss dropped "
        "from 332,000 down to 75. ~15 minutes total.",
        "<b>Task 4 student:</b> 100% success from round 1; loss dropped to under 1 (an even cleaner fit than Task 1).",
    ]),
    P("So inside the simulator, distillation works beautifully — we can compress a winning teacher into a small, "
      "real-robot-friendly student without losing performance."),
]

# ===================== TRANSFER =====================
story += [
    H1("7. The Hard Problem We Are Investigating: Sim-to-Sim Transfer"),
    P("Before risking the real robot, we test whether a policy trained in Isaac (PhysX physics) also works in a "
      "<b>different</b> simulator, MuJoCo. If a policy can't even survive a change of simulator, it certainly won't "
      "survive the real world. This is a cheap, honest stress test. <b>Task 1's policy currently fails it — it "
      "scores 0% in MuJoCo despite ~77-100% in Isaac.</b> We spent significant effort finding out exactly why."),
    H2("What we built to investigate"),
    bullets([
        "A <b>contract dumper</b> that records Isaac's exact joint ordering, limits, and default pose (Isaac "
        "interleaves left/right joints by kinematic depth — a subtle ordering bug source we caught).",
        "A <b>ground-truth rollout recorder</b> in Isaac (confirmed the student gets 9/10 there).",
        "A full <b>MuJoCo evaluation harness</b> that rebuilds the same observation, action mapping, and tools.",
    ]),
    H2("Bugs we found and fixed by careful comparison"),
    bullets([
        "<b>Foot-contact mismatch:</b> MuJoCo only reported contact on one ankle part; fixed so it matches Isaac "
        "<b>exactly</b> (observation difference 0.0).",
        "<b>Leg collapse:</b> the robot's legs sagged under MuJoCo's torque model; we stiffened the leg gains 20x to "
        "match how Isaac's actuators behave, so the robot stands at the right height.",
        "<b>Friction matched</b> on stick and target.",
    ]),
    callout("After fixing all the harness bugs, the observation and action contract is <b>verified identical</b> to "
            "Isaac (all 43 actions agree). And yet transfer is <b>still 0%</b>. That tells us the problem is "
            "<b>not a coding bug</b> — it is something deeper about the policy itself.", color=HexColor("#fdeef0")),
]

# ===================== ROOT CAUSE =====================
story += [
    PageBreak(),
    H1("8. Root Cause — Why Task 1 Doesn't Transfer (Confirmed)"),
    P("After step-by-step tracing in both simulators, we have a confident diagnosis:"),
    bullets([
        "<b>The winning Task 1 behavior is a ballistic lunge, not a careful reach.</b> Success in Isaac comes from "
        "<i>propelling the stick</i> at the target in a fast, semi-ballistic motion — the hands actually stay over a "
        "meter away.",
        "<b>The policy is 'bang-bang'.</b> Every one of the 43 joint commands is slammed to its maximum (±1700). "
        "It's an all-or-nothing motion finely tuned to PhysX's exact physics.",
        "<b>Such a motion is maximally sensitive to the physics engine.</b> The identical command sequence in MuJoCo "
        "produces a totally different trajectory: the robot launches off the ground (both feet leave within 2 steps) "
        "and the stick gets knocked sideways, away from the target.",
        "<b>The policy is blind and open-loop.</b> It only senses its own body, so once the grasp/launch goes wrong "
        "it cannot see or correct the mistake.",
    ]),
    callout("<b>Bottom line:</b> This is genuine <b>over-fitting to PhysX physics</b>, not a harness bug. A precise "
            "open-loop lunge that works in one simulator simply won't reproduce in another.", color=LIGHT),
    H2("Fixes we already tried (and what they taught us)"),
    data_table([
        ["Attempt", "Result", "Lesson"],
        ["Tanh + saturation penalty (smoother actions)", "Isaac still 100%, MuJoCo still 0%", "Smoothing is necessary but not sufficient"],
        ["Closed-loop obs (add object positions) + restitution randomization + 10M retrain", "Isaac ~77%, MuJoCo still 0/30", "Even seeing objects didn't fix a ballistic motion"],
    ], col_widths=[2.6*inch, 1.9*inch, 2.1*inch]),
    Spacer(1, 6),
    P("Importantly, smoothing the actions did <b>not</b> hurt Isaac success — so we can make the policy gentler for "
      "free. The remaining issue is the <i>nature of the motion</i> (a lunge), not its smoothness."),
]

# ===================== NEXT STEPS =====================
story += [
    H1("9. Next Steps"),
    H2("To make Task 1 transfer (teacher-side fix)"),
    bullets([
        "Add <b>action-rate and torque penalties directly in PPO</b> so the teacher learns a smooth, quasi-static "
        "reach instead of a violent lunge from the start.",
        "Apply <b>heavier domain randomization</b> during training: PD gains, restitution, contact stiffness, and "
        "control latency — so the policy can't rely on one exact physics setting.",
        "One ~4-hour retrain, then re-distill and re-evaluate in MuJoCo.",
        "<b>Alternative demonstration:</b> Task 4 (the lever) is naturally slow and quasi-static, not ballistic, so "
        "it is a much better candidate to show successful transfer first.",
    ]),
    H2("Project-level"),
    bullets([
        "<b>Critical blocker:</b> RCAC cluster allocation. Full-scale training of Tasks 2, 3, and 5 — and all of "
        "Phase 4 (sim-to-real) — depend on getting compute on the cluster.",
        "Once a transferable policy exists, extend the MuJoCo cross-validation to Task 4.",
    ]),
    rule(),
    H1("10. One-Paragraph Summary for Leadership"),
    P("We have a working research pipeline end-to-end: five tool-use tasks built and tested, three curiosity "
      "methods compared head-to-head (<b>DRND clearly wins, reaching 100% on Task 1</b>), and a distillation step "
      "that cleanly compresses winning policies for deployment. Our most valuable finding is a <b>rigorous, "
      "well-diagnosed understanding of why one policy doesn't yet transfer across simulators</b>: the learned "
      "motion is a physics-specific ballistic lunge from a blind, open-loop policy. We have verified this is a real "
      "learning/physics issue, not a software bug, and we have a concrete plan to fix it (smoother training rewards "
      "+ stronger domain randomization). The main external dependency is cluster compute to scale the remaining "
      "tasks."),
    Spacer(1, 0.2*inch),
    Paragraph("End of report &middot; project_updatev1", small_st),
]

doc = SimpleDocTemplate(
    "/home/kevin/projects/CDL/project_updatev1.pdf",
    pagesize=letter,
    leftMargin=0.9*inch, rightMargin=0.9*inch,
    topMargin=0.8*inch, bottomMargin=0.8*inch,
    title="CDL Project Update v1", author="Yagnesh Dawankar",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Wrote /home/kevin/projects/CDL/project_updatev1.pdf")
