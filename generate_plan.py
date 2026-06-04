"""
Generates plan.pdf — a comprehensive, step-by-step implementation plan for the
Curiosity-Driven RL for Emergent Tool Use on Humanoid Robot (Unitree G1) project.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.colors import HexColor

# ── Color palette ──────────────────────────────────────────────────────────────
RED       = HexColor("#C0392B")
DARK_NAVY = HexColor("#1A2238")
MID_BLUE  = HexColor("#2E4057")
LIGHT_BG  = HexColor("#F4F6FB")
ACCENT    = HexColor("#E8F0FE")
GRAY_LINE = HexColor("#D0D3DA")
GREEN     = HexColor("#1E8449")
ORANGE    = HexColor("#D35400")
PURPLE    = HexColor("#6C3483")
TABLE_HDR = HexColor("#2E4057")
TABLE_ALT = HexColor("#EAF0FA")

# ── Page layout ────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch

def make_doc(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=0.75 * inch,
        title="CDL Project Implementation Plan",
        author="Purdue University — School of Mechanical Engineering",
    )
    return doc

# ── Style definitions ──────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

styles = {
    "cover_title": S("cover_title", "Title",
                     fontSize=28, leading=34, textColor=DARK_NAVY,
                     alignment=TA_CENTER, spaceAfter=8),
    "cover_sub":   S("cover_sub", "Normal",
                     fontSize=13, leading=18, textColor=MID_BLUE,
                     alignment=TA_CENTER, spaceAfter=6),
    "cover_label": S("cover_label", "Normal",
                     fontSize=10, leading=14, textColor=RED,
                     alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold"),
    "cover_meta":  S("cover_meta", "Normal",
                     fontSize=9, leading=13, textColor=HexColor("#555555"),
                     alignment=TA_CENTER),

    "h1": S("h1", "Heading1",
            fontSize=16, leading=20, textColor=RED,
            fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
            borderPad=0),
    "h2": S("h2", "Heading2",
            fontSize=13, leading=17, textColor=DARK_NAVY,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4),
    "h3": S("h3", "Heading3",
            fontSize=11, leading=15, textColor=MID_BLUE,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3),
    "h4": S("h4", "Normal",
            fontSize=10, leading=13, textColor=MID_BLUE,
            fontName="Helvetica-BoldOblique", spaceBefore=8, spaceAfter=2),

    "body": S("body", "Normal",
              fontSize=9.5, leading=14, textColor=HexColor("#222222"),
              alignment=TA_JUSTIFY, spaceAfter=5),
    "body_left": S("body_left", "Normal",
                   fontSize=9.5, leading=14, textColor=HexColor("#222222"),
                   alignment=TA_LEFT, spaceAfter=5),
    "bullet": S("bullet", "Normal",
                fontSize=9.5, leading=13, textColor=HexColor("#222222"),
                leftIndent=14, bulletIndent=0, spaceAfter=3),
    "sub_bullet": S("sub_bullet", "Normal",
                    fontSize=9, leading=12, textColor=HexColor("#444444"),
                    leftIndent=28, bulletIndent=14, spaceAfter=2),
    "note": S("note", "Normal",
              fontSize=8.5, leading=12, textColor=HexColor("#555555"),
              leftIndent=12, fontName="Helvetica-Oblique", spaceAfter=4),
    "table_hdr": S("table_hdr", "Normal",
                   fontSize=8.5, leading=11, textColor=colors.white,
                   fontName="Helvetica-Bold", alignment=TA_CENTER),
    "table_cell": S("table_cell", "Normal",
                    fontSize=8.5, leading=11, textColor=DARK_NAVY,
                    alignment=TA_LEFT),
    "table_cell_c": S("table_cell_c", "Normal",
                      fontSize=8.5, leading=11, textColor=DARK_NAVY,
                      alignment=TA_CENTER),
    "risk_high":  S("risk_high", "Normal",
                    fontSize=8.5, leading=11, textColor=RED,
                    fontName="Helvetica-Bold", alignment=TA_CENTER),
    "risk_med":   S("risk_med", "Normal",
                    fontSize=8.5, leading=11, textColor=ORANGE,
                    fontName="Helvetica-Bold", alignment=TA_CENTER),
    "risk_low":   S("risk_low", "Normal",
                    fontSize=8.5, leading=11, textColor=GREEN,
                    fontName="Helvetica-Bold", alignment=TA_CENTER),
    "phase_hdr":  S("phase_hdr", "Normal",
                    fontSize=11, leading=14, textColor=colors.white,
                    fontName="Helvetica-Bold", alignment=TA_CENTER),
    "week_label": S("week_label", "Normal",
                    fontSize=9, leading=12, textColor=DARK_NAVY,
                    fontName="Helvetica-Bold"),
    "caption":    S("caption", "Normal",
                    fontSize=8, leading=11, textColor=HexColor("#666666"),
                    alignment=TA_CENTER, fontName="Helvetica-Oblique",
                    spaceBefore=2, spaceAfter=6),
    "toc_entry":  S("toc_entry", "Normal",
                    fontSize=9.5, leading=14, textColor=MID_BLUE),
    "metric_val": S("metric_val", "Normal",
                    fontSize=9, leading=12, textColor=GREEN,
                    fontName="Helvetica-Bold", alignment=TA_CENTER),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def HR(color=GRAY_LINE, width=1):
    return HRFlowable(width="100%", thickness=width, color=color,
                      spaceAfter=6, spaceBefore=6)

def P(text, style="body"):
    return Paragraph(text, styles[style])

def B(text, indent=0):
    if indent == 0:
        return Paragraph(f"• {text}", styles["bullet"])
    return Paragraph(f"◦ {text}", styles["sub_bullet"])

def SP(h=6):
    return Spacer(1, h)

def tbl(data, col_widths, row_styles=None, row_height=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  TABLE_HDR),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8.5),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT]),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("GRID",         (0, 0), (-1, -1), 0.4, GRAY_LINE),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])
    if row_styles:
        for rs in row_styles:
            ts.add(*rs)
    t.setStyle(ts)
    return t

def phase_box(phase_num, title, weeks, color=TABLE_HDR):
    data = [[P(f"PHASE {phase_num}", "phase_hdr")],
            [P(title, "phase_hdr")],
            [P(weeks, "caption")]]
    t = Table(data, colWidths=[1.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), color),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX",           (0, 0), (-1, -1), 1, colors.white),
    ]))
    return t

# ── Section builders ──────────────────────────────────────────────────────────

def build_cover():
    elems = []
    elems.append(SP(60))
    elems.append(P("IMPLEMENTATION PLAN", "cover_label"))
    elems.append(SP(4))
    elems.append(P("Curiosity-Driven Reinforcement Learning<br/>for Emergent Tool Use on a Humanoid Robot", "cover_title"))
    elems.append(SP(10))
    elems.append(HR(RED, 2))
    elems.append(SP(10))
    elems.append(P("Platform: Unitree G1 EDU Humanoid  ·  Simulator: NVIDIA Isaac Lab", "cover_sub"))
    elems.append(SP(30))

    meta = [
        ["Institution",   "Purdue University — School of Mechanical Engineering"],
        ["Location",      "West Lafayette, IN 47907"],
        ["Document",      "Full Step-by-Step Implementation Plan"],
        ["Project Duration", "16 Weeks (4 Phases)"],
        ["Compute Budget","~200–500 GPU-hours (NVIDIA A100/H100)"],
        ["Date",          "May 2026"],
    ]
    meta_data = [[P(k, "cover_meta"), P(v, "cover_meta")] for k, v in meta]
    t = Table(meta_data, colWidths=[2.0*inch, 4.0*inch])
    t.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), DARK_NAVY),
        ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#444444")),
        ("ALIGN",     (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING",(0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, GRAY_LINE),
    ]))
    elems.append(t)
    elems.append(SP(40))
    elems.append(HR(GRAY_LINE))
    elems.append(P("This document provides an exhaustive, step-by-step plan for implementing the Curiosity-Driven "
                   "RL project. Every sub-task, dependency, technical detail, software artifact, hardware procedure, "
                   "evaluation criterion, risk, and mitigation is captured so that no work item is left ambiguous.",
                   "note"))
    elems.append(PageBreak())
    return elems


def build_toc():
    elems = []
    elems.append(P("Table of Contents", "h1"))
    elems.append(HR())
    sections = [
        ("1", "Project Overview & Research Goals"),
        ("2", "Environment & Hardware Setup (Pre-Phase)"),
        ("3", "Phase 1 — Environment Design in Isaac Lab  (Weeks 1–3)"),
        ("3.1", "3.1  Software Installation & Infrastructure"),
        ("3.2", "3.2  G1 URDF Integration & Validation"),
        ("3.3", "3.3  Task Environment Implementation (5 Tasks)"),
        ("3.4", "3.4  Object Randomisation & Reward Specification"),
        ("3.5", "3.5  Phase 1 Deliverables & Exit Criteria"),
        ("4", "Phase 2 — Curiosity-Driven RL Training  (Weeks 4–9)"),
        ("4.1", "4.1  Algorithm Implementation — RND / DRND / RDD Module"),
        ("4.2", "4.2  Asymmetric Actor-Critic Architecture"),
        ("4.3", "4.3  Observation & Action Space Specification"),
        ("4.4", "4.4  Reward Annealing & Curriculum Staging"),
        ("4.5", "4.5  Training Infrastructure & Cluster Setup"),
        ("4.6", "4.6  Hyperparameter Sweeps & Ablation Runs"),
        ("4.7", "4.7  Emergence Analysis & Logging"),
        ("4.8", "4.8  Phase 2 Deliverables & Exit Criteria"),
        ("5", "Phase 3 — Policy Distillation & Robustification  (Weeks 10–11)"),
        ("5.1", "5.1  Phase A — Behaviour Cloning via DAgger"),
        ("5.2", "5.2  Phase B — Residual RL Refinement"),
        ("5.3", "5.3  Sim-to-Sim Cross-Validation in MuJoCo"),
        ("5.4", "5.4  Phase 3 Deliverables & Exit Criteria"),
        ("6", "Phase 4 — Sim-to-Real Transfer  (Weeks 12–14)"),
        ("6.1", "6.1  Domain Randomisation Ranges"),
        ("6.2", "6.2  Safety Protocol & Gantry Setup"),
        ("6.3", "6.3  Hardware Deployment Procedure"),
        ("6.4", "6.4  Real-World Evaluation Protocol"),
        ("6.5", "6.5  Phase 4 Deliverables & Exit Criteria"),
        ("7", "Phase 5 — Paper Writing & Code Release  (Weeks 15–16)"),
        ("8", "Evaluation Plan & Success Metrics"),
        ("9", "Ablation Study Design"),
        ("10", "Risk Register & Mitigation Strategies"),
        ("11", "Required Resources Checklist"),
        ("12", "Literature Positioning & Novelty Arguments"),
        ("13", "Week-by-Week Master Schedule"),
    ]
    for num, title in sections:
        indent = 14 if "." in num and num != "1" else 0
        dot_num = f"<b>{num}</b>" if "." not in num else num
        elems.append(Paragraph(
            f"<para leftIndent='{indent}'>{dot_num} &nbsp;&nbsp; {title}</para>",
            styles["toc_entry"]))
    elems.append(PageBreak())
    return elems


def build_overview():
    elems = []
    elems.append(P("1.  Project Overview & Research Goals", "h1"))
    elems.append(HR())
    elems.append(P("<b>Project Title:</b> Curiosity-Driven Reinforcement Learning for Emergent Tool Use "
                   "on a Humanoid Robot", "body"))
    elems.append(P("<b>Platform:</b> Unitree G1 EDU (43 DOF, Dex3-1 dexterous hands, NVIDIA Jetson Orin NX "
                   "onboard, 275 TOPS)", "body"))
    elems.append(P("<b>Simulator:</b> NVIDIA Isaac Lab (Isaac Sim 4.x) with GPU-parallelised environments", "body"))
    elems.append(SP(6))

    elems.append(P("1.1  Core Thesis", "h2"))
    elems.append(P("A full-sized humanoid robot can discover tool-use behaviours <i>purely through curiosity-driven "
                   "reinforcement learning</i>, without demonstrations, pre-programmed heuristics, or dense reward "
                   "shaping for tool interaction.  The robot must independently realise that nearby objects can "
                   "extend its capabilities — standing on a box to reach an elevated button, hooking a stick to "
                   "retrieve an occluded item — through intrinsic motivation alone.", "body"))

    elems.append(P("1.2  Three Core Research Questions", "h2"))
    rq_data = [
        [P("RQ", "table_hdr"), P("Question", "table_hdr"), P("Success Criterion", "table_hdr")],
        [P("RQ1\nEmergence", "table_cell_c"),
         P("Can a humanoid agent discover tool-use behaviours in sparse-reward environments purely through "
           "intrinsic curiosity, without demonstrations or task-specific reward shaping?", "table_cell"),
         P("≥60% of seeds independently discover a tool-use solution on ≥1 task", "table_cell")],
        [P("RQ2\nWhole-Body", "table_cell_c"),
         P("Does the humanoid morphology enable qualitatively different tool-use strategies vs. "
           "fixed-base manipulators — e.g., carrying a tool while walking or repositioning to gain leverage?", "table_cell"),
         P("Agent exploits bipedal locomotion in ≥1 discovered strategy", "table_cell")],
        [P("RQ3\nTransfer", "table_cell_c"),
         P("Can curiosity-trained policies transfer to the physical G1 via domain randomisation "
           "and asymmetric actor-critic, and do emergent tool-use behaviours survive the reality gap?", "table_cell"),
         P("≥30% success rate on physical G1 hardware", "table_cell")],
    ]
    elems.append(tbl(rq_data, [0.8*inch, 3.3*inch, 2.1*inch]))
    elems.append(SP(4))

    elems.append(P("1.3  Why This is Novel", "h2"))
    novelty = [
        "No published paper demonstrates <i>emergent</i> tool use on a bipedal humanoid trained from scratch "
        "with pure curiosity-driven RL, no demonstrations, and only sparse extrinsic rewards.",
        "HumanoidBench (RSS 2024) explicitly shows that state-of-the-art RL algorithms — including "
        "intrinsic-motivation baselines — <i>fail</i> on hard humanoid whole-body manipulation tasks. "
        "This project directly attacks that open problem.",
        "All existing G1 work (ASAP, VideoMimic, DemoHLM, HOMIE) relies on motion-capture data, "
        "teleoperation, or at least one demonstration. This work uses <b>zero demonstrations</b>.",
        "The combination of (humanoid bipedal) × (curiosity-driven intrinsic reward) × (no demos) × "
        "(sparse extrinsic reward only) × (emergent tool use) is unoccupied in the 2023–2026 literature.",
        "Sim-to-real deployment on the physical Unitree G1 is a high-cost, high-reward differentiator: "
        "almost all curiosity papers stop at simulation.",
    ]
    for n in novelty:
        elems.append(B(n))
    elems.append(SP(4))

    elems.append(P("1.4  Expected Contributions", "h2"))
    contribs = [
        "<b>First demonstration</b> of curiosity-driven emergent tool use on a full-sized bipedal humanoid.",
        "<b>Open-source task suite</b> (Isaac Lab + G1 URDF) for benchmarking tool-use emergence in humanoid RL.",
        "<b>Empirical analysis</b> of RND vs. DRND vs. RDD vs. ICM vs. QFLEX curiosity mechanisms in high-DOF "
        "embodied settings.",
        "<b>Sim-to-real pipeline</b> for multi-phase whole-body behaviours extending the OmniXtreme two-stage "
        "architecture to manipulation-centric tasks.",
        "<b>Negative-result contribution</b> if tool use does not emerge: a rigorous characterisation of the "
        "'humanoid hard-exploration' problem — itself publishable as a benchmark akin to Montezuma's Revenge.",
    ]
    for c in contribs:
        elems.append(B(c))
    elems.append(PageBreak())
    return elems


def build_pre_phase():
    elems = []
    elems.append(P("2.  Environment & Hardware Setup  (Pre-Phase — Before Week 1)", "h1"))
    elems.append(HR())
    elems.append(P("All items in this section must be completed <b>before</b> Phase 1 begins. "
                   "These are prerequisites, not tracked in the 16-week timeline.", "body"))
    elems.append(SP(4))

    elems.append(P("2.1  Accounts, Access & Keys", "h2"))
    items = [
        "Obtain Purdue RCAC cluster accounts (Anvil, Gilbreth, or Negishi) for all team members.",
        "Request GPU allocation: ≥4 NVIDIA A100 or H100 nodes for parallel Isaac Lab training.",
        "Verify SSH key-pair setup and two-factor authentication on the cluster.",
        "Register for an NVIDIA Omniverse / Isaac Lab licence (free for academic use).",
        "Create a shared GitHub/GitLab private repository with main, dev, and experiment branches.",
        "Set up Weights & Biases (wandb) project for experiment tracking — create API keys for all members.",
        "Set up a shared network drive or S3 bucket for checkpoint storage (≥1 TB recommended).",
    ]
    for i in items: elems.append(B(i))

    elems.append(P("2.2  Local Development Machines", "h2"))
    items = [
        "Each workstation should have ≥1 NVIDIA GPU (RTX 3090 or better) for local debugging.",
        "Install Ubuntu 22.04 LTS (required by Isaac Lab and ROS 2 Humble).",
        "Install NVIDIA driver 525+, CUDA 12.1+, cuDNN 8.9+.",
        "Install Docker 24+ with NVIDIA Container Toolkit for reproducible environments.",
        "Clone NVIDIA Isaac Lab repository and run the built-in sanity-check scene.",
        "Clone unitree_rl_gym (official Unitree RL repository) and run the G1 standing task.",
        "Install MuJoCo 3.x locally for sim-to-sim cross-validation (separate conda env).",
        "Install TensorRT for onboard inference testing (match Jetson Orin NX version).",
    ]
    for i in items: elems.append(B(i))

    elems.append(P("2.3  Hardware Inventory Check", "h2"))
    hw_data = [
        [P("Item", "table_hdr"), P("Required Spec", "table_hdr"), P("Status Check", "table_hdr")],
        [P("Unitree G1 EDU robot", "table_cell"),
         P("43 DOF, Dex3-1 hands, Jetson Orin NX", "table_cell"),
         P("Confirm availability; if to-procure, initiate order immediately", "table_cell")],
        [P("Safety gantry / overhead tether", "table_cell"),
         P("Load-rated for ≥70 kg robot + dynamic loads", "table_cell"),
         P("Inspect mount points; load-test before any real-robot run", "table_cell")],
        [P("Tool objects", "table_cell"),
         P("Wooden blocks 30×30×30 cm (1–3 kg); PVC rods 80–120 cm; foam wedges", "table_cell"),
         P("Fabricate or procure; label with mass/friction properties", "table_cell")],
        [P("Ethernet switch + cables", "table_cell"),
         P("1 Gbps for unitree_sdk2_python communication", "table_cell"),
         P("Confirm low-latency (<1 ms) on test bench", "table_cell")],
        [P("E-stop remote", "table_cell"),
         P("Wireless, <50 ms latency", "table_cell"),
         P("Test before any untethered robot motion", "table_cell")],
        [P("RGB-D camera (external)", "table_cell"),
         P("Optional — for environment monitoring, not policy input", "table_cell"),
         P("Check mount positions relative to task scene", "table_cell")],
    ]
    elems.append(tbl(hw_data, [1.5*inch, 2.2*inch, 2.5*inch]))

    elems.append(P("2.4  Software Dependency Versions (Pin These)", "h2"))
    sw_data = [
        [P("Package", "table_hdr"), P("Version", "table_hdr"), P("Notes", "table_hdr")],
        [P("Isaac Lab", "table_cell"),       P("4.x (Isaac Sim 4.x backend)", "table_cell"), P("GPU-parallel RL environments", "table_cell")],
        [P("unitree_rl_gym", "table_cell"),  P("Latest main (2025)", "table_cell"),           P("G1 URDF + PPO boilerplate", "table_cell")],
        [P("rsl_rl", "table_cell"),          P("1.0.x", "table_cell"),                        P("PPO implementation used by unitree_rl_gym", "table_cell")],
        [P("MuJoCo", "table_cell"),          P("3.x", "table_cell"),                          P("Sim-to-sim cross-validation", "table_cell")],
        [P("TensorRT", "table_cell"),        P("8.6+ (match Jetson Orin NX SDK)", "table_cell"), P("Onboard inference optimisation", "table_cell")],
        [P("unitree_sdk2_python", "table_cell"), P("Latest", "table_cell"),                   P("Hardware communication layer", "table_cell")],
        [P("ROS 2 Humble", "table_cell"),    P("Humble Hawksbill", "table_cell"),             P("Hardware state/command bridge", "table_cell")],
        [P("PyTorch", "table_cell"),         P("2.1+", "table_cell"),                         P("Neural network training", "table_cell")],
        [P("wandb", "table_cell"),           P("Latest", "table_cell"),                        P("Experiment tracking", "table_cell")],
    ]
    elems.append(tbl(sw_data, [1.7*inch, 2.0*inch, 2.5*inch]))
    elems.append(PageBreak())
    return elems


def build_phase1():
    elems = []
    elems.append(P("3.  Phase 1 — Environment Design in Isaac Lab  (Weeks 1–3)", "h1"))
    elems.append(HR())
    elems.append(P("Goal: Deliver a fully validated Isaac Lab environment suite containing all 5 tasks, "
                   "with the G1 URDF integrated, procedural object randomisation implemented, and training "
                   "infrastructure tested on the cluster (≥1 GPU, ≥256 parallel envs confirmed).", "body"))
    elems.append(SP(6))

    # 3.1
    elems.append(P("3.1  Software Installation & Infrastructure  (Week 1, Days 1–3)", "h2"))
    steps = [
        ("Install Isaac Lab on cluster", [
            "Pull the official NVIDIA Isaac Lab Docker image onto the cluster.",
            "Configure SLURM job scripts for multi-GPU Isaac Lab training (4096 envs target).",
            "Run the provided Isaac Lab sanity-check scene; confirm GPU utilisation ≥80%.",
            "Install conda environment with all Python dependencies (torch, rsl_rl, wandb, numpy).",
            "Install unitree_rl_gym and run G1 standing task — confirm the G1 URDF loads without errors.",
        ]),
        ("Set up version control & CI", [
            "Create project repository structure: src/envs/, src/agents/, src/distill/, src/sim2real/, configs/, scripts/.",
            "Add pre-commit hooks: black formatter, flake8 linter, unit-test runner.",
            "Set up GitHub Actions (or cluster-native CI) to run unit tests on every push.",
            "Tag commit v0.0.1 — baseline: Isaac Lab confirmed running with G1 URDF.",
        ]),
        ("Set up experiment tracking", [
            "Create wandb project 'CDL-HumanoidToolUse'.",
            "Define standard logging keys: episode_reward, intrinsic_reward, extrinsic_reward, "
            "success_rate, tool_contact_count, alpha_annealing.",
            "Write a lightweight experiment launcher script (launch.py) that accepts a YAML config and logs to wandb.",
            "Test wandb sweep agent on dummy environment.",
        ]),
    ]
    for title, sub in steps:
        elems.append(P(f"<b>{title}</b>", "h3"))
        for s in sub: elems.append(B(s))

    # 3.2
    elems.append(P("3.2  G1 URDF Integration & Validation  (Week 1, Days 4–5)", "h2"))
    steps32 = [
        "Import G1 URDF (43 DOF) into Isaac Lab using the unitree_rl_gym URDF path.",
        "Verify all 43 joints are correctly parsed: hip, knee, ankle (6 DOF per leg), waist (3 DOF), "
        "shoulder, elbow, wrist (7 DOF per arm), Dex3-1 finger joints.",
        "Confirm Dex3-1 hand joint limits, friction, and gear ratio are correctly set from the hardware datasheet.",
        "Implement a PD controller test: command all joints to zero position; verify no physics instabilities.",
        "Measure G1 reach envelope in simulation (should be ≈1.6 m max reach): confirm it matches the proposal spec.",
        "Confirm IMU sensor (orientation, angular velocity) is correctly attached to the pelvis link.",
        "Confirm wrist force-torque sensor is correctly attached to both wrists.",
        "Confirm the head camera (64×64 egocentric depth image at 50 Hz) is correctly placed and outputs valid depth.",
        "Run 100-step rollout with random actions; confirm no NaN in observations or actions.",
        "Document joint indices, observation vector layout, and action space bounds in a dedicated doc-string / README.",
    ]
    for s in steps32: elems.append(B(s))

    # 3.3
    elems.append(P("3.3  Task Environment Implementation — All 5 Tasks  (Weeks 1–2)", "h2"))
    elems.append(P("Each task is implemented as a separate Isaac Lab environment class inheriting from a shared "
                   "<code>ToolUseBase</code> environment. The base class handles G1 spawning, physics settings, "
                   "observation computation, reward computation, and reset logic.", "body"))

    task_data = [
        [P("Task", "table_hdr"), P("Goal Description", "table_hdr"),
         P("Latent Solution", "table_hdr"), P("Difficulty", "table_hdr"),
         P("Key Implementation Details", "table_hdr")],
        [P("1. Distant Target", "table_cell"),
         P("Touch a target mounted beyond arm's reach across a 0.6 m gap", "table_cell"),
         P("Pick up a long stick (80–120 cm PVC rod) to bridge the gap", "table_cell"),
         P("Easy", "table_cell"),
         P("Stick must be placeable anywhere; target is a force-sensitive pad. Use as curriculum Stage 1.", "table_cell")],
        [P("2. Elevated Button", "table_cell"),
         P("Press a button mounted 2.0 m high (G1 max reach ≈1.6 m)", "table_cell"),
         P("Push a box underneath, climb onto it, then press button", "table_cell"),
         P("Medium", "table_cell"),
         P("Box must support G1 weight (≥80 kg). Implement contact-force check on box top surface.", "table_cell")],
        [P("3. Occluded Retrieval", "table_cell"),
         P("Retrieve an object placed behind a narrow barrier/gap (15 cm slot)", "table_cell"),
         P("Use stick/rod to push or hook the object out", "table_cell"),
         P("Medium", "table_cell"),
         P("Barrier is rigid. Object must move ≥0.5 m to count as retrieved. Track object displacement.", "table_cell")],
        [P("4. Weight Lever", "table_cell"),
         P("Lift a 10 kg object beyond G1's direct payload capacity", "table_cell"),
         P("Use a plank as a lever with a fulcrum object", "table_cell"),
         P("Hard", "table_cell"),
         P("Implement torque/contact sensing on plank. Fulcrum object must be correctly placed by agent.", "table_cell")],
        [P("5. Composite", "table_cell"),
         P("Navigate past an obstacle, then reach an elevated goal", "table_cell"),
         P("Sequence: carry box → place → climb → reach", "table_cell"),
         P("Hard", "table_cell"),
         P("Combine Elevated Button + locomotion. Long episode horizon (3000 steps). Add to curriculum last.", "table_cell")],
    ]
    elems.append(tbl(task_data, [0.9*inch, 1.4*inch, 1.4*inch, 0.6*inch, 1.9*inch]))
    elems.append(SP(4))

    elems.append(P("For each task, the implementation checklist is:", "h3"))
    task_impl = [
        "Write environment class (e.g., <code>ElevatedButtonEnv</code>) with <code>__init__</code>, "
        "<code>reset()</code>, <code>step()</code>, <code>compute_observations()</code>, "
        "<code>compute_reward()</code>.",
        "Spawn all objects (goal, tools, distractors) using Isaac Lab's asset API.",
        "Implement procedural randomisation of object positions, orientations, masses, and friction "
        "(see Section 3.4).",
        "Implement binary extrinsic reward: +1 only on goal completion, 0 otherwise; time penalty −0.001/step.",
        "Implement success detection logic (force threshold, object displacement, button press signal).",
        "Write unit test: reset 100 times, confirm no interpenetration or NaN observations.",
        "Run 10-minute training run with random policy and confirm ≤0.1% accidental success rate "
        "(task must be non-trivially hard).",
        "Visually inspect 5 rollouts in the Isaac Lab viewer to confirm physics looks correct.",
        "Log task-specific diagnostic: tool_contact_events, goal_proximity_over_time.",
    ]
    for s in task_impl: elems.append(B(s))

    # 3.4
    elems.append(P("3.4  Object Randomisation & Reward Specification  (Week 2)", "h2"))
    elems.append(P("Randomisation prevents the policy from memorising fixed object configurations and "
                   "forces learning a generalised tool-use strategy.", "body"))
    rand_data = [
        [P("Property", "table_hdr"), P("Range", "table_hdr"), P("Distribution", "table_hdr"), P("Purpose", "table_hdr")],
        [P("Object position (XY)", "table_cell"),  P("±30% of nominal", "table_cell"),  P("Uniform", "table_cell"), P("Prevent memorisation of exact location", "table_cell")],
        [P("Object orientation", "table_cell"),    P("Full SO(2) yaw randomisation", "table_cell"), P("Uniform", "table_cell"), P("Force orientation-agnostic grasping", "table_cell")],
        [P("Object mass", "table_cell"),           P("±30% of nominal", "table_cell"),  P("Uniform", "table_cell"), P("Generalise to real object mass variance", "table_cell")],
        [P("Object friction", "table_cell"),       P("±50% of nominal", "table_cell"),  P("Uniform", "table_cell"), P("Sim-to-real friction gap", "table_cell")],
        [P("Object colour/texture", "table_cell"), P("Random from 8-colour palette", "table_cell"), P("Categorical", "table_cell"), P("Prevent visual cue overfitting", "table_cell")],
        [P("Goal position", "table_cell"),         P("±10% of nominal (height fixed)", "table_cell"), P("Uniform", "table_cell"), P("Slight variance for robustness", "table_cell")],
        [P("Distractor objects", "table_cell"),    P("1–3 random objects in scene", "table_cell"), P("Random choice", "table_cell"), P("Prevent tool identification by exclusion", "table_cell")],
    ]
    elems.append(tbl(rand_data, [1.4*inch, 1.4*inch, 0.9*inch, 2.5*inch]))
    elems.append(SP(4))

    elems.append(P("<b>Extrinsic Reward Design (Binary Sparse):</b>", "h3"))
    elems.append(B("r_extrinsic = +1.0  upon goal completion (task-specific success condition), 0 otherwise."))
    elems.append(B("r_time = −0.001 per timestep (encourages efficiency; does NOT guide tool-use strategy)."))
    elems.append(B("No intermediate reward shaping for picking up tools, approaching objects, or making contact."))
    elems.append(B("Episode terminates on success, on robot fall (pelvis height < 0.5 m), or at 2,000 steps."))
    elems.append(B("For the Composite task: episode terminates at 3,000 steps."))

    # 3.5
    elems.append(P("3.5  Phase 1 Deliverables & Exit Criteria", "h2"))
    d1 = [
        "All 5 Isaac Lab environments implemented, unit-tested, and committed to the repository.",
        "G1 URDF validated: all 43 DOF correct, sensors confirmed, PD controller stable.",
        "Training infrastructure tested: SLURM job script runs 4,096 parallel envs on ≥4 A100 GPUs.",
        "wandb project live: at least one dummy-agent training run logged with all required metrics.",
        "Random-agent baseline success rates measured and documented (<0.1% for hard tasks).",
        "Phase 1 review meeting: demo 5 Isaac Lab environments in real-time viewer to advisor.",
    ]
    for d in d1: elems.append(B(d))
    elems.append(P("<b>Exit criterion:</b> Phase 2 begins only when all 5 environments pass unit tests "
                   "AND the cluster can sustain ≥4,096 parallel Isaac Lab environments for ≥1 hour without crash.", "note"))
    elems.append(PageBreak())
    return elems


def build_phase2():
    elems = []
    elems.append(P("4.  Phase 2 — Curiosity-Driven RL Training  (Weeks 4–9)", "h2"))
    elems.append(HR())
    elems.append(P("Goal: Train curiosity-driven policies across the full task suite. Achieve at least one "
                   "clear instance of emergent tool-use behaviour (≥20% success rate on Distant Target task "
                   "with RND/DRND). Complete all ablation runs and hyperparameter sweeps.", "body"))
    elems.append(SP(4))

    # 4.1
    elems.append(P("4.1  Algorithm Implementation — Curiosity Module  (Week 4)", "h2"))
    elems.append(P("We implement three curiosity mechanisms to enable the ablation study. "
                   "All are implemented as modular PyTorch nn.Module classes that plug into the training loop.", "body"))

    elems.append(P("<b>4.1.1  RND — Random Network Distillation (Primary, Burda et al. 2019)</b>", "h3"))
    rnd_steps = [
        "Implement fixed random target network f_target: obs → ℝ^d (d=512 for image obs, d=64 for state-vector obs).",
        "Implement trainable predictor network f_pred: obs → ℝ^d (same architecture as target).",
        "Intrinsic reward: r_int = ||f_pred(o) − f_target(o)||² per timestep.",
        "Predictor network is trained with MSE loss; target network is frozen.",
        "Apply running mean/variance normalisation to r_int to prevent reward scale explosion.",
        "Implement separate value head for intrinsic returns (do NOT mix intrinsic and extrinsic returns in a single head — use two separate critics).",
        "Unit test: confirm r_int decreases monotonically as agent revisits the same state.",
    ]
    for s in rnd_steps: elems.append(B(s))

    elems.append(P("<b>4.1.2  DRND — Distributional RND (Yang et al., ICML 2024)</b>", "h3"))
    drnd_steps = [
        "Implement an ensemble of N=10 random target networks (each with different random initialisations).",
        "Compute intrinsic bonus as: r_int = Var(f_target_i(o)) — high variance = novel state.",
        "This 'bonus inconsistency' fix addresses vanilla RND's over-rewarding of already-explored states.",
        "Implement same two-head critic structure as RND.",
        "Add DRND as a command-line flag: --curiosity=drnd.",
        "Baseline comparison: confirm DRND outperforms RND on Distant Target task within 100M steps.",
    ]
    for s in drnd_steps: elems.append(B(s))

    elems.append(P("<b>4.1.3  RDD — Random Distribution Distillation (Fang et al. 2025, arXiv:2505.11044)</b>", "h3"))
    rdd_steps = [
        "The target network outputs a <i>distribution</i> N(μ_θ(s), σ²) rather than a fixed vector.",
        "Sample f_tar(s) ~ N(μ_θ(s), σ²·I) at each step.",
        "Predictor minimises MSE to the sampled target output.",
        "Intrinsic bonus: b(s) = (1/d) ||f_θ(s) − μ_θ(s)||².",
        "This unifies count-based and prediction-error exploration: the pseudo-count term (σ²/n) ensures "
        "proper exploration decay; the discrepancy term captures predictor convergence.",
        "RDD is theoretically stronger than both RND and DRND and should be the primary comparator against vanilla RND.",
        "Set σ=1.0 (fixed), output dim d=512 (image) or d=64 (state-vector) per the RDD ablation results.",
        "Add as --curiosity=rdd flag.",
    ]
    for s in rdd_steps: elems.append(B(s))

    elems.append(P("<b>4.1.4  ICM — Intrinsic Curiosity Module (Pathak et al. 2017) — Ablation Baseline</b>", "h3"))
    icm_steps = [
        "Implement a forward model: predict next observation embedding from current observation + action.",
        "Intrinsic reward = forward model prediction error.",
        "Also implement the inverse model (action prediction) as regularisation.",
        "Add as --curiosity=icm flag.",
        "ICM is included for the ablation study (RQ1) — not expected to be the best performer.",
    ]
    for s in icm_steps: elems.append(B(s))

    # 4.2
    elems.append(P("4.2  Asymmetric Actor-Critic Architecture  (Week 4)", "h2"))
    elems.append(P("The key architectural insight: the <b>Actor</b> (deployed policy) sees only what is available "
                   "on the physical robot, while the <b>Critic</b> (training only) receives privileged simulator information.", "body"))

    arch_data = [
        [P("Component", "table_hdr"), P("Inputs", "table_hdr"), P("Architecture", "table_hdr"), P("Output", "table_hdr"), P("Deployed?", "table_hdr")],
        [P("Actor\n(deployed policy)", "table_cell"),
         P("• Proprioception: joint positions (43), velocities (43), torques (43)\n"
           "• IMU: orientation (4-quat), angular velocity (3)\n"
           "• Wrist force-torque: left (6) + right (6)\n"
           "• Depth image: 64×64 egocentric (head camera)\n"
           "Total: ≈165 + 4096 (image flattened)", "table_cell"),
         P("MLP: [512, 256, 128] + CNN encoder for depth image (4 conv layers → 256-dim latent). "
           "Concatenate proprioceptive features + image latent.", "table_cell"),
         P("Target joint positions at 50 Hz (43-dim continuous)", "table_cell"),
         P("YES — runs on Jetson Orin NX", "table_cell")],
        [P("Critic\n(training only)", "table_cell"),
         P("All actor inputs PLUS privileged:\n"
           "• Full object poses (position + quaternion for all objects)\n"
           "• Contact states (which links are in contact)\n"
           "• Goal position (exact 3D coordinate)\n"
           "• Terrain mesh info", "table_cell"),
         P("MLP: [512, 256, 128]. Receives concatenated actor obs + privileged info.", "table_cell"),
         P("Value estimate V(s) for GAE advantage computation", "table_cell"),
         P("NO — discarded after training", "table_cell")],
        [P("RND/DRND/RDD Module", "table_cell"),
         P("Actor observation vector (no privileged info — keeps sim-to-real compatible)", "table_cell"),
         P("Target net: [256, 256] → d-dim. Predictor net: [256, 256] → d-dim.", "table_cell"),
         P("Intrinsic reward scalar r_int per step", "table_cell"),
         P("NO — training only", "table_cell")],
        [P("Intrinsic Critic\n(second value head)", "table_cell"),
         P("Actor observations + intrinsic reward stream", "table_cell"),
         P("Shares backbone with extrinsic critic; separate output head.", "table_cell"),
         P("V_int(s) — separate from V_ext(s)", "table_cell"),
         P("NO", "table_cell")],
    ]
    elems.append(tbl(arch_data, [1.0*inch, 2.0*inch, 1.5*inch, 0.9*inch, 0.7*inch]))

    # 4.3
    elems.append(P("4.3  Observation & Action Space Specification  (Week 4)", "h2"))
    elems.append(P("<b>Action Space:</b> Continuous, 43-dimensional. Each dimension is a target joint position "
                   "(PD-controlled at 50 Hz). Actions are clipped to joint limits. Apply action smoothing "
                   "(exponential moving average with α=0.8) to prevent jerky motion.", "body"))
    elems.append(P("<b>Observation Normalisation:</b> Apply running mean/variance normalisation independently "
                   "to each observation channel. This is critical for stable training with mixed-scale inputs "
                   "(e.g., joint angles in radians vs. force-torque in Newtons).", "body"))
    elems.append(P("<b>Episode Horizon:</b> 2,000 steps (40 seconds at 50 Hz) for tasks 1–4. "
                   "3,000 steps (60 seconds) for the Composite task.", "body"))

    # 4.4
    elems.append(P("4.4  Reward Annealing & Curriculum Staging  (Weeks 4–6)", "h2"))
    elems.append(P("<b>Total reward:</b> r_total = r_extrinsic + α · r_intrinsic", "body"))
    elems.append(P("<b>α annealing schedule:</b> α is linearly annealed from 1.0 → 0.1 over the full "
                   "training run. This shifts the agent from exploration to exploitation as it discovers "
                   "successful strategies. Monitor α on wandb.", "body"))

    elems.append(P("<b>Curriculum Staging (unlock sequence):</b>", "h3"))
    cur_data = [
        [P("Stage", "table_hdr"), P("Task Unlocked", "table_hdr"), P("Unlock Condition", "table_hdr"), P("Approx. Steps", "table_hdr")],
        [P("1", "table_cell"), P("Distant Target (Easy)", "table_cell"), P("Start of training", "table_cell"), P("0", "table_cell")],
        [P("2", "table_cell"), P("Elevated Button (Medium)", "table_cell"), P(">20% success on Distant Target", "table_cell"), P("~50–100M", "table_cell")],
        [P("3", "table_cell"), P("Occluded Retrieval (Medium)", "table_cell"), P(">20% success on Elevated Button", "table_cell"), P("~100–200M", "table_cell")],
        [P("4", "table_cell"), P("Weight Lever (Hard)", "table_cell"), P(">20% success on Occluded Retrieval", "table_cell"), P("~200–350M", "table_cell")],
        [P("5", "table_cell"), P("Composite (Hard)", "table_cell"), P(">20% success on Elevated Button", "table_cell"), P("~200–500M", "table_cell")],
    ]
    elems.append(tbl(cur_data, [0.5*inch, 1.5*inch, 2.2*inch, 1.0*inch]))
    elems.append(P("Note: Stages 4 and 5 may run in parallel once Stage 3 unlocks. "
                   "All tasks share the same policy; curriculum adjusts the distribution of environment "
                   "instances allocated to each task.", "note"))

    # 4.5
    elems.append(P("4.5  Training Infrastructure & Cluster Setup  (Week 4–5)", "h2"))
    infra = [
        "Write SLURM job script: 4 A100 nodes, 4 GPUs each, 48-hour wall time, auto-requeue on timeout.",
        "Configure Isaac Lab for 4,096 parallel environments distributed across GPUs (1,024 per GPU).",
        "Implement checkpoint saving every 10M steps and on every new peak success rate.",
        "Implement automatic experiment resumption from checkpoint if job is killed (SLURM preemption).",
        "Set episode horizon to 2,000 steps; batch size = 4,096 envs × 16 steps = 65,536 transitions per update.",
        "PPO update: 10 mini-epochs per rollout, mini-batch size 8,192.",
        "Estimated training throughput: ~10M steps/hour on 4×A100; full 500M step run ≈50 hours.",
        "Run a 10M-step smoke test on the cluster before launching full runs.",
        "Monitor GPU utilisation, memory, and step/sec in real-time via wandb system metrics.",
    ]
    for s in infra: elems.append(B(s))

    # 4.6
    elems.append(P("4.6  Hyperparameter Sweeps & Ablation Runs  (Weeks 5–8)", "h2"))
    hp_data = [
        [P("Hyperparameter", "table_hdr"), P("Search Range", "table_hdr"), P("Default", "table_hdr"), P("Search Method", "table_hdr")],
        [P("Learning rate (actor)", "table_cell"), P("[1e-4, 3e-4, 1e-3]", "table_cell"), P("3e-4", "table_cell"), P("Grid", "table_cell")],
        [P("Learning rate (critic)", "table_cell"), P("[3e-4, 1e-3]", "table_cell"), P("1e-3", "table_cell"), P("Grid", "table_cell")],
        [P("GAE λ", "table_cell"), P("[0.90, 0.95, 0.98]", "table_cell"), P("0.95", "table_cell"), P("Grid", "table_cell")],
        [P("PPO clip ε", "table_cell"), P("[0.1, 0.2, 0.3]", "table_cell"), P("0.2", "table_cell"), P("Grid", "table_cell")],
        [P("Discount γ_ext", "table_cell"), P("[0.99, 0.999]", "table_cell"), P("0.999", "table_cell"), P("Grid", "table_cell")],
        [P("Discount γ_int", "table_cell"), P("[0.99, 0.999]", "table_cell"), P("0.99", "table_cell"), P("Grid", "table_cell")],
        [P("α initial (curiosity weight)", "table_cell"), P("[0.5, 1.0, 2.0]", "table_cell"), P("1.0", "table_cell"), P("Grid", "table_cell")],
        [P("α final (after annealing)", "table_cell"), P("[0.05, 0.1, 0.2]", "table_cell"), P("0.1", "table_cell"), P("Grid", "table_cell")],
        [P("RND/DRND/RDD embedding dim d", "table_cell"), P("[64, 256, 512]", "table_cell"), P("512", "table_cell"), P("Grid", "table_cell")],
        [P("DRND ensemble size N", "table_cell"), P("[5, 10, 20]", "table_cell"), P("10", "table_cell"), P("Grid", "table_cell")],
        [P("Action smoothing α", "table_cell"), P("[0.6, 0.8, 0.9]", "table_cell"), P("0.8", "table_cell"), P("Grid", "table_cell")],
    ]
    elems.append(tbl(hp_data, [2.0*inch, 1.5*inch, 0.8*inch, 1.2*inch]))
    elems.append(P("Run sweeps using wandb Sweeps (Bayesian optimisation after initial grid). "
                   "Run each configuration for 100M steps with 3 seeds. "
                   "Select best configuration for full 500M-step training runs.", "note"))

    elems.append(P("<b>Ablation study design (4 ablations, each run for 200M steps, 5 seeds):</b>", "h3"))
    abl_data = [
        [P("Ablation", "table_hdr"), P("Variant A", "table_hdr"), P("Variant B", "table_hdr"), P("Research Question", "table_hdr")],
        [P("A1: Curiosity mechanism", "table_cell"), P("RND vs. DRND vs. RDD vs. ICM", "table_cell"), P("No curiosity (PPO only)", "table_cell"), P("Which curiosity method works best on high-DOF humanoid?", "table_cell")],
        [P("A2: Critic design", "table_cell"), P("Privileged asymmetric critic", "table_cell"), P("Symmetric actor-critic (no privilege)", "table_cell"), P("Does privileged critic accelerate learning?", "table_cell")],
        [P("A3: Visual input", "table_cell"), P("Ego depth image + proprioception", "table_cell"), P("Proprioception only (no depth)", "table_cell"), P("Does visual input accelerate tool discovery?", "table_cell")],
        [P("A4: Curriculum", "table_cell"), P("Staged curriculum (easy→hard)", "table_cell"), P("All tasks simultaneously", "table_cell"), P("Does curriculum staging improve emergence rate?", "table_cell")],
    ]
    elems.append(tbl(abl_data, [1.2*inch, 1.5*inch, 1.5*inch, 2.0*inch]))

    # 4.7
    elems.append(P("4.7  Emergence Analysis & Logging  (Weeks 6–9)", "h2"))
    emerge = [
        "Define 'tool-use emergence event': first episode where the agent (a) makes contact with a tool object AND "
        "(b) successfully completes the task within the same episode. Log the training step at which this first occurs.",
        "Log tool_contact_count (number of timesteps per episode where a tool is grasped) over training.",
        "Log tool_use_rate = fraction of successful episodes that involved tool contact.",
        "Every 50M steps, run 100 evaluation episodes (no exploration noise) and log success_rate per task.",
        "At end of training, run 10 independent seeds and compute Tool Use Emergence Rate "
        "(% of seeds that discovered tool use).",
        "Save video rollouts every 50M steps using Isaac Lab's built-in recorder.",
        "Implement trajectory clustering: use k-means (k=5) on 50-dim trajectory embeddings to identify "
        "distinct tool-use strategies per task. Target: ≥2 distinct strategies per task.",
        "Run HumanoidBench intrinsic-motivation baselines for comparison if possible "
        "(confirms our method against the existing benchmark).",
    ]
    for s in emerge: elems.append(B(s))

    # 4.8
    elems.append(P("4.8  Phase 2 Deliverables & Exit Criteria", "h2"))
    d2 = [
        "RND, DRND, RDD, and ICM modules fully implemented, tested, and committed.",
        "Full 500M-step training run completed on ≥1 task with ≥1 curiosity method.",
        "Tool use emergence observed in at least one task (≥20% success rate on Distant Target as minimum bar).",
        "All 4 ablation variants run for 200M steps with 5 seeds each.",
        "Hyperparameter sweep completed; best config identified.",
        "wandb dashboard showing all training curves, emergence events, and ablation comparisons.",
        "Video recordings of at least 3 clear emergent tool-use episodes saved.",
        "Phase 2 review meeting: present learning curves and best-performing episodes to advisor.",
    ]
    for d in d2: elems.append(B(d))
    elems.append(P("<b>Exit criterion:</b> Phase 3 begins when ≥1 task has achieved >20% success rate "
                   "with a curiosity-driven policy AND ≥3 clean tool-use rollouts are recorded.", "note"))
    elems.append(PageBreak())
    return elems


def build_phase3():
    elems = []
    elems.append(P("5.  Phase 3 — Policy Distillation & Robustification  (Weeks 10–11)", "h1"))
    elems.append(HR())
    elems.append(P("Goal: Convert the noisy curiosity-driven policy into a clean, deployable policy via "
                   "DAgger-based behaviour cloning, then refine it with residual RL under hardware-realistic "
                   "actuator constraints. Cross-validate in MuJoCo to confirm behaviours are not "
                   "simulator-specific artifacts.", "body"))
    elems.append(SP(4))

    # 5.1
    elems.append(P("5.1  Phase A — Behaviour Cloning via DAgger  (Week 10, Days 1–4)", "h2"))
    elems.append(P("The curiosity-driven policy uses exploration noise (RND overhead, random perturbations) "
                   "that make it unsuitable for hardware deployment. DAgger distils its <i>behaviour</i> into "
                   "a clean student policy without exploration overhead.", "body"))
    dagger_steps = [
        "Collect demonstration dataset D from the best-performing curiosity policy: run 10,000 episodes "
        "with the trained actor (deterministic, no exploration noise); save (observation, action) pairs.",
        "Filter dataset: keep only episodes with tool-use success (final r_extrinsic = +1) to ensure "
        "the student learns successful strategies, not failed explorations.",
        "Train a student policy (same architecture as actor, NO RND module) via behaviour cloning: "
        "minimise ||π_student(o) − π_teacher(o)||² using Adam, lr=1e-4, for 100 epochs.",
        "After each epoch, collect new rollouts with the student policy and add to dataset D "
        "(the DAgger loop: student-generated states prevent distribution shift).",
        "Run DAgger for 20 iterations (each iteration: 500 episodes rollout + 50 epochs BC training).",
        "Evaluate student policy every 5 DAgger iterations: confirm success rate tracks teacher within 10%.",
        "Confirm the student policy has NO RND module and runs inference in <10 ms (required for 50 Hz control).",
        "Save the distilled student policy checkpoint as student_policy_v1.pt.",
    ]
    for s in dagger_steps: elems.append(B(s))

    # 5.2
    elems.append(P("5.2  Phase B — Residual RL Refinement  (Week 10, Days 5 – Week 11, Day 3)", "h2"))
    elems.append(P("The distilled base policy may have sim-to-real gaps due to idealised actuator models. "
                   "Residual RL fine-tunes a small MLP on top of the frozen base policy under realistic "
                   "actuator constraints.", "body"))
    residual_steps = [
        "<b>Freeze base policy:</b> student_policy_v1.pt weights are frozen — no gradient flows through them.",
        "<b>Add residual MLP:</b> a lightweight 2-layer MLP [128, 64] takes the base policy's action + "
        "current observation and outputs a residual correction δa. Final action = π_base(o) + δa.",
        "<b>Actuator constraints applied during residual training:</b>",
        [
            "Torque-speed curves: joint torque limits from the G1 hardware spec (varies per joint).",
            "Overcurrent protection: clip torque if sustained for >0.5 s above thermal limit.",
            "Joint position/velocity limits: hard clipped to G1 hardware limits.",
            "Communication latency: simulate 0–20 ms random action delay.",
            "Motor response curves: apply first-order lag filter (τ=0.02 s) to action execution.",
        ],
        "<b>Domain randomisation for residual training:</b>",
        [
            "Mass: ±20% of nominal for all links.",
            "Friction (ground + objects): ±40% of nominal.",
            "Joint damping: ±30% of nominal.",
            "Observation noise: Gaussian noise σ=0.01 on all proprioceptive channels.",
            "Gravity direction: ±3° tilt from vertical.",
        ],
        "Train residual MLP with PPO for 50M steps (much shorter than curiosity training — base provides strong prior).",
        "Evaluate: success rate of (base + residual) vs. base alone; confirm improvement ≥10%.",
        "Save final checkpoint: deployable_policy_v1.pt = frozen base + trained residual.",
    ]
    for s in residual_steps:
        if isinstance(s, list):
            for sub in s: elems.append(B(sub, indent=1))
        else:
            elems.append(B(s))

    # 5.3
    elems.append(P("5.3  Sim-to-Sim Cross-Validation in MuJoCo  (Week 11, Days 4–5)", "h2"))
    elems.append(P("Deploy the trained policy in MuJoCo 3.x (independent of Isaac Lab) to verify that "
                   "behaviours are not simulator-specific artifacts. If the policy fails completely in MuJoCo, "
                   "it signals a physics modelling gap that must be resolved before hardware deployment.", "body"))
    mujoco_steps = [
        "Convert G1 URDF to MuJoCo XML format using the mujoco-compile tool.",
        "Implement the same 5 task environments in MuJoCo (simplified versions acceptable — "
        "same object geometry, same reward structure).",
        "Load deployable_policy_v1.pt and run 100 evaluation episodes in MuJoCo.",
        "Target: ≥50% of the Isaac Lab success rate achieved in MuJoCo.",
        "If success rate drops >50% in MuJoCo, investigate the specific physics discrepancy "
        "(contact model, friction, actuator model) and apply targeted domain randomisation fixes.",
        "Document MuJoCo success rate in the paper as evidence against simulator overfitting.",
    ]
    for s in mujoco_steps: elems.append(B(s))

    # 5.4
    elems.append(P("5.4  Phase 3 Deliverables & Exit Criteria", "h2"))
    d3 = [
        "DAgger distillation complete: clean student policy with no RND overhead, inference <10 ms.",
        "Residual RL training complete: deployable_policy_v1.pt saved.",
        "MuJoCo cross-validation complete: ≥50% of Isaac Lab success rate achieved.",
        "Policy inference latency confirmed <10 ms on a local GPU (TensorRT export is Phase 4).",
        "Phase 3 review: demonstrate clean policy rollout (no exploration jitter) in Isaac Lab viewer.",
    ]
    for d in d3: elems.append(B(d))
    elems.append(P("<b>Exit criterion:</b> Phase 4 begins when deployable_policy_v1.pt achieves ≥50% success "
                   "rate in MuJoCo AND inference is confirmed <10 ms on CPU-equivalent hardware.", "note"))
    elems.append(PageBreak())
    return elems


def build_phase4():
    elems = []
    elems.append(P("6.  Phase 4 — Sim-to-Real Transfer  (Weeks 12–14)", "h1"))
    elems.append(HR())
    elems.append(P("Goal: Deploy the trained policy on the physical Unitree G1 EDU robot. Conduct a "
                   "systematic real-world evaluation protocol. Document successes and failures. "
                   "Achieve ≥30% success rate on the physical robot on ≥1 task.", "body"))
    elems.append(SP(4))

    # 6.1
    elems.append(P("6.1  Domain Randomisation Ranges for Sim-to-Real  (Already Applied in Phase 3)", "h2"))
    dr_data = [
        [P("Parameter", "table_hdr"), P("Sim Range", "table_hdr"), P("Purpose", "table_hdr")],
        [P("Link masses", "table_cell"), P("±20% of CAD spec", "table_cell"), P("Compensate for payload variation", "table_cell")],
        [P("Ground friction (μ)", "table_cell"), P("±40% (0.3–1.0)", "table_cell"), P("Different floor surfaces", "table_cell")],
        [P("Joint damping", "table_cell"), P("±30% of nominal", "table_cell"), P("Actuator wear variation", "table_cell")],
        [P("Observation noise (σ)", "table_cell"), P("Gaussian, σ=0.01 all channels", "table_cell"), P("Encoder/IMU noise", "table_cell")],
        [P("Action delay", "table_cell"), P("0–20 ms random", "table_cell"), P("Communication latency", "table_cell")],
        [P("Gravity direction", "table_cell"), P("±3° from vertical", "table_cell"), P("Uneven floor, calibration error", "table_cell")],
        [P("Object mass (real tools)", "table_cell"), P("±30% of labelled mass", "table_cell"), P("Manufacturing tolerance", "table_cell")],
        [P("Object friction (real tools)", "table_cell"), P("±50% of nominal", "table_cell"), P("Surface variation", "table_cell")],
    ]
    elems.append(tbl(dr_data, [1.5*inch, 1.5*inch, 3.2*inch]))

    # 6.2
    elems.append(P("6.2  Safety Protocol  (MUST be followed before ANY robot power-on)", "h2"))
    elems.append(P("⚠️  All real-world tests carry risk of hardware damage and physical injury. "
                   "The following protocol is MANDATORY and must be reviewed by the faculty advisor "
                   "before any hardware session.", "note"))
    safety = [
        "<b>Pre-session checklist (complete before every session):</b>",
        ["Inspect all robot joints for mechanical wear or unusual resistance.",
         "Check Dex3-1 finger integrity — no bent linkages or loose screws.",
         "Verify the safety gantry is load-rated and tether is attached to the robot's harness point.",
         "Confirm the emergency stop (E-stop) remote is charged and within reach of all team members.",
         "Confirm the lab is clear of non-essential personnel.",
         "Power on robot in damping mode (joints limp) first; confirm no error LEDs.",
         "Run factory self-test via unitree_sdk2_python; confirm all actuator IDs respond."],
        "<b>Behavioural kill-switch implementation:</b>",
        ["Implement a software kill-switch that monitors joint torque in real-time (100 Hz).",
         "If any joint exceeds 90% of its maximum rated torque for >100 ms, immediately send zero-torque command.",
         "If centre-of-mass deviation exceeds ±0.3 m from the tether attachment point, send zero-torque command.",
         "If any foot contact force exceeds 3× body weight, send zero-torque command.",
         "The E-stop remote cuts power to all motors immediately (hardware-level, not software)."],
        "<b>Graduated deployment progression:</b>",
        ["Stage 1 (Week 12): Static tool interaction — robot is standing still, tethered. "
         "Test arm reaching and grasping motions only. No locomotion.",
         "Stage 2 (Week 13): Slow-motion full task — run policy at 25% speed (override control loop to "
         "command 0.25× the policy's action magnitudes). Observe for instabilities.",
         "Stage 3 (Week 14): Full speed — run policy at 100% if Stage 2 shows no instabilities "
         "across 20 consecutive episodes.",
         "At no stage should untethered bipedal locomotion be attempted unless the faculty advisor approves."],
    ]
    for s in safety:
        if isinstance(s, list):
            for sub in s: elems.append(B(sub, indent=1))
        else:
            elems.append(B(s))

    # 6.3
    elems.append(P("6.3  Hardware Deployment Procedure  (Weeks 12–14)", "h2"))
    deploy_steps = [
        "<b>TensorRT export (do this before Week 12):</b>",
        ["Export deployable_policy_v1.pt to TensorRT engine for the Jetson Orin NX.",
         "Target inference latency: ≤10 ms (50 Hz control requires action computation in <20 ms total).",
         "Validate TensorRT output matches PyTorch output within 1e-4 absolute error on 1000 test inputs.",
         "Flash the TensorRT engine to the Jetson Orin NX onboard the G1."],
        "<b>Communication setup:</b>",
        ["Connect development laptop to G1 via Ethernet (1 Gbps).",
         "Run unitree_sdk2_python control loop: subscribe to joint state topic, publish joint position commands.",
         "Confirm round-trip latency <5 ms on the Ethernet link.",
         "Implement ROS 2 Humble bridge node: /joint_states → policy → /joint_commands.",
         "Implement real-time logging: record all observations and actions to a local bag file "
         "for post-hoc analysis."],
        "<b>Observation computation on Jetson:</b>",
        ["Proprioceptive observations computed directly from unitree_sdk2_python joint state.",
         "IMU data: orientation from built-in IMU (quaternion), angular velocity from built-in gyroscope.",
         "Wrist force-torque: read from Dex3-1 wrist sensors via SDK.",
         "Depth image: capture from head camera at 50 Hz; resize to 64×64; apply same normalisation as training.",
         "Concatenate all observations and feed to TensorRT engine."],
        "<b>Task scene setup (physical lab):</b>",
        ["Arrange the physical task scene matching the simulation layout (measured to ±2 cm).",
         "Label all tool objects with their mass and confirm it matches the simulation ±30% range.",
         "Measure friction coefficients of physical objects with a tilt-test; confirm within ±50% range.",
         "Place safety padding around all task objects to prevent damage on robot fall.",
         "Set up an external overhead camera for video documentation (not used by the policy)."],
    ]
    for s in deploy_steps:
        if isinstance(s, list):
            for sub in s: elems.append(B(sub, indent=1))
        else:
            elems.append(B(s))

    # 6.4
    elems.append(P("6.4  Real-World Evaluation Protocol  (Weeks 13–14)", "h2"))
    elems.append(P("Conduct a systematic evaluation protocol matching the simulation evaluation "
                   "to enable direct comparison.", "body"))
    eval_steps = [
        "Run 50 episodes per task on the physical G1 (prioritise Distant Target and Elevated Button tasks).",
        "For each episode: reset the task scene to a randomised configuration (within ±30% position, ±50% friction).",
        "Record: success (binary), episode length, tool contact events (from force-torque sensor), "
        "any safety-kill activations.",
        "Compute real-world success rate and compare to simulation success rate.",
        "Record high-quality video of every successful trial for the paper's supplementary video.",
        "If success rate <10% on the physical robot after 50 episodes: invoke risk mitigation protocol "
        "(see Section 10).",
        "Conduct qualitative analysis: compare the robot's tool-use strategy in sim vs. real — "
        "does the same strategy (e.g., approaching from the same angle) emerge?",
    ]
    for s in eval_steps: elems.append(B(s))

    # 6.5
    elems.append(P("6.5  Phase 4 Deliverables & Exit Criteria", "h2"))
    d4 = [
        "TensorRT policy running on Jetson Orin NX at ≤10 ms latency.",
        "50 real-world evaluation episodes completed per task (at minimum Distant Target + Elevated Button).",
        "Video documentation: ≥5 successful real-world tool-use episodes recorded in high quality.",
        "Real-world success rate measured and documented: target ≥30% on ≥1 task.",
        "Safety log: record all E-stop activations and kill-switch triggers.",
        "Phase 4 review: live demo or video demo of real-robot tool use to advisor.",
    ]
    for d in d4: elems.append(B(d))
    elems.append(P("<b>Exit criterion:</b> Phase 5 begins when ≥1 task shows ≥10% real-world success rate "
                   "(relaxed from 30% target to allow paper writing to begin in parallel).", "note"))
    elems.append(PageBreak())
    return elems


def build_phase5():
    elems = []
    elems.append(P("7.  Phase 5 — Paper Writing & Code Release  (Weeks 15–16)", "h1"))
    elems.append(HR())

    paper_tasks = [
        ("<b>Week 15 — Draft:</b>",
         ["Write Abstract (confirm final claims match achieved results).",
          "Write Introduction (motivation, gap statement, contributions — use HumanoidBench as headline evidence).",
          "Write Related Work (5 subsections: curiosity RL methods; emergent tool use; humanoid whole-body RL; "
          "sim-to-real; benchmarks — cite all 26 papers from the literature review).",
          "Write Method section (4.1 env design, 4.2 curiosity module, 4.3 architecture, 4.4 training details).",
          "Generate all figures: pipeline diagram, task suite figure, learning curves, ablation plots, "
          "trajectory clustering visualisations, real-robot photo/video stills.",
          "Write Experiments section: environment descriptions, training setup, all quantitative results tables.",
          "Internal draft review: all co-authors read and comment."]),
        ("<b>Week 16 — Finalise & Release:</b>",
         ["Revise draft based on internal review.",
          "Write Conclusion and Limitations sections.",
          "Prepare supplementary: full hyperparameter table, additional ablation plots, extended video.",
          "Produce supplementary video: 3–5 minutes showing simulation training, emerged behaviours, "
          "and real-robot deployment.",
          "Code release preparation: clean up codebase, write README with installation and training instructions, "
          "add MIT or Apache 2.0 license, upload to GitHub.",
          "Submit to target venue (ICRA / IROS / CoRL — select based on timeline).",
          "Archive checkpoints and datasets on Purdue shared storage for reproducibility."]),
    ]
    for title, tasks in paper_tasks:
        elems.append(P(title, "h3"))
        for t in tasks: elems.append(B(t))

    elems.append(P("<b>Target Venues (in order of preference):</b>", "h3"))
    venues = [
        "CoRL 2026 (Conference on Robot Learning) — submission deadline typically July 2026. "
        "Best fit: manipulation + RL + sim-to-real.",
        "IROS 2026 (IEEE/RSJ Intelligent Robots and Systems) — submission typically March 2026. "
        "High impact for hardware demonstration papers.",
        "ICRA 2027 — submission typically September 2026. Largest robotics venue.",
        "NeurIPS 2026 (Robotics workshop) — for initial rapid dissemination if full paper not ready.",
    ]
    for v in venues: elems.append(B(v))
    elems.append(PageBreak())
    return elems


def build_metrics():
    elems = []
    elems.append(P("8.  Evaluation Plan & Success Metrics", "h1"))
    elems.append(HR())

    m_data = [
        [P("Metric", "table_hdr"), P("Definition", "table_hdr"),
         P("Target", "table_hdr"), P("Measurement Method", "table_hdr"), P("Phase", "table_hdr")],
        [P("Tool Use Emergence Rate", "table_cell"),
         P("% of seeds (out of 10) that independently discover a tool-use solution on ≥1 task", "table_cell"),
         P("≥60%", "metric_val"), P("10 independent seeds; each run 500M steps; classify as 'emerged' if tool_use_rate>5%", "table_cell"),
         P("Phase 2", "table_cell")],
        [P("Task Success Rate (sim)", "table_cell"),
         P("% of episodes completed successfully after policy convergence (200 eval episodes)", "table_cell"),
         P("≥50% (sim)", "metric_val"), P("Run 200 eval episodes with no exploration noise; count successes", "table_cell"),
         P("Phase 2", "table_cell")],
        [P("Exploration Efficiency", "table_cell"),
         P("Number of training steps to first tool-use discovery event", "table_cell"),
         P("<500M steps", "metric_val"), P("Log first episode with tool_contact AND success; record training step", "table_cell"),
         P("Phase 2", "table_cell")],
        [P("Behavioural Diversity", "table_cell"),
         P("Number of distinct tool-use strategies per task (k-means on trajectory embeddings)", "table_cell"),
         P("≥2 per task", "metric_val"), P("k-means (k=5) on 50-dim trajectory embeddings from 1000 successful episodes", "table_cell"),
         P("Phase 2", "table_cell")],
        [P("Sim-to-Real Transfer Rate", "table_cell"),
         P("Success rate of tool-use behaviours on physical G1 hardware", "table_cell"),
         P("≥30%", "metric_val"), P("50 real-world evaluation episodes per task", "table_cell"),
         P("Phase 4", "table_cell")],
        [P("Ablation Delta", "table_cell"),
         P("Success rate improvement of best curiosity method vs. no-curiosity PPO baseline", "table_cell"),
         P("≥3× improvement", "metric_val"), P("200M step runs, 5 seeds each; compare best curiosity vs. PPO-only", "table_cell"),
         P("Phase 2", "table_cell")],
        [P("Inference Latency (hardware)", "table_cell"),
         P("Policy inference time on Jetson Orin NX (TensorRT)", "table_cell"),
         P("≤10 ms", "metric_val"), P("Time 1000 forward passes on Jetson Orin NX; report mean ± std", "table_cell"),
         P("Phase 4", "table_cell")],
        [P("MuJoCo Cross-Validation", "table_cell"),
         P("Success rate in MuJoCo as % of Isaac Lab success rate", "table_cell"),
         P("≥50%", "metric_val"), P("100 eval episodes in MuJoCo; compare to Isaac Lab baseline", "table_cell"),
         P("Phase 3", "table_cell")],
    ]
    elems.append(tbl(m_data, [1.3*inch, 1.8*inch, 0.6*inch, 1.8*inch, 0.7*inch]))
    elems.append(PageBreak())
    return elems


def build_ablations():
    elems = []
    elems.append(P("9.  Ablation Study Design", "h1"))
    elems.append(HR())
    elems.append(P("All ablations run for 200M steps with 5 independent seeds. Results reported as "
                   "mean ± std of task success rate at convergence.", "body"))
    abl_full = [
        [P("ID", "table_hdr"), P("Name", "table_hdr"), P("What is varied", "table_hdr"),
         P("Variants", "table_hdr"), P("Expected finding", "table_hdr"), P("Seeds", "table_hdr")],
        [P("A1", "table_cell"), P("Curiosity mechanism", "table_cell"),
         P("Which intrinsic reward module is used", "table_cell"),
         P("RND / DRND / RDD / ICM / None (PPO only)", "table_cell"),
         P("DRND or RDD > RND > ICM >> None; confirms intrinsic motivation is necessary", "table_cell"),
         P("5 per variant", "table_cell")],
        [P("A2", "table_cell"), P("Critic design", "table_cell"),
         P("Whether the critic receives privileged information", "table_cell"),
         P("Asymmetric (privileged) / Symmetric (no privilege)", "table_cell"),
         P("Asymmetric critic accelerates learning by 30–50% in wall steps", "table_cell"),
         P("5 per variant", "table_cell")],
        [P("A3", "table_cell"), P("Visual input", "table_cell"),
         P("Whether the actor receives the 64×64 egocentric depth image", "table_cell"),
         P("Depth image + proprioception / Proprioception only", "table_cell"),
         P("Depth image helps on tasks requiring spatial reasoning (Occluded Retrieval)", "table_cell"),
         P("5 per variant", "table_cell")],
        [P("A4", "table_cell"), P("Curriculum staging", "table_cell"),
         P("Whether tasks are introduced in difficulty order", "table_cell"),
         P("Staged curriculum / All tasks simultaneously", "table_cell"),
         P("Curriculum improves emergence rate on hard tasks; all-simultaneous leads to catastrophic forgetting", "table_cell"),
         P("5 per variant", "table_cell")],
        [P("A5", "table_cell"), P("α annealing", "table_cell"),
         P("Whether curiosity weight is annealed", "table_cell"),
         P("Annealed 1.0→0.1 / Fixed α=1.0 / Fixed α=0.1", "table_cell"),
         P("Annealing achieves best of both: exploration early, exploitation later", "table_cell"),
         P("3 per variant", "table_cell")],
        [P("A6", "table_cell"), P("QFLEX comparison", "table_cell"),
         P("Value-guided flow exploration (Wei et al. 2026) vs. RND-based", "table_cell"),
         P("RDD / QFLEX-adapted for continuous humanoid", "table_cell"),
         P("QFLEX may outperform on high-DOF exploration; important baseline for paper", "table_cell"),
         P("5 per variant", "table_cell")],
    ]
    elems.append(tbl(abl_full, [0.3*inch, 1.0*inch, 1.2*inch, 1.3*inch, 1.8*inch, 0.6*inch]))
    elems.append(PageBreak())
    return elems


def build_risks():
    elems = []
    elems.append(P("10.  Risk Register & Mitigation Strategies", "h1"))
    elems.append(HR())

    risk_data = [
        [P("Risk", "table_hdr"), P("Likelihood", "table_hdr"), P("Impact", "table_hdr"),
         P("Mitigation Strategy", "table_hdr"), P("Fallback", "table_hdr")],

        [P("Tool use does not emerge within compute budget (500M steps)", "table_cell"),
         P("Medium", "risk_med"), P("High", "risk_high"),
         P("Start with easiest task (Distant Target). Use curriculum learning. "
           "Monitor every 50M steps; if no contact events by 100M steps, switch to hybrid "
           "approach (light reward shaping + curiosity).", "table_cell"),
         P("Hybrid: add small contact bonus (+0.01) without specifying which object. "
           "Characterise as 'humanoid hard-exploration benchmark' — still publishable.", "table_cell")],

        [P("Sim-to-real gap causes behaviour collapse on physical G1", "table_cell"),
         P("Medium", "risk_med"), P("Medium", "risk_med"),
         P("Extensive domain randomisation (Section 6.1). Residual RL with hardware constraints. "
           "Sim-to-sim validation in MuJoCo before hardware.", "table_cell"),
         P("Use ASAP-style delta-action model (fine-tune on 10 real-world rollouts). "
           "Even simulation results alone are publishable contribution.", "table_cell")],

        [P("Dex3-1 grasp limitations prevent reliable tool manipulation", "table_cell"),
         P("Low", "risk_low"), P("High", "risk_high"),
         P("Design tool objects with cylindrical/prismatic graspable handles matching Dex3-1 geometry. "
           "Simplify grasping primitives. Use magnetic or velcro-assisted grasps as a fallback.", "table_cell"),
         P("If Dex3-1 is unreliable: use simpler 2-finger gripper approximation in sim; "
           "demonstrate on at least stick-pushing tasks that do not require precision grasp.", "table_cell")],

        [P("Hardware damage during sim-to-real deployment", "table_cell"),
         P("Low", "risk_low"), P("High", "risk_high"),
         P("Safety gantry mandatory for all real-world tests. Torque limits enforced in residual RL. "
           "Graduated deployment: static → slow-motion → full-speed. "
           "E-stop remote always within reach.", "table_cell"),
         P("All real-world experiments halted; paper based on simulation results only. "
           "Simulation contribution is still novel and publishable.", "table_cell")],

        [P("Compute budget exceeded (cluster queue delays / preemption)", "table_cell"),
         P("Medium", "risk_med"), P("Medium", "risk_med"),
         P("Implement SLURM auto-requeue. Run shorter ablations first; full runs in background. "
           "Budget ≥600 GPU-hours to have headroom.", "table_cell"),
         P("Reduce number of seeds from 10 to 5 for emergence rate estimate. "
           "Focus ablations on A1 (curiosity mechanism) and A2 (critic design) as highest-priority.", "table_cell")],

        [P("Competing paper scoops the exact result before submission", "table_cell"),
         P("Medium", "risk_med"), P("High", "risk_high"),
         P("Move quickly. Monitor arXiv weekly (especially Abbeel/Pathak/NVIDIA groups). "
           "Frame contribution around the G1 hardware demo — almost no curiosity papers show real-robot results.", "table_cell"),
         P("If scooped in simulation: emphasise the real-robot demo as unique contribution. "
           "If scooped entirely: shift to negative result / benchmark paper.", "table_cell")],

        [P("Isaac Lab API changes break environment implementation", "table_cell"),
         P("Low", "risk_low"), P("Medium", "risk_med"),
         P("Pin Isaac Lab version in Docker image from Day 1. "
           "Do not update Isaac Lab mid-project unless a critical bug requires it.", "table_cell"),
         P("Roll back to pinned version. Fix environment if API change is unavoidable.", "table_cell")],

        [P("Unitree G1 not available (still to be procured)", "table_cell"),
         P("Low", "risk_low"), P("High", "risk_high"),
         P("Initiate procurement immediately (pre-phase). Design all simulation work to be "
           "hardware-independent. Phases 1–3 are entirely simulation-based.", "table_cell"),
         P("If G1 arrives after Week 12: compress Phase 4 to 1 week of evaluation. "
           "If G1 never arrives: simulation-only paper.", "table_cell")],
    ]
    elems.append(tbl(risk_data, [1.3*inch, 0.7*inch, 0.6*inch, 2.0*inch, 1.6*inch]))
    elems.append(PageBreak())
    return elems


def build_resources():
    elems = []
    elems.append(P("11.  Required Resources Checklist", "h1"))
    elems.append(HR())

    elems.append(P("11.1  Compute Resources", "h2"))
    comp_data = [
        [P("Resource", "table_hdr"), P("Specification", "table_hdr"), P("Quantity", "table_hdr"), P("When Needed", "table_hdr"), P("Source", "table_hdr")],
        [P("GPU cluster nodes", "table_cell"), P("≥4 NVIDIA A100 80GB or H100 80GB", "table_cell"), P("≥4 nodes", "table_cell"), P("Weeks 1–11", "table_cell"), P("Purdue RCAC (Anvil/Gilbreth/Negishi)", "table_cell")],
        [P("Total GPU-hours", "table_cell"), P("~200–500 GPU-hours (full suite + ablations)", "table_cell"), P("1 allocation", "table_cell"), P("Weeks 4–11", "table_cell"), P("RCAC allocation request", "table_cell")],
        [P("Local GPU workstation", "table_cell"), P("≥1 RTX 3090 or A5000 per researcher", "table_cell"), P("2 workstations", "table_cell"), P("All weeks", "table_cell"), P("Lab hardware", "table_cell")],
        [P("Jetson Orin NX (onboard G1)", "table_cell"), P("275 TOPS, part of G1 EDU", "table_cell"), P("1 (built-in)", "table_cell"), P("Weeks 12–14", "table_cell"), P("G1 EDU spec", "table_cell")],
        [P("Storage", "table_cell"), P("≥1 TB for checkpoints + video", "table_cell"), P("1 TB", "table_cell"), P("All weeks", "table_cell"), P("RCAC scratch storage", "table_cell")],
    ]
    elems.append(tbl(comp_data, [1.2*inch, 1.8*inch, 0.7*inch, 0.8*inch, 1.7*inch]))

    elems.append(P("11.2  Hardware Resources", "h2"))
    hw_data2 = [
        [P("Item", "table_hdr"), P("Spec", "table_hdr"), P("Qty", "table_hdr"), P("Status", "table_hdr"), P("Action Required", "table_hdr")],
        [P("Unitree G1 EDU", "table_cell"), P("43 DOF, Dex3-1, Jetson Orin NX", "table_cell"), P("1", "table_cell"), P("[existing / procure]", "table_cell"), P("Confirm now; procure if needed (lead time ≈8 weeks)", "table_cell")],
        [P("Safety gantry", "table_cell"), P("Load-rated ≥100 kg, overhead mount", "table_cell"), P("1", "table_cell"), P("[check lab]", "table_cell"), P("Load-test before Phase 4", "table_cell")],
        [P("Wooden blocks (tool)", "table_cell"), P("30×30×30 cm, 1–3 kg each", "table_cell"), P("5", "table_cell"), P("Procure", "table_cell"), P("Buy or fabricate in workshop; measure mass precisely", "table_cell")],
        [P("PVC rods (tool)", "table_cell"), P("80–120 cm length, <0.5 kg", "table_cell"), P("5", "table_cell"), P("Procure", "table_cell"), P("Buy from hardware store; sand ends smooth", "table_cell")],
        [P("Foam wedges (tool)", "table_cell"), P("Variable height fulcrum blocks", "table_cell"), P("3", "table_cell"), P("Procure", "table_cell"), P("Cut from high-density foam", "table_cell")],
        [P("Ethernet switch", "table_cell"), P("1 Gbps managed switch", "table_cell"), P("1", "table_cell"), P("[check lab]", "table_cell"), P("Confirm <1 ms latency with G1 SDK test", "table_cell")],
        [P("E-stop remote", "table_cell"), P("Wireless, <50 ms latency", "table_cell"), P("2", "table_cell"), P("[check lab]", "table_cell"), P("Test wireless range in deployment lab", "table_cell")],
        [P("Overhead RGB-D camera", "table_cell"), P("1080p@30fps for monitoring only", "table_cell"), P("1", "table_cell"), P("[check lab]", "table_cell"), P("Mount above task area for video documentation", "table_cell")],
    ]
    elems.append(tbl(hw_data2, [1.2*inch, 1.3*inch, 0.4*inch, 0.8*inch, 2.5*inch]))

    elems.append(P("11.3  Software Resources", "h2"))
    sw_data2 = [
        [P("Software", "table_hdr"), P("Version", "table_hdr"), P("Licence", "table_hdr"), P("Purpose", "table_hdr")],
        [P("NVIDIA Isaac Lab", "table_cell"), P("4.x (Isaac Sim 4.x)", "table_cell"), P("Free (academic)", "table_cell"), P("Primary training simulator", "table_cell")],
        [P("unitree_rl_gym", "table_cell"), P("Latest main", "table_cell"), P("BSD-3", "table_cell"), P("G1 URDF + PPO boilerplate", "table_cell")],
        [P("rsl_rl", "table_cell"), P("1.0.x", "table_cell"), P("BSD-3", "table_cell"), P("PPO implementation", "table_cell")],
        [P("MuJoCo 3.x", "table_cell"), P("3.1+", "table_cell"), P("Apache 2.0", "table_cell"), P("Sim-to-sim cross-validation", "table_cell")],
        [P("TensorRT", "table_cell"), P("8.6+ (Jetson-compatible)", "table_cell"), P("NVIDIA proprietary", "table_cell"), P("Onboard inference optimisation", "table_cell")],
        [P("unitree_sdk2_python", "table_cell"), P("Latest", "table_cell"), P("Unitree proprietary", "table_cell"), P("Hardware communication", "table_cell")],
        [P("ROS 2 Humble", "table_cell"), P("Humble Hawksbill", "table_cell"), P("Apache 2.0", "table_cell"), P("Hardware state/command bridge", "table_cell")],
        [P("PyTorch", "table_cell"), P("2.1+", "table_cell"), P("BSD-3", "table_cell"), P("Neural network training", "table_cell")],
        [P("Weights & Biases", "table_cell"), P("Latest", "table_cell"), P("Free (academic)", "table_cell"), P("Experiment tracking", "table_cell")],
        [P("Python 3.10+", "table_cell"), P("3.10 or 3.11", "table_cell"), P("PSF", "table_cell"), P("All scripting", "table_cell")],
    ]
    elems.append(tbl(sw_data2, [1.5*inch, 1.2*inch, 1.1*inch, 2.4*inch]))
    elems.append(PageBreak())
    return elems


def build_literature():
    elems = []
    elems.append(P("12.  Literature Positioning & Novelty Arguments", "h1"))
    elems.append(HR())
    elems.append(P("This section documents exactly how to position this work against the most "
                   "dangerous adjacent papers. Every reviewer will ask about these.", "body"))

    pos_data = [
        [P("Paper", "table_hdr"), P("How it differs from our work", "table_hdr"), P("Our distinction", "table_hdr")],
        [P("HumanoidBench\n(Sferrazza/Abbeel, RSS 2024)", "table_cell"),
         P("Same robot family (H1), reports that RL including intrinsic-motivation baselines FAILS on "
           "hard whole-body manipulation tasks", "table_cell"),
         P("We attack the exact open problem. HumanoidBench is our primary motivating evidence.", "table_cell")],
        [P("ASAP\n(He et al., RSS 2025)", "table_cell"),
         P("Same G1, same Isaac Gym, whole-body, sim-to-real. BUT: motion-tracking objective "
           "(not sparse reward), uses retargeted human motion data (not zero demos), no tools, no novel behaviour.", "table_cell"),
         P("Demonstration-free, emergent, tool-focused. Reuse their delta-action sim-to-real technique.", "table_cell")],
        [P("'Opening the Sim-to-Real Door'\n(Wang et al., Dec 2025)", "table_cell"),
         P("Closest 2025 work on humanoid + RL + manipulation sim-to-real. Uses teacher-student with "
           "privileged info; task-specific reward; not curiosity-driven.", "table_cell"),
         P("No privileged teacher, intrinsic motivation only, free-form emergent tool discovery.", "table_cell")],
        [P("QFLEX\n(Wei et al., Tsinghua 2026)", "table_cell"),
         P("Value-guided flow exploration for high-DOF continuous control. Strongest pure-exploration "
           "method for high-DOF. Does NOT claim emergent tool use, not on humanoid hardware.", "table_cell"),
         P("Must treat QFLEX as a primary baseline. Include QFLEX-adapted as ablation A6.", "table_cell")],
        [P("DRND\n(Yang et al., ICML 2024)", "table_cell"),
         P("Strongest 2024 RND successor for high-DOF manipulation. Validated on Atari + Adroit, "
           "NOT on humanoid whole-body control.", "table_cell"),
         P("DRND is our primary curiosity baseline. We extend it to 43-DOF humanoid loco-manipulation.", "table_cell")],
        [P("DemoHLM\n(Unitree G1, Oct 2025)", "table_cell"),
         P("Same G1 hardware, 10 loco-manipulation tasks, RGB-D. Still requires one demonstration; "
           "not curiosity-driven; not emergent.", "table_cell"),
         P("Zero demonstrations. Intrinsic curiosity only.", "table_cell")],
        [P("GR00T N1.6 / Helix / Gemini 1.5\n(2025–2026)", "table_cell"),
         P("Foundation model VLAs trained on massive teleoperation datasets. Different paradigm entirely.", "table_cell"),
         P("Zero pre-training data. Pure exploration from scratch. Better sample-efficient for novel tools "
           "not seen during pre-training.", "table_cell")],
        [P("Baker et al. / OpenAI\n(ICLR 2020)", "table_cell"),
         P("Canonical emergent tool use reference — hide-and-seek agents. Simplified spherical agents "
           "with grab/lock actions, NOT a high-DOF humanoid.", "table_cell"),
         P("This project is the direct successor: can a single-agent, curiosity-only approach reproduce "
           "Baker-style emergence on a bipedal humanoid?", "table_cell")],
    ]
    elems.append(tbl(pos_data, [1.3*inch, 2.4*inch, 2.5*inch]))
    elems.append(PageBreak())
    return elems


def build_schedule():
    elems = []
    elems.append(P("13.  Week-by-Week Master Schedule", "h1"))
    elems.append(HR())

    sch_data = [
        [P("Week", "table_hdr"), P("Phase", "table_hdr"), P("Key Tasks", "table_hdr"), P("Deliverables / Checkpoints", "table_hdr")],

        # Pre
        [P("Pre\n(Before W1)", "table_cell"), P("Pre-Phase\nSetup", "table_cell"),
         P("RCAC accounts, GPU allocation, hardware inventory, G1 procurement, software install, "
           "GitHub repo setup, wandb project, safety gantry check", "table_cell"),
         P("All pre-phase checklist items complete. Isaac Lab running on cluster.", "table_cell")],

        # Phase 1
        [P("Week 1", "table_cell"), P("Phase 1\nEnv Design", "table_cell"),
         P("Isaac Lab install + cluster test. G1 URDF integration + validation (all 43 DOF). "
           "Begin Distant Target + Elevated Button environment code.", "table_cell"),
         P("G1 URDF loads in Isaac Lab. 100-step random rollout: no NaN. Two env classes started.", "table_cell")],

        [P("Week 2", "table_cell"), P("Phase 1\nEnv Design", "table_cell"),
         P("Complete all 5 task environments. Implement procedural randomisation module. "
           "Write unit tests for all environments. wandb integration test.", "table_cell"),
         P("All 5 environments pass unit tests. Randomisation confirmed. wandb live.", "table_cell")],

        [P("Week 3", "table_cell"), P("Phase 1\nEnv Design", "table_cell"),
         P("Full cluster test: 4096 parallel envs, 4×A100, 1 hour. Random-agent baseline success rates. "
           "Isaac Lab viewer demo. Phase 1 review meeting.", "table_cell"),
         P("<b>PHASE 1 EXIT:</b> 4096 envs stable for 1 hour. All 5 envs in viewer demo.", "table_cell")],

        # Phase 2
        [P("Week 4", "table_cell"), P("Phase 2\nCuriosity Training", "table_cell"),
         P("Implement RND, DRND, RDD, ICM modules. Implement asymmetric actor-critic. "
           "Set up PPO training loop with dual-head critic. Write observation normalisation.", "table_cell"),
         P("All curiosity modules implemented and unit-tested. Training loop runs for 1M steps.", "table_cell")],

        [P("Week 5", "table_cell"), P("Phase 2\nCuriosity Training", "table_cell"),
         P("Launch first full training run on Distant Target task (RND, 500M steps). "
           "Run hyperparameter sweep (lr, GAE λ, clip ε) on 100M steps. Monitor wandb.", "table_cell"),
         P("First training run launched. Sweep results for top-3 hyperparameter configs.", "table_cell")],

        [P("Week 6", "table_cell"), P("Phase 2\nCuriosity Training", "table_cell"),
         P("Monitor Distant Target training for emergence events. Launch DRND run. "
           "Launch RDD run. Check if tool_contact_count is increasing. "
           "If no contact by 100M steps: review curriculum, consider light reward shaping.", "table_cell"),
         P("First tool-contact event logged (target). Learning curves clearly showing curiosity driving exploration.", "table_cell")],

        [P("Week 7", "table_cell"), P("Phase 2\nCuriosity Training", "table_cell"),
         P("If Distant Target shows ≥20% success: unlock Elevated Button in curriculum. "
           "Launch ablation A1 (curiosity mechanisms) runs. "
           "Launch ablation A2 (critic design) runs.", "table_cell"),
         P("Curriculum Stage 2 unlocked. Ablations A1 + A2 running.", "table_cell")],

        [P("Week 8", "table_cell"), P("Phase 2\nCuriosity Training", "table_cell"),
         P("Continue full training runs. Launch ablation A3 (visual input) + A4 (curriculum staging). "
           "If budget allows: launch ablation A6 (QFLEX comparison). "
           "Collect video rollouts every 50M steps.", "table_cell"),
         P("Ablations A3 + A4 running. At least 3 video recordings of tool-use episodes saved.", "table_cell")],

        [P("Week 9", "table_cell"), P("Phase 2\nCuriosity Training", "table_cell"),
         P("Collect final training results. Run 10-seed emergence rate experiment. "
           "Trajectory clustering analysis. Phase 2 review meeting. "
           "Select best checkpoint for Phase 3 distillation.", "table_cell"),
         P("<b>PHASE 2 EXIT:</b> ≥1 task at >20% success. Emergence rate computed. Best checkpoint selected.", "table_cell")],

        # Phase 3
        [P("Week 10", "table_cell"), P("Phase 3\nDistillation", "table_cell"),
         P("DAgger distillation (20 iterations). Build demonstration dataset from best curiosity policy. "
           "Train student policy. Verify inference <10 ms. "
           "Begin residual RL: freeze base, train residual MLP.", "table_cell"),
         P("student_policy_v1.pt saved. Inference confirmed <10 ms.", "table_cell")],

        [P("Week 11", "table_cell"), P("Phase 3\nDistillation", "table_cell"),
         P("Complete residual RL (50M steps). Domain randomisation applied. "
           "MuJoCo cross-validation (100 eval episodes). "
           "Compare deployable policy vs. base in MuJoCo. Phase 3 review meeting.", "table_cell"),
         P("<b>PHASE 3 EXIT:</b> deployable_policy_v1.pt. MuJoCo ≥50% of Isaac Lab success rate.", "table_cell")],

        # Phase 4
        [P("Week 12", "table_cell"), P("Phase 4\nSim-to-Real", "table_cell"),
         P("TensorRT export to Jetson Orin NX. Validate latency ≤10 ms on hardware. "
           "Stage 1 hardware tests: static tool interaction only, tethered. "
           "Physical task scene setup.", "table_cell"),
         P("TensorRT running on G1. Stage 1 (static) tests complete, no safety events.", "table_cell")],

        [P("Week 13", "table_cell"), P("Phase 4\nSim-to-Real", "table_cell"),
         P("Stage 2: Slow-motion (25% speed) full task tests, tethered. "
           "Run 20 consecutive episodes per task. Log safety events. "
           "Begin video documentation.", "table_cell"),
         P("Stage 2 tests complete. Any instabilities identified and addressed.", "table_cell")],

        [P("Week 14", "table_cell"), P("Phase 4\nSim-to-Real", "table_cell"),
         P("Stage 3: Full-speed evaluation (if Stage 2 clear). "
           "50 real-world evaluation episodes. Record successes. "
           "High-quality video recording of successful episodes. Phase 4 review meeting.", "table_cell"),
         P("<b>PHASE 4 EXIT:</b> 50 episodes evaluated. ≥1 task shows real-world success. Video saved.", "table_cell")],

        # Phase 5
        [P("Week 15", "table_cell"), P("Phase 5\nPaper Writing", "table_cell"),
         P("Write Abstract, Introduction, Related Work, Method sections. "
           "Generate all figures and plots. Internal draft review.", "table_cell"),
         P("Full first draft circulated to all co-authors and advisor.", "table_cell")],

        [P("Week 16", "table_cell"), P("Phase 5\nPaper Writing", "table_cell"),
         P("Revise draft. Write Conclusion + Limitations. "
           "Produce supplementary video. Code release preparation. "
           "Submit to target venue.", "table_cell"),
         P("<b>PROJECT COMPLETE:</b> Paper submitted. Code released on GitHub. Checkpoints archived.", "table_cell")],
    ]
    elems.append(tbl(sch_data, [0.55*inch, 0.85*inch, 2.8*inch, 2.0*inch]))
    elems.append(SP(8))
    elems.append(HR(RED))
    elems.append(P("End of Implementation Plan — Curiosity-Driven RL for Emergent Tool Use on Humanoid Robot",
                   "caption"))
    elems.append(P("Purdue University · School of Mechanical Engineering · April 2026", "caption"))
    return elems


# ── Build & save ──────────────────────────────────────────────────────────────

def main():
    out = "/home/kevin/projects/CDL/plan.pdf"
    doc = make_doc(out)

    story = []
    story += build_cover()
    story += build_toc()
    story += build_overview()
    story += build_pre_phase()
    story += build_phase1()
    story += build_phase2()
    story += build_phase3()
    story += build_phase4()
    story += build_phase5()
    story += build_metrics()
    story += build_ablations()
    story += build_risks()
    story += build_resources()
    story += build_literature()
    story += build_schedule()

    doc.build(story)
    print(f"PDF generated: {out}")

if __name__ == "__main__":
    main()
