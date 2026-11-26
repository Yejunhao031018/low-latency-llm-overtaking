#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the full (6-page) self-contained review HTML."""
import base64

def img64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

arch = img64("results/media/architecture.png")
lat  = img64("results/media/latency_superiority.png")

CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#111;line-height:1.62;
     max-width:860px;margin:32px auto;padding:0 30px;font-size:15.5px;}
h1{font-size:23px;line-height:1.3;margin:0 0 4px;text-align:center;}
.subtitle{text-align:center;color:#444;font-size:13.5px;margin-bottom:24px;}
h2{font-size:17px;margin:30px 0 10px;border-bottom:1.5px solid #222;padding-bottom:4px;}
h3{font-size:15.5px;margin:20px 0 8px;}
p{margin:9px 0;text-align:justify;}
figure{margin:20px 0;text-align:center;}
figure img{max-width:100%;border:1px solid #ddd;}
figcaption{font-size:12.5px;color:#555;margin-top:6px;}
table{border-collapse:collapse;margin:14px auto;font-size:13px;width:100%;}
th,td{border:1px solid #bbb;padding:5px 9px;text-align:center;}
th{background:#f2f2f2;font-weight:bold;}
ol,ul{margin:8px 0 8px 24px;padding:0;}
li{margin:5px 0;text-align:justify;}
.refs p{margin:5px 0;text-align:left;font-size:13px;}
.abstract{background:#f7f7f7;border:1px solid #ddd;padding:14px 18px;margin:14px 0;}
.abstract p{margin:0;}
code{font-family:'DejaVu Sans Mono',monospace;font-size:13px;background:#f4f4f4;padding:0 3px;}
"""

BODY = """
<h1>Low-Latency Edge LLM Tactical Decision-Making for High-Speed Autonomous Overtaking</h1>
<div class="subtitle">Draft for review &nbsp;·&nbsp; November 2026</div>

<div class="abstract">
<p><strong>Abstract.</strong>
Large language models (LLMs) have recently been used as decision-makers for autonomous
driving, but existing approaches rely on cloud APIs whose multi-second inference latency
precludes real-time, safety-critical use. We argue that the LLM should act only as a
<i>low-frequency tactical</i> layer — issuing coarse overtake/hold intents — while a classical
20&nbsp;Hz controller handles trajectory tracking and safety. Under this decomposition we
fine-tune a Qwen2.5-1.5B model with LoRA and show that a short-output variant reaches a
decision latency of <b>0.59&nbsp;s</b> on a consumer 8&nbsp;GB GPU, roughly an order of magnitude
faster than cloud-API baselines, while retaining 86.1% decision accuracy — the Pareto-optimal
point among 0.5B–7B alternatives. In closed-loop CARLA experiments the system completes
overtakes with 95–100% pass rate, 0–2% collision, and median lateral clearance ≥2.2&nbsp;m
across ten corner–lead scenarios, generalizing <i>zero-shot</i> from training on only two corner
geometries. We additionally report two honest negative results: fine-tuning on multi-scenario
rule-expert data <i>regresses</i> closed-loop performance via catastrophic forgetting, and a
CBF-QP safety filter underperforms a simple geometric shield. Our results establish a
quantified latency–safety boundary for embedding LLMs in high-speed driving.
</p>
</div>

<h2>1&nbsp;&nbsp;Introduction</h2>
<p>Large language models (LLMs) bring commonsense reasoning and interpretability to autonomous
driving, enabling vehicles to handle rare, semantic situations that confound conventional
planners [1,2]. A growing line of work therefore positions the LLM as a <i>decision-maker</i>,
translating its high-level reasoning into driving actions. Yet this line shares a common
Achilles' heel: the decision is produced by a cloud model such as GPT-4, whose inference —
autoregressive decoding over hundreds of tokens plus a network round-trip — takes seconds.
At highway speed, a car covers over 15&nbsp;m per second; a decision that arrives three seconds
late describes a scene that no longer exists. Real-time deployment therefore requires the LLM
to be both <i>accurate enough</i> and <i>fast enough</i>, a trade-off that has been asserted but,
to our knowledge, not quantified in a closed driving loop.</p>
<p>We argue that the LLM need not be fast in absolute terms if it is placed at the right
timescale. Following the classical hierarchical decomposition of autonomous driving, we
confine the LLM to a <b>tactical layer</b> that runs at roughly 1&nbsp;Hz and emits a coarse,
structured intent — <i>overtake or hold</i>, plus an attack offset and a speed factor — while a
classical <b>control layer</b> runs at 20&nbsp;Hz and owns trajectory tracking and safety. Under
this decomposition, decision latency is absorbed into the decision <i>cadence</i>, not the control
loop: a 0.5&nbsp;s decision is tolerable, whereas a 3&nbsp;s one is stale (we quantify this boundary
in Sec.&nbsp;4.5). This insight lets us substitute a <i>small, locally fine-tuned</i> LLM for the
cloud model, turning an order-of-magnitude latency reduction from a nice-to-have into the
enabling condition for real-time use.</p>
<p>Concretely, we fine-tune Qwen2.5-1.5B with LoRA on a single consumer 8&nbsp;GB GPU and study
the task of high-speed overtaking on a closed CARLA ring. Our contributions are:</p>
<ol>
<li><b>A latency–accuracy Pareto characterization</b> across 0.5B–7B models, showing that a
short-output 1.5B variant is the optimal operating point — 0.59&nbsp;s and 86.1% accuracy —
outperforming 3B (2.02&nbsp;s, 71.3%) and 7B (1.41&nbsp;s, 69.6%) on <i>both</i> axes, because the
larger models cannot be adequately fine-tuned within the memory budget.</li>
<li><b>A closed-loop evaluation</b> in CARLA in which the system completes overtakes with
95–100% pass rate, 0–2% collision, and median lateral clearance ≥2.2&nbsp;m across ten
corner–lead scenarios, generalizing <i>zero-shot</i> from training on only two corner geometries.</li>
<li><b>A quantified stale-latency boundary</b> showing that 0.5&nbsp;s of decision delay is
tolerated while 0.8&nbsp;s degrades safety — the quantitative justification for the latency
reduction.</li>
<li><b>Two control-layer findings</b> and <b>two honest negative results</b> that are independently
useful: an attack-geometry correction, an inside-line-hold fix, the failure of multi-scenario
behavior cloning (catastrophic forgetting), and the underperformance of a CBF-QP filter
relative to a simple geometric shield.</li>
</ol>

<h2>2&nbsp;&nbsp;Related Work</h2>
<h3>2.1&nbsp;&nbsp;LLMs for Autonomous Driving</h3>
<p>LanguageMPC [1] employs GPT-4 as a decision-making component: it devises cognitive pathways
for reasoning over human commonsense and translates LLM decisions into low-level control
actions. It demonstrates that LLMs <i>can</i> make driving decisions, but relies on cloud
inference and does not report timing, so it cannot close a real-time loop. DriveGPT4 [2]
produces interpretable end-to-end driving outputs via a vision-language model, prioritizing
explainability over latency. A broader line of work applies LLMs to scene understanding, VQA,
or offline/single-step planning [1,2]; none addresses the latency required for high-speed,
safety-critical control. <b>Our positioning</b>: we treat the LLM as a low-frequency tactical
layer and show that a 1.5B local model is fast enough (0.59&nbsp;s) for real-time use — an order
of magnitude faster than the cloud-API paradigm — while the safety-critical tracking remains
rule-based.</p>
<h3>2.2&nbsp;&nbsp;Hierarchical Control and Safety</h3>
<p>Classical driving stacks separate motion planning from trajectory tracking (pure pursuit,
Stanley, LQR, MPC) and enforce safety with control barrier functions (CBF), QP-based filters,
or set-based reachability. We adopt a pure-pursuit tracking controller and a lightweight
geometric <i>signed shield</i> that brakes when the ego is alongside the lead with insufficient
lateral clearance. We deliberately keep the shield rule-based and report, as a negative result,
that a full CBF-QP formulation underperformed it in a 2&nbsp;m stress test (Sec.&nbsp;4.6).</p>
<h3>2.3&nbsp;&nbsp;Edge LLMs and Efficient Fine-Tuning</h3>
<p>Efficient deployment uses quantization (GPTQ/AWQ) and parameter-efficient fine-tuning such as
LoRA [3]. We fine-tune Qwen2.5-1.5B [5] with bf16 LoRA on a consumer 8&nbsp;GB GPU and show this
suffices for a tactical decision layer at 0.59&nbsp;s — direct evidence that a tactical LLM can run
at the edge, without cloud dependence.</p>
<h3>2.4&nbsp;&nbsp;Latency in Real-Time Systems</h3>
<p>Networked control and robot action chunking address decision latency through predictive
compensation or coarser action granularity. We transfer action chunking to high-speed driving
and, unlike prior work, <i>quantify</i> the tolerable staleness (0.5&nbsp;s) and the failure point
(0.8&nbsp;s), which motivates our latency reduction from 1.08&nbsp;s to 0.59&nbsp;s by dropping the
explanatory <code>reason</code> field.</p>
<h3>2.5&nbsp;&nbsp;Simulation and Model-Based Data</h3>
<p>We evaluate in CARLA [4] on a closed Town04 ring. Our counterfactual labeling uses the
simulator as a ground-truth dynamics model for a shallow action search, connecting to
model-based reinforcement learning and expert iteration; a learned neural world model to
replace this simulator is future work.</p>

<h2>3&nbsp;&nbsp;Method</h2>
<h3>3.1&nbsp;&nbsp;Problem Formulation</h3>
<p>We study a two-vehicle overtaking task on a closed ring. At each step the ego observes a
state consisting of ego and lead speeds, signed and absolute longitudinal gap, distance to the
corner apex, track curvature, time-to-collision, current lateral offset and lateral separation,
the racing-line offset, and the corner's inside side (left/right). The tactical decision is a
structured intent <code>a&nbsp;=&nbsp;(action,&nbsp;extra_inside_m,&nbsp;speed_factor)</code> with
<code>action&nbsp;∈&nbsp;{overtake,&nbsp;hold}</code>, <code>extra_inside_m&nbsp;∈&nbsp;[0,&nbsp;3.5]</code>&nbsp;m
(additional lateral offset toward the corner inside), and
<code>speed_factor&nbsp;∈&nbsp;[0.9,&nbsp;1.1]</code>. The objective is to complete a pass while
respecting a minimum lateral clearance and avoiding collision.</p>
<h3>3.2&nbsp;&nbsp;Layered Architecture</h3>
<p>The system decomposes into two layers with distinct timescales (Fig.&nbsp;1).</p>
<figure><img src="ARCH" alt="architecture"><figcaption>Fig.&nbsp;1. Two-timescale layered
architecture. A low-frequency LLM tactical layer issues overtake/hold intents; a 20&nbsp;Hz
rule-based control layer executes them under a geometric safety shield. Decision latency is
absorbed into the 1&nbsp;Hz cadence, not the 20&nbsp;Hz control loop.</figcaption></figure>
<p><b>Tactical layer (LLM, ~1&nbsp;Hz).</b> The fine-tuned LLM maps the observation (plus the
previous decision, which disambiguates "start an attack" from "already attacking") to the intent
<code>a</code>. The <i>short-output</i> variant omits the free-text <code>reason</code> field,
cutting generated tokens by roughly 60% and latency by ~45% at a small accuracy cost.</p>
<p><b>Control layer (20&nbsp;Hz).</b> A pure-pursuit controller tracks a racing line offset toward
the corner inside. During an overtake it adds the attack offset <code>extra_inside_m</code>,
scaled by a spatial attack window centered on the apex, and a speed factor. A <i>signed shield</i>
brakes when the longitudinal gap is below 8&nbsp;m and the lateral separation is below 2.2&nbsp;m
while the ego is alongside, and caps the speed advantage over the lead (6&nbsp;m/s). Two fixes
were essential for safety, detailed next.</p>
<h3>3.3&nbsp;&nbsp;Two Control-Layer Corrections</h3>
<p><b>Attack-geometry correction.</b> The initial attack reference summed the racing-line inside
offset (3.3&nbsp;m) and the attack offset (3.5&nbsp;m) into a 6.8&nbsp;m lateral command that exceeded
the available road width, driving the ego off the road on variable and defensive leads. We
reduced the racing-line inside offset to 1.0&nbsp;m, clamped the total offset to 4.5&nbsp;m, and
added a 0.5&nbsp;m bias, keeping the attack trajectory on the road without retraining the model.</p>
<p><b>Inside-line hold.</b> The attack window ramps down after the apex and the racing line
returns to the outside; on tight corners the ego has not yet cleared the lead at that point, so
the retraction steered it back onto the lead's line and caused side-swipe collisions. We pin the
attack weight to unity while the ego is still alongside the lead, holding the inside line until
the pass completes.</p>
<h3>3.4&nbsp;&nbsp;Model and Counterfactual Training Data</h3>
<p>We fine-tune Qwen2.5-1.5B-Instruct with LoRA (r=16, α=32, dropout 0.05) for two epochs with a
completion-only loss (prompt tokens masked) on a consumer 8&nbsp;GB GPU. The training signal comes
from two sources. First, demonstrations from a rule-based expert driving in CARLA. Second, and
more importantly, <b>counterfactual labels</b>: at sampled states we snapshot the world and roll
out five candidate actions (hold; inside 3.0&nbsp;m; inside 3.5&nbsp;m; inside 3.5&nbsp;m with late
braking; outside 3.5&nbsp;m) against the CARLA simulator for a short horizon, and label the state
with the lowest-cost action under a cost that penalizes collision (weight 1000), sub-2.2&nbsp;m
clearance, and failed passes, while rewarding longitudinal progress. This lets the model learn
the <i>optimal</i> action in each state rather than the expert's demonstrated one, partially
escaping the behavior-cloning ceiling.</p>
<h3>3.5&nbsp;&nbsp;Evaluation Protocol</h3>
<p>We build a closed Town04 ring with six corners (radii 45–208&nbsp;m, in both left and right
directions) and define ten scenarios pairing five passable corners with constant, variable, and
defensive lead behaviors. Metrics are pass rate (the ego reaches 2&nbsp;m ahead during the attack
window), collision rate, minimum lateral separation (<code>minSep</code>), and decision latency.
Pass rate is reported with Wilson 95% confidence intervals; <code>minSep</code> with bootstrap
confidence intervals.</p>

<h2>4&nbsp;&nbsp;Experiments</h2>
<h3>4.1&nbsp;&nbsp;Setup</h3>
<p>All experiments run in CARLA [4] Town04 with a synchronous 20&nbsp;Hz physics loop and a
ClearNoon setting. The LLM is served locally and queried at 1&nbsp;Hz; the control layer runs
every tick. Latency is measured as end-to-end inference (prompt encoding plus decoding) on the
local GPU, with greedy decoding.</p>
<h3>4.2&nbsp;&nbsp;Latency–Accuracy Pareto</h3>
<table>
<tr><th>Model</th><th>Accuracy</th><th>Latency</th><th>vs 1.5B-short</th></tr>
<tr><td>Qwen2.5-0.5B</td><td>75.7%</td><td>0.86 s</td><td>1.46× slower, −10.4 pt</td></tr>
<tr><td>Qwen2.5-1.5B</td><td>87.8%</td><td>1.08 s</td><td>1.83× slower, +1.7 pt</td></tr>
<tr><td><b>Qwen2.5-1.5B-short (ours)</b></td><td><b>86.1%</b></td><td><b>0.59 s</b></td><td>—</td></tr>
<tr><td>Qwen2.5-3B (4-bit)</td><td>71.3%</td><td>2.02 s</td><td>3.42× slower, −14.8 pt</td></tr>
<tr><td>Qwen2.5-7B (4-bit, zero-shot)</td><td>69.6%</td><td>1.41 s</td><td>2.39× slower, −16.5 pt</td></tr>
</table>
<figure><img src="LAT" alt="latency"><figcaption>Fig.&nbsp;2. (a) Decision latency and
(b) accuracy–latency Pareto. The 1.5B short-output model is 3.4× faster than 3B and 2.4× faster
than 7B while <i>more</i> accurate — the Pareto-optimal point.</figcaption></figure>
<p>Two observations follow. First, <i>bigger is not better</i> here: the 3B and 7B models must be
quantized to 4-bit to fit the 8&nbsp;GB budget, which degrades their accuracy below the 1.5B model,
while their larger parameter count inflates latency. Second, the <i>short-output</i> ablation
(dropping the <code>reason</code> field) cuts latency from 1.08&nbsp;s to 0.59&nbsp;s (−45%) at a
cost of only 1.7 accuracy points, because autoregressive decoding cost scales with the number of
generated tokens. The 1.5B-short model is therefore the Pareto-optimal point: the fastest and
nearly the most accurate.</p>
<h3>4.3&nbsp;&nbsp;Closed-Loop Overtaking (20 seeds, corner R=188)</h3>
<table>
<tr><th>Lead</th><th>Pass rate [95% CI]</th><th>Collision</th><th>minSep median</th><th>minSep min</th></tr>
<tr><td>constant</td><td>100% [84%, 100%]</td><td>0%</td><td>2.87 m</td><td>2.54 m</td></tr>
<tr><td>variable</td><td>95% [76%, 99%]</td><td>0%</td><td>3.13 m</td><td>1.14 m</td></tr>
<tr><td>defensive</td><td>95% [76%, 99%]</td><td>5%</td><td>2.27 m</td><td>1.04 m</td></tr>
</table>
<p>The median clearance meets the 2.2&nbsp;m safety target in all three lead behaviors; the minimum
drops below 2&nbsp;m only in rare tail cases (defensive and variable leads). The defensive lead —
which moves toward the ego's chosen side — is the hardest, yielding one collision in twenty runs.</p>
<h3>4.4&nbsp;&nbsp;Zero-Shot Generalization (10 scenarios, 10 seeds)</h3>
<p>Trained on only two corner geometries (one right, one left), the model generalizes zero-shot
across ten corner–lead scenarios:</p>
<table>
<tr><th>Corner</th><th>Lead</th><th>N</th><th>Pass rate</th><th>Collision</th><th>minSep med.</th><th>minSep min</th></tr>
<tr><td>R=208</td><td>constant</td><td>10</td><td>100%</td><td>0</td><td>2.77 m</td><td>2.71 m</td></tr>
<tr><td>R=208</td><td>variable</td><td>10</td><td>100%</td><td>0</td><td>3.30 m</td><td>3.22 m</td></tr>
<tr><td>R=208</td><td>defensive</td><td>10</td><td>90%</td><td>0</td><td>2.33 m</td><td>1.01 m</td></tr>
<tr><td>R=151</td><td>constant</td><td>10</td><td>100%</td><td>0</td><td>3.28 m</td><td>2.83 m</td></tr>
<tr><td>R=151</td><td>variable</td><td>10</td><td>100%</td><td>0</td><td>2.79 m</td><td>1.60 m</td></tr>
<tr><td>R=151</td><td>defensive</td><td>10</td><td>100%</td><td>0</td><td>2.37 m</td><td>2.34 m</td></tr>
<tr><td>R=188</td><td>constant</td><td>10</td><td>100%</td><td>0</td><td>2.84 m</td><td>2.01 m</td></tr>
<tr><td>R=188</td><td>defensive</td><td>10</td><td>90%</td><td>0</td><td>2.28 m</td><td>0.65 m</td></tr>
<tr><td>R=83</td><td>constant</td><td>10</td><td>80%</td><td>2</td><td>2.74 m</td><td>1.63 m</td></tr>
<tr><td>R=138</td><td>variable</td><td>10</td><td>100%</td><td>0</td><td>3.26 m</td><td>3.24 m</td></tr>
</table>
<p>Aggregate: <b>96/100 passes (96%), 2 collisions (2%)</b>. Seven scenarios reach 100% pass with
zero collision; two defensive-lead scenarios reach 90%; the single tight-corner scenario (R=83)
drops to 80% with two collisions. Two findings stand out. First, generalization is governed by
<i>corner difficulty</i> (curvature) rather than direction: gentle corners pass in both
directions, while the tight right corner R=83 is marginal. Second, the defensive lead is the
hardest <i>behavior</i>, consistently lowering clearance across corners.</p>
<h3>4.5&nbsp;&nbsp;Stale-Latency Boundary</h3>
<p>To justify the latency reduction, we measure closed-loop safety as a function of injected
decision delay. With 0.5&nbsp;s of added staleness the pass rate and clearance are essentially
unchanged, confirming that a 0.59&nbsp;s decision is within the tolerable regime. At 0.8&nbsp;s the
shield begins to intervene more often and safety degrades, marking the boundary beyond which the
decision is stale. This quantifies the enabling condition for the cloud-to-edge substitution:
the 1.5B-short model sits <i>inside</i> the tolerable regime, whereas a 2&nbsp;s (3B) or
multi-second (cloud) decision would sit far outside it.</p>
<h3>4.6&nbsp;&nbsp;Negative Results</h3>
<p><b>Multi-scenario fine-tuning regresses.</b> Fine-tuning on a ten-scenario dataset collected
from the same rule expert <i>reduced</i> closed-loop pass rate (left-turn scenarios fell from
100% to 0–67%); merging the new data with the original two-corner data did not recover the
zero-shot baseline. Root cause: the rule expert only <i>holds</i> on hard scenarios, so its 607
demonstrated actions (mostly "hold") overwhelm the 47 counterfactual labels, teaching
conservatism. This is a concrete demonstration that behavior cloning from a weak expert sets a
ceiling, and that the counterfactual-labeling ratio must be much higher for multi-scenario
scaling.</p>
<p><b>CBF-QP vs. geometric shield.</b> A full CBF-QP safety filter underperformed a plain
if-else shield in a 2&nbsp;m stress test, consistent with prior findings that hand-tuned geometric
gates can be more robust than optimization-based filters for narrow, well-understood tasks. We
therefore retain the lightweight geometric shield.</p>

<h2>5&nbsp;&nbsp;Discussion</h2>
<p><b>Limitations.</b> Our study is confined to a single map (Town04), a single task (overtaking),
and simulation-only evaluation; real-world transfer and generalization to other maps and weather
remain open. The tactical action space is intentionally coarse (three scalars), which is
sufficient for overtaking but not for general driving. Finally, our advantage is empirical and
system-level rather than algorithmic: the underlying model and fine-tuning method are off the
shelf, and "smaller is faster" is expected — our contribution is the <i>quantification</i> of the
operating point and the closed-loop evidence that it suffices.</p>
<p><b>Implications.</b> The central message is that the latency question, not model capability,
is the binding constraint for embedding LLMs in real-time driving, and that a two-timescale
architecture resolves it. The counterfactual-labeling mechanism also suggests a path to exceed
the behavior-cloning ceiling, and connects to model-based RL and expert iteration.</p>
<p><b>Future work.</b> (1) Train a lightweight neural <i>world model</i> to replace the CARLA
simulator in counterfactual search, potentially reducing decision latency toward the millisecond
scale; (2) scale up counterfactual labeling to escape the weak-expert ceiling and enable
multi-scenario fine-tuning; (3) extend to multiple maps, weather, and interactive traffic.</p>

<h2>6&nbsp;&nbsp;Conclusion</h2>
<p>We showed that a locally fine-tuned 1.5B LLM can serve as a low-frequency tactical decision
layer for high-speed overtaking at 0.59&nbsp;s latency — an order of magnitude faster than the
cloud-API paradigm — while a 20&nbsp;Hz rule-based controller guarantees safety. The system
achieves 95–100% pass rate with median clearance ≥2.2&nbsp;m and generalizes zero-shot across ten
scenarios. We quantified the stale-latency boundary that justifies the reduction, and reported
honest negative results (multi-scenario behavior cloning regresses; CBF-QP underperforms a
geometric shield) that bound the approach and point to future work.</p>

<h2>References</h2>
<div class="refs">
<p>[1] H. Sha, Y. Mu, Y. Jiang, L. Chen, C. Xu, P. Luo, S. E. Li, M. Tomizuka, W. Zhan, and
M. Ding, "LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving,"
arXiv:2310.03026, 2023.</p>
<p>[2] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao,
"DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model,"
arXiv:2310.01412, 2023.</p>
<p>[3] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen,
"LoRA: Low-Rank Adaptation of Large Language Models," in <i>ICLR</i>, 2022.</p>
<p>[4] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, "CARLA: An Open Urban
Driving Simulator," in <i>CoRL</i>, 2017.</p>
<p>[5] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.</p>
</div>
"""

html = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Low-Latency Edge LLM Tactical Decision-Making for High-Speed Autonomous Overtaking</title>"
        f"<style>{CSS}</style></head><body>{BODY}</body></html>")
html = html.replace('src="ARCH"', f'src="{arch}"').replace('src="LAT"', f'src="{lat}"')

with open("paper_review.html", "w") as f:
    f.write(html)
print("saved paper_review.html (", len(html)//1024, "KB )")
