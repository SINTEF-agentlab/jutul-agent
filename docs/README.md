---
hide:
  - navigation
  - toc
---

<div class="ja-hero" markdown>
<div class="ja-hero-text" markdown>
<h1>jutul-agent</h1>
<p class="ja-tagline">A scientific AI agent for differentiable simulators built
on the <a href="https://github.com/sintefmath/Jutul.jl">Jutul</a> framework.
Ask for a simulation in plain language. The agent sets it up, runs it in a
persistent Julia session, analyses and plots the results, and iterates:
fixing mistakes and refining the next run.</p>

[Get started](usage.md){ .md-button .md-button--primary }
[How it works](architecture.md){ .md-button }
</div>
<div class="ja-terminal">
<div class="ja-terminal-bar"><span></span><span></span><span></span>
<div class="ja-terminal-title">jutul-agent · my-battery-run</div></div>
<pre><code><span class="ja-t-dim">$</span> uv tool install jutul-agent
<span class="ja-t-dim">$</span> jutul-agent key openai
<span class="ja-t-dim">$</span> mkdir my-battery-run &amp;&amp; cd my-battery-run
<span class="ja-t-dim">$</span> jutul-agent init --sim battmo
<span class="ja-t-dim">$</span> jutul-agent
jutul-agent server on <span class="ja-t-ok">http://127.0.0.1:8742</span> (simulator: battmo)</code></pre>
</div>
</div>

<p align="center">
  <img src="images/web-fimbul.png" alt="jutul-agent web UI: a Fimbul geothermal doublet with the 3D temperature field over time" width="47%">
  &nbsp;&nbsp;&nbsp;
  <img src="images/web-battmo.png" alt="jutul-agent web UI: a BattMo discharge-rate investigation with plots and a report" width="47%">
</p>
<p align="center" markdown><small>The web UI (`jutul-agent`): chat on the left; interactive plots and
reports pinned in a canvas on the right.</small></p>

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Usage**

    ---

    Install, initialise a workspace, start the web UI, pick models.

    [:octicons-arrow-right-24: Installation and usage](usage.md) ·
    [CLI](cli.md) ·
    [Configuration](configuration.md) ·
    [Models](models.md) ·
    [Approval and safety](approval.md)

-   :material-cog-outline:{ .lg .middle } **How it works**

    ---

    The component map and the pieces behind it: the persistent Julia
    kernel, the filesystem, memory, context, and the trace.

    [:octicons-arrow-right-24: Architecture](architecture.md) ·
    [Turns](turns.md) ·
    [Kernel](julia-kernel.md) ·
    [Filesystem](filesystem.md) ·
    [Memory](memory.md) ·
    [Context](context.md) ·
    [Trace](trace.md)

-   :material-puzzle-plus:{ .lg .middle } **Extending the agent**

    ---

    Add a simulator as data, improve behavior with the right lever, and
    prove every change against the benchmark.

    [:octicons-arrow-right-24: Adding a simulator](adding-a-simulator.md) ·
    [Improving the agent](improving-the-agent.md) ·
    [Evaluation](evaluation.md)

-   :material-flask-outline:{ .lg .middle } **Development**

    ---

    Repo layout, the test tiers and what CI runs, dependency policy, and
    the agent-first working method behind it all.

    [:octicons-arrow-right-24: Development](development.md) ·
    [Testing](testing.md) ·
    [Agent-first](agent-first.md)

</div>

## Supported simulators

<p class="ja-logos">
  <a href="https://github.com/sintefmath/JutulDarcy.jl"><img src="https://raw.githubusercontent.com/sintefmath/JutulDarcy.jl/main/docs/src/assets/logo_wide.png" alt="JutulDarcy" style="height:1.6rem"></a>
  <a href="https://github.com/BattMoTeam/BattMo.jl"><img src="https://raw.githubusercontent.com/BattMoTeam/BattMo.jl/main/docs/src/assets/battmologo_text.png" alt="BattMo" style="height:2.8rem"></a>
  <a href="https://github.com/sintefmath/Fimbul.jl"><img src="https://raw.githubusercontent.com/sintefmath/Fimbul.jl/main/docs/src/assets/logo_text_wide.png" alt="Fimbul" style="height:1.9rem"></a>
  <a href="https://github.com/sintefmath/Mocca.jl"><img src="https://raw.githubusercontent.com/sintefmath/Mocca.jl/main/docs/src/assets/mocca_small_transparent.png" alt="Mocca" style="height:2.4rem"></a>
</p>

| Simulator | Package | Domain |
|---|---|---|
| `jutuldarcy` | [JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl) | Reservoir / multi-phase flow |
| `battmo` | [BattMo.jl](https://github.com/BattMoTeam/BattMo.jl) | Lithium-ion (and other) battery cells |
| `fimbul` | [Fimbul.jl](https://github.com/sintefmath/Fimbul.jl) | Geothermal (ATES, BTES, doublet, EGS) |
| `mocca` | [Mocca.jl](https://github.com/sintefmath/Mocca.jl) | Adsorption-based CO₂ capture |
