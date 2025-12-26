#     - Concrete detail, strong verbs, show > tell; minimal

app = mo.App(width="full")

# ----------------------------
# Imports
# ----------------------------
@app.cell
def _():
    import html
    import json
    import os
    from typing import Any

    import marimo as mo
    from pydantic import BaseModel, Field

    return Any, BaseModel, Field, html, json, mo, os


# ----------------------------
# Models
# ----------------------------
@app.cell
def _(BaseModel, Field):
    class StoryStart(BaseModel):
        """
        You are a sharp, imaginative fiction writer.

        Task:
        - Produce (1) a concise, compelling story premise and (2) the opening paragraph that launches the story.

        Rules:
        - If the user provides ideas, weave them in organically (don’t just repeat them).
        - If the user provides no ideas, invent something fresh with a surprising combination of
        genre, setting, protagonist, conflict, and twist.
        - Premise: 2–5 sentences, stakes + hook, no spoilers.
        - Opening paragraph: 120–180 words, vivid and concrete, minimal clichés, clear POV,
        grounded scene, ends with a soft hook.
        - Tone should follow user preferences; default to PG-13 if none are given.
        - Avoid copying user phrasing verbatim; enrich and reframe.
        - If user ideas conflict, choose one coherent direction and proceed.

        Output only fields that match this schema.
        """
        premise: str = Field(..., description="2–5 sentences. Stakes + hook, no spoilers.")
        opening_paragraph: str = Field(..., description="120–180 words. Vivid, grounded, ends with a soft hook.")


    class StoryContinue(BaseModel):
        """
        You are a skilled novelist. Write the next paragraph only.

        Inputs:
        - You will be given the story premise and the story-so-far (either the opening paragraph + latest paragraph,
        or a compact analysis summary). Use them to preserve continuity.

        Rules:
        - Output exactly one paragraph of story prose (no headings, no bullets, no analysis).
        - Preserve continuity: characters, tone, POV, tense, world rules.
        - Length target: ~120–200 words unless told otherwise.
        - Concrete detail, strong verbs, show > tell; minimal clichés.
        - Dialogue (if any) should reveal motive or conflict; avoid summary dumps.
        - End with a soft hook/turn that invites the next paragraph.

        Output only fields that match this schema.
        """
        next_paragraph: str = Field(..., description="Exactly one paragraph of continuation prose.")


    class StoryAnalysis(BaseModel):
        """
        You are a story analyst. Produce a succinct “Story-So-Far” handoff so another model can write
        the next paragraph without breaking continuity. Do not write new story prose.

        Inputs:
        - You will be given the story premise and the story text so far (opening paragraph + one or more continuation paragraphs).

        Rules:
        - Extract ground truth only from the provided text/premise. No inventions.
        - Capture continuity anchors: cast, goals, stakes, conflicts, setting rules, POV/tense,
        tone/style markers, motifs, and notable objects.
        - Map causality and current situation.
        - List active threads/hazards: open questions, ticking clocks, contradictions to avoid.
        - Provide 3–5 next-paragraph seeds as beats only (no prose paragraphs).

        Output only fields that match this schema.
        """
        logline: str
        cast: List[str] = Field(default_factory=list, description="Bullets: Name — role/goal/conflict; ties.")
        world_rules: List[str] = Field(default_factory=list, description="Bullets: constraints/rules implied by text.")
        pov_tense_tone: str = Field(..., description="Compact string for POV, tense, and tone/style markers.")
        timeline: List[str] = Field(default_factory=list, description="Bullets: key event → consequence.")
        current_situation: str
        active_threads: List[str] = Field(default_factory=list)
        continuity_landmines: List[str] = Field(default_factory=list)
        ambiguities: List[str] = Field(default_factory=list)
        next_paragraph_seeds: List[str] = Field(..., min_length=3, max_length=5, description="Beats-only options, no prose.")


    return StoryAnalysis, StoryContinue, StoryStart


# ----------------------------
# LLM helpers (OpenAI)
# ----------------------------
@app.cell
def _(Any, StoryAnalysis, StoryContinue, StoryStart):
    _SYSTEM = "You are a helpful assistant. Follow the response model docstring"

    def _require_openai() -> Any:
        from openai import OpenAI  # type: ignore

        return OpenAI()

    def _chat(model: str, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        client = _require_openai()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def story_start(model: str, user_prompt: str, temperature: float = 1.2) -> StoryStart:
        _doc = (StoryStart.__doc__ or "").strip()
        _out = _chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"{_doc}\n\n{user_prompt}"},
            ],
            temperature=temperature,
        )
        return StoryStart.model_validate_json(_out)

    def story_continue_from_text(
        model: str, premise: str, story_text: str, temperature: float = 0.4
    ) -> StoryContinue:
        _doc = (StoryContinue.__doc__ or "").strip()
        _out = _chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{_doc}\n\n"
                        f"Premise:\n{premise}\n\n"
                        f"Story so far:\n{story_text}\n\n"
                        f"Continue."
                    ),
                },
            ],
            temperature=temperature,
        )
        return StoryContinue.model_validate_json(_out)

    def story_continue_from_summary(
        model: str, premise: str, summary: StoryAnalysis, temperature: float = 0.4
    ) -> StoryContinue:
        _doc = (StoryContinue.__doc__ or "").strip()
        _out = _chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{_doc}\n\n"
                        f"Premise:\n{premise}\n\n"
                        f"Checkpoint summary:\n{summary.summary}\n\n"
                        f"Continue."
                    ),
                },
            ],
            temperature=temperature,
        )
        return StoryContinue.model_validate_json(_out)

    def story_analysis(
        model: str, premise: str, story_text: str, temperature: float = 0.2
    ) -> StoryAnalysis:
        _doc = (StoryAnalysis.__doc__ or "").strip()
        _out = _chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"{_doc}\n\nPremise:\n{premise}\n\nStory text:\n{story_text}\n\nAnalyze.",
                },
            ],
            temperature=temperature,
        )
        return StoryAnalysis.model_validate_json(_out)

    return story_analysis, story_continue_from_summary, story_continue_from_text, story_start


# ----------------------------
# Styles
# ----------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
<style>
:root {
  --bg: #0b0f17;
  --panel: rgba(255,255,255,0.06);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --border: rgba(255,255,255,0.08);
}

* { box-sizing: border-box; }

body, .marimo {
  background: radial-gradient(900px 700px at 15% 10%, rgba(128,90,213,0.22), transparent 50%),
              radial-gradient(700px 650px at 85% 25%, rgba(56,189,248,0.18), transparent 45%),
              radial-gradient(800px 700px at 40% 90%, rgba(16,185,129,0.10), transparent 50%),
              var(--bg);
  color: var(--text);
}

.stage {
  max-width: 980px;
  margin: 22px auto;
  padding: 0 18px;
}

.h1 {
  font-size: 34px;
  font-weight: 760;
  letter-spacing: -0.02em;
  margin: 0 0 6px 0;
}

.sub { color: var(--muted); margin: 0; }

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.card h3 {
  margin: 0 0 10px 0;
  font-size: 14px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.72);
}

.pre { white-space: pre-wrap; line-height: 1.55; }

.footer {
  position: sticky;
  bottom: 14px;
  z-index: 5;
  margin-top: 18px;
}

.composer {
  max-width: 980px;
  margin: 0 auto;
  padding: 10px 12px;
  border-radius: 18px;
  background: rgba(0,0,0,0.35);
  border: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(16px);
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.composer .marimo-application textarea {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}

.pill {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.09);
  color: rgba(255,255,255,0.78);
  font-size: 12px;
}

hr.sep {
  border: none;
  height: 1px;
  background: rgba(255,255,255,0.08);
  margin: 14px 0;
}
</style>
"""
    )
    return


# ----------------------------
# State
# ----------------------------
@app.cell
def _(mo):
    state, set_state = mo.state(
        {
            "premise": None,
            "opening": None,
            "paragraphs": [],
            "summaries": [],
            "pending": None,
        }
    )
    return state, set_state


# ----------------------------
# Header + story rendering
# ----------------------------
@app.cell
def _(Any, html, mo, state):
    _s = state

    _title = "LLM"
    _subtitle = "LLM Learns Lore Mostly"

    mo.Html(
        f"""
<div class="stage">
  <div class="h1">{html.escape(_title)}</div>
  <div class="sub">{html.escape(_subtitle)}</div>
</div>
"""
    )

    if not _s["premise"]:
        mo.stop(True)

    _blocks: list[Any] = []
    _blocks.append(
        mo.Html(
            f"""
<div class="stage">
  <div class="card">
    <h3>Premise</h3>
    <div class="pre">{html.escape(_s["premise"])}</div>
  </div>
  <div class="card" style="margin-top: 14px;">
    <h3>Opening</h3>
    <div class="pre">{html.escape(_s["opening"] or "")}</div>
  </div>
</div>
"""
        )
    )

    for _p in _s["paragraphs"]:
        _blocks.append(
            mo.Html(
                f"""
<div class="stage">
  <div class="card"><div class="pre">{html.escape(_p)}</div></div>
</div>
"""
            )
        )

    mo.vstack(_blocks)
    return


# ----------------------------
# Controls
# ----------------------------
@app.cell
def _(mo, os):
    model = mo.ui.text(value=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), label="Model")
    temp_start = mo.ui.slider(0.0, 2.0, step=0.05, value=2.0, label="Start temperature")
    temp_loop = mo.ui.slider(0.0, 1.5, step=0.05, value=0.2, label="Loop temperature")
    prefer_summary = mo.ui.switch(value=True, label="Prefer summary for continuing")
    return model, prefer_summary, temp_loop, temp_start


@app.cell
def _(model, mo, prefer_summary, temp_loop, temp_start):
    mo.Html('<div class="stage"><div class="card">')
    mo.hstack(
        [
            mo.Html('<span class="pill">⚙ Settings</span>'),
            model,
            temp_start,
            temp_loop,
            prefer_summary,
        ],
        gap=12,
        align="center",
    )
    mo.Html("</div></div>")
    return


# ----------------------------
# Prompt composer + send handler
# ----------------------------
@app.cell
def _(
    StoryAnalysis,
    mo,
    model,
    prefer_summary,
    state,
    set_state,
    story_continue_from_summary,
    story_continue_from_text,
    story_start,
    temp_loop,
    temp_start,
):
    _s = state

    prompt = mo.ui.text_area(value="", placeholder="Enter your prompt here…", label="", rows=3)
    send = mo.ui.button(label="Send", kind="primary")

    mo.Html('<div class="footer"><div class="composer">')
    mo.display(prompt)
    mo.display(send)
    mo.Html("</div></div>")

    if send.value:
        _user_text = prompt.value.strip()
        if not _user_text:
            mo.callout("Enter a prompt.", kind="warning")
        else:
            try:
                if not _s["premise"]:
                    _start = story_start(model.value, _user_text, temperature=temp_start.value)
                    _s["premise"] = _start.premise
                    _s["opening"] = _start.opening_paragraph
                    _s["paragraphs"] = []
                    _s["summaries"] = []
                    _s["pending"] = None
                    set_state(_s)
                else:
                    _premise = _s["premise"]

                    if prefer_summary.value and _s["summaries"]:
                        _checkpoint = StoryAnalysis.model_validate(_s["summaries"][-1])
                        _cont = story_continue_from_summary(
                            model.value, _premise, _checkpoint, temperature=temp_loop.value
                        )
                    else:
                        _last_para_for_continue = _s["paragraphs"][-1] if _s["paragraphs"] else ""
                        _story_text_for_continue = (_s["opening"] or "") + (
                            ("\n\n" + _last_para_for_continue) if _last_para_for_continue else ""
                        )
                        _cont = story_continue_from_text(
                            model.value,
                            _premise,
                            _story_text_for_continue,
                            temperature=temp_loop.value,
                        )

                    _s["pending"] = _cont.next_paragraph
                    set_state(_s)

            except Exception as _e:
                mo.callout(str(_e), kind="danger")

    return prompt, send


# ----------------------------
# Pending paragraph editor + accept/checkpoint
# ----------------------------
@app.cell
def _(mo, model, state, set_state, story_analysis, temp_loop):
    _s = state
    if not _s["premise"] or not _s["pending"]:
        mo.stop(True)

    editor = mo.ui.text_area(value=_s["pending"], label="Edit next paragraph", rows=6)
    accept = mo.ui.button(label="Accept", kind="success")
    checkpoint = mo.ui.button(label="Checkpoint (analyze)", kind="neutral")

    mo.Html('<div class="stage">')
    mo.Html('<div class="card"><h3>Pending paragraph</h3>')
    mo.display(editor)
    mo.Html("</div>")
    mo.hstack([accept, checkpoint])
    mo.Html("</div>")

    if accept.value:
        _s["paragraphs"].append(editor.value.strip())
        _s["pending"] = None
        set_state(_s)

    if checkpoint.value:
        try:
            _premise_for_checkpoint = _s["premise"]
            _last_para_for_checkpoint = _s["paragraphs"][-1] if _s["paragraphs"] else ""
            _story_text_for_checkpoint = (_s["opening"] or "") + (
                ("\n\n" + _last_para_for_checkpoint) if _last_para_for_checkpoint else ""
            )
            _analysis = story_analysis(
                model.value,
                _premise_for_checkpoint,
                _story_text_for_checkpoint,
                temperature=temp_loop.value,
            )
            _s["summaries"].append(_analysis.model_dump())
            set_state(_s)
            mo.callout("Checkpoint saved.", kind="success")
        except Exception as _e:
            mo.callout(str(_e), kind="danger")

    return

# ----------------------------
# Export / Import
# ----------------------------
@app.cell
def _(json, mo, state, set_state):
    _s = state
    if not _s["premise"]:
        mo.stop(True)

    export_btn = mo.ui.button(label="Export state JSON")
    import_box = mo.ui.text_area(
        value="",
        placeholder="Paste JSON to import (overwrites).",
        rows=6,
        label="",
    )
    import_btn = mo.ui.button(label="Import JSON")

    panel = mo.accordion(
        {
            "State": mo.vstack(
                [
                    mo.hstack([export_btn]),
                    mo.Html("<hr class='sep'/>"),
                    import_box,
                    mo.hstack([import_btn]),
                ],
                gap=10,
            )
        }
    )

    mo.Html('<div class="stage"><div class="card">')
    mo.display(panel)
    mo.Html("</div></div>")

    if export_btn.value:
        _dumped = json.dumps(_s, indent=2)
        mo.download(_dumped, filename="story_state.json")
        mo.callout("Export ready.", kind="success")

    if import_btn.value and import_box.value.strip():
        try:
            _loaded = json.loads(import_box.value)
            for _k in ["premise", "opening", "paragraphs", "summaries", "pending"]:
                if _k not in _loaded:
                    raise ValueError(f"Missing key: {_k}")
            set_state(_loaded)
            mo.callout("Imported.", kind="success")
        except Exception as _e:
            mo.callout(str(_e), kind="danger")

    return


if __name__ == "__main__":
    app.run()