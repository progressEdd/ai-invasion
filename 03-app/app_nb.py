import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import html
    import json
    import os
    import marimo
    from typing import Any, List

    from pydantic import BaseModel, Field
    return Any, BaseModel, Field, List


@app.cell
def _(BaseModel, Field, List):
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
                        f"Checkpoint summary:\n{summary.model_dump_json(indent=2)}\n\n"
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
        _summary_json = summary.model_dump_json(indent=2)
        _out = _chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{_doc}\n\n"
                        f"Premise:\n{premise}\n\n"
                        f"Checkpoint summary (JSON):\n{_summary_json}\n\n"
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
    return


@app.cell
def _():
    def _(Any, html, marimo, state):
        _s = state()

        _title = "LLM"
        _subtitle = "LLM Learns Lore Mostly"

        _blocks: list[Any] = []
        _blocks.append(
            marimo.Html(
                f"""
    <div class="stage">
      <div class="h1">{html.escape(_title)}</div>
      <div class="sub">{html.escape(_subtitle)}</div>
    </div>
    """
            )
        )

        if not _s["premise"]:
            marimo.vstack(_blocks)
            return

        _blocks.append(
            marimo.Html(
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
                marimo.Html(
                    f"""
    <div class="stage">
      <div class="card"><div class="pre">{html.escape(_p)}</div></div>
    </div>
    """
                )
            )

        marimo.vstack(_blocks)
        return

    return


@app.cell
def _():
    def _(marimo, model, prefer_summary, temp_loop, temp_start):
        _row = marimo.hstack(
            [
                marimo.Html('<span class="pill">⚙ Settings</span>'),
                model,
                temp_start,
                temp_loop,
                prefer_summary,
            ],
            gap=12,
            align="center",
        )

        marimo.Html(f'<div class="stage"><div class="card">{_row}</div></div>')
        return

    return


@app.cell
def _():
    def _(marimo):
        prompt = marimo.ui.text_area(
            value="",
            placeholder="Enter your prompt here…",
            label="",
            rows=3,
        )
        send = marimo.ui.run_button(label="Send", kind="success")

        _composer = marimo.hstack([prompt, send], gap=10, align="end", widths=[1, 0])
        marimo.Html(f'<div class="footer"><div class="composer">{_composer}</div></div>')

        return prompt, send

    return


@app.cell
def _():
    def _(marimo):
        prompt = marimo.ui.text_area(
            value="",
            placeholder="Enter your prompt here…",
            label="",
            rows=3,
        )
        send = marimo.ui.run_button(label="Send", kind="success")

        _composer = marimo.hstack([prompt, send], gap=10, align="end", widths=[1, 0])
        marimo.Html(f'<div class="footer"><div class="composer">{_composer}</div></div>')

        return prompt, send

    return


@app.cell
def _():
    def _(
        StoryAnalysis,
        marimo,
        model,
        prefer_summary,
        prompt,
        send,
        set_state,
        state,
        story_continue_from_summary,
        story_continue_from_text,
        story_start,
        temp_loop,
        temp_start,
    ):
        marimo.stop(not send.value)

        _user_text = prompt.value.strip()
        if not _user_text:
            marimo.callout("Enter a prompt.", kind="warning")
            return

        try:
            _s = state()

            if not _s["premise"]:
                _start = story_start(model.value, _user_text, temperature=temp_start.value)
                set_state(
                    {
                        "premise": _start.premise,
                        "opening": _start.opening_paragraph,
                        "paragraphs": [],
                        "summaries": [],
                        "pending": None,
                    }
                )
                return

            _premise = _s["premise"]

            if prefer_summary.value and _s["summaries"]:
                _checkpoint = StoryAnalysis.model_validate(_s["summaries"][-1])
                _cont = story_continue_from_summary(
                    model.value, _premise, _checkpoint, temperature=temp_loop.value
                )
            else:
                _last_para = _s["paragraphs"][-1] if _s["paragraphs"] else ""
                _story_text = (_s["opening"] or "") + (("\n\n" + _last_para) if _last_para else "")
                _cont = story_continue_from_text(
                    model.value,
                    _premise,
                    _story_text,
                    temperature=temp_loop.value,
                )

            set_state(lambda s: {**s, "pending": _cont.next_paragraph})

        except Exception as _e:
            marimo.callout(str(_e), kind="danger")

        return

    return


@app.cell
def _():
    def _(marimo, state):
        _s = state()
        marimo.stop((not _s["premise"]) or (not _s["pending"]))

        editor = marimo.ui.text_area(value=_s["pending"], label="Edit next paragraph", rows=6)
        accept = marimo.ui.run_button(label="Accept", kind="success")
        checkpoint = marimo.ui.run_button(label="Checkpoint (analyze)", kind="neutral")

        _controls = marimo.hstack([accept, checkpoint], gap=10, align="center")
        _panel = marimo.vstack([editor, _controls], gap=10)

        marimo.Html(
            f"""
    <div class="stage">
      <div class="card">
        <h3>Pending paragraph</h3>
        {_panel}
      </div>
    </div>
    """
        )

        return editor, accept, checkpoint

    return


@app.cell
def _():
    def _(
        marimo,
        model,
        set_state,
        state,
        story_analysis,
        temp_loop,
        editor,
        accept,
        checkpoint,
    ):
        _s = state()
        marimo.stop((not _s["premise"]) or (not _s["pending"]))

        if accept.value:
            _para = editor.value.strip()
            if not _para:
                marimo.callout("Paragraph is empty.", kind="warning")
                return

            set_state(lambda s: {**s, "paragraphs": s["paragraphs"] + [_para], "pending": None})
            return

        if checkpoint.value:
            try:
                _premise = _s["premise"]
                _last_para = _s["paragraphs"][-1] if _s["paragraphs"] else ""
                _story_text = (_s["opening"] or "") + (("\n\n" + _last_para) if _last_para else "")

                _analysis = story_analysis(
                    model.value,
                    _premise,
                    _story_text,
                    temperature=temp_loop.value,
                )

                set_state(lambda s: {**s, "summaries": s["summaries"] + [_analysis.model_dump()]})
                marimo.callout("Checkpoint saved.", kind="success")

            except Exception as _e:
                marimo.callout(str(_e), kind="danger")

        return

    return


@app.cell
def _():
    def _(json, marimo, state):
        _s = state()
        marimo.stop(not _s["premise"])

        export_btn = marimo.ui.run_button(label="Export state JSON", kind="neutral")
        import_box = marimo.ui.text_area(
            value="",
            placeholder="Paste JSON to import (overwrites).",
            rows=6,
            label="",
        )
        import_btn = marimo.ui.run_button(label="Import JSON", kind="warn")

        panel = marimo.accordion(
            {
                "State": marimo.vstack(
                    [
                        marimo.hstack([export_btn]),
                        marimo.Html("<hr class='sep'/>"),
                        import_box,
                        marimo.hstack([import_btn]),
                    ],
                    gap=10,
                )
            }
        )

        marimo.Html(f"<div class='stage'>{panel}</div>")

        return export_btn, import_box, import_btn

    return


@app.cell
def _():
    def _(json, marimo, export_btn, import_box, import_btn, set_state, state):
        _s = state()
        marimo.stop(not _s["premise"])

        if export_btn.value:
            _dumped = json.dumps(_s, indent=2)
            marimo.download(_dumped, filename="story_state.json")
            marimo.callout("Export ready.", kind="success")
            return

        if import_btn.value and import_box.value.strip():
            try:
                _loaded = json.loads(import_box.value)
                for _k in ["premise", "opening", "paragraphs", "summaries", "pending"]:
                    if _k not in _loaded:
                        raise ValueError(f"Missing key: {_k}")
                set_state(_loaded)
                marimo.callout("Imported.", kind="success")
            except Exception as _e:
                marimo.callout(str(_e), kind="danger")

        return

    return


@app.cell
def _():
    def _(json, marimo, export_btn, import_box, import_btn, set_state, state):
        _s = state()
        marimo.stop(not _s["premise"])

        if export_btn.value:
            _dumped = json.dumps(_s, indent=2)
            marimo.download(_dumped, filename="story_state.json")
            marimo.callout("Export ready.", kind="success")
            return

        if import_btn.value and import_box.value.strip():
            try:
                _loaded = json.loads(import_box.value)
                for _k in ["premise", "opening", "paragraphs", "summaries", "pending"]:
                    if _k not in _loaded:
                        raise ValueError(f"Missing key: {_k}")
                set_state(_loaded)
                marimo.callout("Imported.", kind="success")
            except Exception as _e:
                marimo.callout(str(_e), kind="danger")

        return

    return


if __name__ == "__main__":
    app.run()
