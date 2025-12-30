import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns", layout_file="layouts/app.grid.json")


@app.cell
def _():
    import marimo as mo

    llm_backend = mo.ui.radio(
        options=["llamacpp", "lmstudio", "ollama", "openai", "azure"],
        value="llamacpp",
        label="LLM backend",
    )

    llm_backend  # ok to display it
    return llm_backend, mo


@app.cell
def _(mo):
    llamacpp_form = (
        mo.md("**llama.cpp**\n\nBase URL: {base_url}")
        .batch(base_url=mo.ui.text(value="http://localhost:8080/v1", label="base_url"))
        .form(label="llama.cpp settings", submit_button_label="Apply")
    )

    lmstudio_form = (
        mo.md("**LM Studio**\n\nBase URL: {base_url}")
        .batch(base_url=mo.ui.text(value="http://localhost:1234/v1", label="base_url"))
        .form(label="LM Studio settings", submit_button_label="Apply")
    )

    ollama_form = (
        mo.md("**Ollama**\n\nHost: {host}")
        .batch(host=mo.ui.text(value="http://localhost:11434", label="Host"))
        .form(label="Ollama settings", submit_button_label="Apply")
    )

    openai_form = (
        mo.md("**OpenAI**\n\nAPI key: {api_key}")
        .batch(api_key=mo.ui.text(kind="password", label="OPENAI_API_KEY"))
        .form(label="OpenAI settings", submit_button_label="Apply")
    )

    azure_form = (
        mo.md(
            "**Azure OpenAI**\n\n"
            "Endpoint: {endpoint}\n\n"
            "API version: {api_version}\n\n"
            "API key: {api_key}\n\n"
            "Deployments (comma-separated): {deployments}"
        )
        .batch(
            endpoint=mo.ui.text(kind="url", label="AZURE_OPENAI_ENDPOINT"),
            api_version=mo.ui.text(value="2024-08-01-preview", label="api_version"),
            api_key=mo.ui.text(kind="password", label="AZURE_OPENAI_API_KEY"),
            deployments=mo.ui.text(label="deployments"),
        )
        .form(label="Azure settings", submit_button_label="Apply")
    )

    refresh_models = mo.ui.button(label="Refresh models")
    return azure_form, llamacpp_form, lmstudio_form, ollama_form, openai_form


@app.cell
def _(
    azure_form,
    llamacpp_form,
    llm_backend,
    lmstudio_form,
    mo,
    ollama_form,
    openai_form,
):
    # --- CELL 3: build client + discover models ---
    from typing import Any, Iterable
    import os

    from openai import OpenAI, AzureOpenAI

    def _normalize_base_url(url: str) -> str:
        url = (url or "").strip().rstrip("/")
        if not url:
            return url
        return url if url.endswith("/v1") else f"{url}/v1"

    def _extract_ids(items: Iterable[Any]) -> list[str]:
        out: list[str] = []
        for it in items or []:
            mid = getattr(it, "id", None)
            if mid is None and isinstance(it, dict):
                mid = it.get("id") or it.get("model")
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
        # stable order, unique
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _try_client_models_list(client: Any) -> tuple[list[str], str | None]:
        try:
            resp = client.models.list()
            data = getattr(resp, "data", None)
            if data is None and isinstance(resp, dict):
                data = resp.get("data")
            return _extract_ids(data), None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    def _try_ollama_list_models(ollama_host_url: str) -> tuple[list[str], str | None]:
        """
        Uses `ollama.list().model_dump()` and returns the `model` fields (your requested behavior).
        """
        try:
            import ollama  # type: ignore
        except Exception as e:
            return [], f"ollama import failed: {type(e).__name__}: {e}"

        host = (ollama_host_url or "").strip().rstrip("/")
        # `OLLAMA_HOST` expects scheme+host+port (no /v1)
        prev = os.environ.get("OLLAMA_HOST")
        if host:
            os.environ["OLLAMA_HOST"] = host
        try:
            payload = ollama.list().model_dump()
            models = []
            for m in (payload or {}).get("models", []):
                name = (m or {}).get("model")
                if isinstance(name, str) and name.strip():
                    models.append(name.strip())
            # unique + keep order
            seen = set()
            uniq = []
            for x in models:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq, None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"
        finally:
            if prev is None:
                os.environ.pop("OLLAMA_HOST", None)
            else:
                os.environ["OLLAMA_HOST"] = prev

    llm_client = None
    available_models: list[str] = []
    model_source_error: str | None = None

    backend = llm_backend.value

    if backend == "llamacpp":
        cfg = llamacpp_form.value or {}
        base_url = _normalize_base_url(cfg.get("base_url") or "http://localhost:8080/v1")
        llm_client = OpenAI(base_url=base_url, api_key="sk-no-key-required")
        available_models, model_source_error = _try_client_models_list(llm_client)

    elif backend == "lmstudio":
        cfg = lmstudio_form.value or {}
        base_url = _normalize_base_url(cfg.get("base_url") or "http://localhost:1234/v1")
        llm_client = OpenAI(base_url=base_url, api_key="lm-studio")  # key required by SDK, ignored by LM Studio
        available_models, model_source_error = _try_client_models_list(llm_client)

    elif backend == "ollama":
        cfg = ollama_form.value or {}
        host = (cfg.get("host") or "").strip() or "http://localhost:11434"
        base_url = _normalize_base_url(host)  # host -> host/v1
        llm_client = OpenAI(base_url=base_url, api_key="ollama")  # required, but unused

        # Your requested listing method:
        available_models, model_source_error = _try_ollama_list_models(host)

    elif backend == "openai":
        cfg = openai_form.value or {}
        api_key = (cfg.get("key") or "").strip()
        llm_client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        if api_key or os.environ.get("OPENAI_API_KEY"):
            available_models, model_source_error = _try_client_models_list(llm_client)
        else:
            available_models, model_source_error = [], "No API key provided."

    elif backend == "azure":
        cfg = azure_form.value or {}
        api_key = (cfg.get("key") or "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "")
        endpoint = (cfg.get("endpoint") or "").strip() or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_version = (cfg.get("api_version") or "").strip() or "2024-08-01-preview"

        llm_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

        # Azure “model” is typically the *deployment name*, so use the user-provided list:
        deployments_raw = (cfg.get("deployments") or "").strip()
        available_models = [x.strip() for x in deployments_raw.split(",") if x.strip()]
        model_source_error = None if available_models else "No deployments provided (enter comma-separated deployment names)."

    mo.vstack(
        [
            mo.md(f"**Backend:** `{backend}`"),
            mo.md(f"**Discovered models:** {len(available_models)}"),
            mo.md(f"**Model discovery error:** `{model_source_error}`") if model_source_error else mo.md(""),
        ]
    )
    return available_models, backend, llm_client


@app.cell
def _(mo):
    # CELL 4A
    MODEL_PLACEHOLDER = "— Select a model —"

    get_model, set_model = mo.state(MODEL_PLACEHOLDER)
    get_model_tick, set_model_tick = mo.state(0)

    return (
        MODEL_PLACEHOLDER,
        get_model,
        get_model_tick,
        set_model,
        set_model_tick,
    )


@app.cell
def _(
    MODEL_PLACEHOLDER,
    available_models: list[str],
    get_model,
    mo,
    set_model,
    set_model_tick,
):
    # CELL 4B: create selector UI (no .value access)
    def _on_model_change(v: str):
        set_model(v)
        set_model_tick(lambda t: t + 1)  # user-driven signal only

    current = (get_model() or MODEL_PLACEHOLDER)

    if available_models:
        options = [MODEL_PLACEHOLDER] + list(available_models)
        if current not in options:
            set_model(MODEL_PLACEHOLDER)  # keep blank-ish selection; don't auto-pick a model
            current = MODEL_PLACEHOLDER

        selector_ui = mo.ui.dropdown(
            options=options,
            value=current,          # must be in options
            label="Model",
            searchable=True,
            on_change=_on_model_change,
        )
    else:
        selector_ui = mo.ui.text(
            label="Custom model/deployment",
            placeholder="e.g., gemma3:12b-it-q8_0 | qwen3:30b-a3b-instruct-2507-q4_K_M | gpt-4o-mini",
            value="" if current == MODEL_PLACEHOLDER else current,
            on_change=_on_model_change,
        )

    mo.vstack([selector_ui])
    return


@app.cell
def _(MODEL_PLACEHOLDER, get_model):
    # CELL 4C: derive llm_model (safe to use anywhere)

    v = (get_model() or "").strip()
    llm_model = "" if (v == MODEL_PLACEHOLDER) else v
    llm_model
    return (llm_model,)


@app.cell
def _(backend, get_model_tick, llm_client, llm_model, mo):
    # CELL 4D Warmup (progress bar + live elapsed, without spamming increments)
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    get_last_warm_tick, set_last_warm_tick = mo.state(0)

    mo.stop(not llm_model)
    mo.stop(backend not in {"ollama", "llamacpp", "lmstudio"})
    mo.stop(llm_client is None)

    tick = get_model_tick()
    mo.stop(tick <= get_last_warm_tick())

    def warm_up_model(client, model: str) -> tuple[bool, str]:
        t0 = time.time()
        try:
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helper."},
                    {"role": "user", "content": "hi"},
                ],
                temperature=0,
                max_tokens=1,
            )
            return True, f"{time.time() - t0:.2f}s"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    start = time.time()

    with mo.status.progress_bar(
        total=1,
        title="Warming up model",
        subtitle=f"{backend} · {llm_model} · 0.0s elapsed",
        show_eta=False,
        show_rate=False,
        remove_on_exit=True,
    ) as bar:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(warm_up_model, llm_client, llm_model)

            while True:
                try:
                    ok, info = fut.result(timeout=0.2)
                    break
                except FuturesTimeout:
                    elapsed = time.time() - start
                    # IMPORTANT: don't increment progress during polling
                    bar.update(increment=0, subtitle=f"{backend} · {llm_model} · {elapsed:.1f}s elapsed")

        total_elapsed = time.time() - start
        bar.update(
            increment=1,
            title="Warm-up complete" if ok else "Warm-up failed",
            subtitle=f"{info} (total {total_elapsed:.2f}s)",
        )

    set_last_warm_tick(tick)

    mo.md(
        f"**Warmup ({backend} / {llm_model}):** {'✅' if ok else '❌'} {info} "
        f"(total: {total_elapsed:.2f}s)"
    )

    return


@app.cell
def _():
    from typing import Type, TypeVar, Sequence, Dict

    T = TypeVar("T")

    def parse_structured(
        client,
        *,
        model: str,
        schema: Type[T],
        user_content: str,
        system_content: str = "You are a helpful assistant. Follow the response model docstring.",
        temperature: float = 0.2,
        extra_messages: Sequence[Dict[str, str]] = (),
    ) -> T:
        messages = [{"role": "system", "content": system_content}]
        messages += list(extra_messages)
        messages += [{"role": "user", "content": user_content}]

        parsed = client.beta.chat.completions.parse(
            model=model,
            response_format=schema,
            messages=messages,
            temperature=temperature,
        ).choices[0].message.parsed

        return schema.model_validate(parsed)
    return (parse_structured,)


@app.cell
def _():
    from typing import Optional, List
    from pydantic import BaseModel, Field


    class StoryStart(BaseModel):
        """
        You are a sharp, imaginative fiction writer.

        Task:
        - Produce (1) a concise, compelling _story _premise and (2) the opening paragraph that launches the _story.

        Rules:
        - If the user provides ideas, weave them in organically (don’t just repeat them).
        - If the user provides no ideas, invent something fresh with a surprising combination of
          genre, setting, protagonist, conflict, and twist.
        - premise: 2–5 sentences, stakes + hook, no spoilers.
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
        - You will be given the _story _premise and the _story-so-far (either the opening paragraph + latest paragraph,
          or a compact analysis summary). Use them to preserve continuity.

        Rules:
        - Output exactly one paragraph of _story prose (no headings, no bullets, no analysis).
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
        You are a _story analyst. Produce a succinct “_story-So-Far” handoff so another model can write
        the next paragraph without breaking continuity. Do not write new _story prose.

        Inputs:
        - You will be given the _story _premise and the _story text so far (opening paragraph + one or more continuation paragraphs).

        Rules:
        - Extract ground truth only from the provided text/_premise. No inventions.
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
def _(mo):
    # --- CELL: _story state (define once; if you already have any of these, don't redefine) ---

    get_start, set_start = mo.state(None)            # storyStart | None
    get_start_err, set_start_err = mo.state("")      # str
    get_prompt, set_prompt = mo.state("")  # starting premise text

    get_premise, set_premise = mo.state("")          # str
    get_opening, set_opening = mo.state("")          # str

    get_story_text, set_story_text = mo.state("")    # approved full _story text
    get_draft_next, set_draft_next = mo.state("")    # draft paragraph being edited
    get_analysis, set_analysis = mo.state(None)      # _storyAnalysis | None
    return (
        get_analysis,
        get_draft_next,
        get_opening,
        get_premise,
        get_prompt,
        get_start_err,
        get_story_text,
        set_analysis,
        set_draft_next,
        set_opening,
        set_premise,
        set_prompt,
        set_start,
        set_start_err,
        set_story_text,
    )


@app.cell
def _(get_prompt, mo, set_prompt):
    prompt_form = mo.ui.text_area(
        value=get_prompt(),
        on_change=set_prompt,
        label="Starting premise",
        placeholder="Enter an idea for a story…",
        rows=4,
        full_width=True,
    )

    lucky_btn = mo.ui.run_button(label="I'm feeling lucky")
    start_btn = mo.ui.run_button(label="Start")

    mo.vstack(
        [
            prompt_form,
            mo.hstack(
                [lucky_btn, mo.vstack([start_btn]).style({"marginLeft": "auto"})],
                gap=12,
            ).style({"width": "100%", "alignItems": "center"}),
        ],
        gap=1.5,
    )
    return lucky_btn, start_btn


@app.cell
def _(lucky_btn, mo, set_prompt):
    mo.stop(not lucky_btn.value)
    set_prompt("I'm feeling lucky — suggest me a premise")
    mo.md("")
    return


@app.cell
def _(user_prompt):
    user_prompt
    return


@app.cell
def _(
    StoryStart,
    get_prompt,
    get_start_err,
    llm_client,
    llm_model,
    mo,
    parse_structured,
    set_analysis,
    set_draft_next,
    set_opening,
    set_premise,
    set_start,
    set_start_err,
    set_story_text,
    start_btn,
):
    mo.stop(not start_btn.value)
    user_prompt = "This is the premise given by the user: " + (get_prompt() or "").strip()

    status_ui = mo.md("")
    if user_prompt:
        try:
            set_start_err("")
            _start = parse_structured(
                llm_client,
                model=llm_model,
                schema=StoryStart,
                user_content=user_prompt,
                temperature=2,
            )

            set_start(_start)
            set_premise((_start.premise or "").strip())
            set_opening((_start.opening_paragraph or "").strip())

            # seed _story body with opening
            set_story_text((_start.opening_paragraph or "").strip())

            # reset downstream
            set_draft_next("")
            set_analysis(None)

            status_ui = mo.md(f"Generated start, prompt sent to model: \n\n {user_prompt}")
        except Exception as e:
            set_start_err(f"{type(e).__name__}: {e}")
            status_ui = mo.md(f"**Error:** `{get_start_err()}`")

    status_ui
    return (user_prompt,)


@app.cell
def _(mo):
    generate_next_btn = mo.ui.run_button(label="Generate next paragraph")
    append_btn = mo.ui.run_button(label="Append to story")
    discard_btn = mo.ui.run_button(label="Discard draft")
    return append_btn, discard_btn, generate_next_btn


@app.cell
def _(
    append_btn,
    discard_btn,
    generate_next_btn,
    get_analysis,
    get_draft_next,
    get_premise,
    get_story_text,
    mo,
    set_draft_next,
    set_story_text,
):
    premise_md = mo.md(f"**Premise:** {get_premise() or ''}")
    premise_md_css = premise_md.style({"margin": "0", "padding": "0", "lineHeight": "0"})

    controls_row = mo.hstack([generate_next_btn, append_btn, discard_btn], gap=10)
    controls_row_css = controls_row.style(
        {
            "width": "100%",
            "flexWrap": "wrap",
            "justifyContent": "flex-start",
            "alignItems": "center",
            "margin": "0",
            "padding": "0",
            "marginTop": "-8px",
        }
    )

    header = mo.vstack([premise_md_css, controls_row_css], gap=1)
    header_css = header.style({"width": "100%", "margin": "0", "padding": "0"})


    # widgets stay as the main variables
    story_body = mo.ui.text_area(
        value=get_story_text(),
        on_change=set_story_text,
        label="Story",
        rows=25,
        full_width=True,
    )
    story_body_css = story_body.style(
        {"margin": "0", "padding": "1", "flex": "1 1 auto", "minHeight": "25"}
    )

    draft_editor = mo.ui.text_area(
        value=get_draft_next(),
        on_change=set_draft_next,
        label="Next paragraph (draft)",
        rows=13,
        full_width=True,
    )
    draft_editor_css = draft_editor.style({"margin": "0", "padding": "0"})


    _analysis_obj = get_analysis()
    analysis_preview = (
        mo.md(f"```json\n{_analysis_obj.model_dump_json(indent=2)}\n```")
        if _analysis_obj is not None
        else mo.md("")
    )
    analysis_preview_css = analysis_preview.style({"margin": "0", "padding": "0", "lineHeight": "1.1"})

    bottom_block = mo.vstack(
        [
            draft_editor_css if (get_draft_next() or "").strip() else mo.md(""),
            analysis_preview_css if _analysis_obj is not None else mo.md(""),
        ],
        gap=0,
    )
    bottom_block_css = bottom_block.style({"margin": "0", "padding": "0", "width": "100%"})


    page = mo.vstack([header_css, story_body_css, bottom_block_css], gap=0)
    page_css = page.style(
        {
            "width": "100%",
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "margin": "0",
            "padding": "0",
            "minHeight": "0",
        }
    )

    page_css
    return draft_editor, story_body


@app.cell
def _(StoryAnalysis, StoryContinue, parse_structured):
    from dataclasses import dataclass
    # from typing import Optional

    @dataclass
    class CycleResult:
        draft_next: str
        analysis: StoryAnalysis

    def run_cycle(
        client,
        *,
        model: str,
        premise: str,
        story_text: str,              # approved full story so far
        temperature_continue: float = 0.2,
        temperature_analyze: float = 0.2,
    ) -> CycleResult:
        # 3) Analyze (premise + approved story text)
        analysis_input = f"Premise:\n{premise}\n\nStory text:\n{story_text}"
        analysis = parse_structured(
            client,
            model=model,
            schema=StoryAnalysis,
            user_content=analysis_input,
            temperature=temperature_analyze,
        )

        # 4) Continue again (use analysis summary)
        cont2_input = (
            f"Premise:\n{premise}\n\n"
            f"Story analysis summary:\n{analysis.model_dump_json(indent=2)}\n\n"
            "Write the next paragraph."
        )
        cont2 = parse_structured(
            client,
            model=model,
            schema=StoryContinue,
            user_content=cont2_input,
            temperature=temperature_continue,
        )

        return CycleResult(draft_next=(cont2.next_paragraph or "").strip(), analysis=analysis)
    return (run_cycle,)


@app.cell
def _(
    generate_next_btn,
    get_opening,
    get_premise,
    get_story_text,
    llm_client,
    llm_model,
    mo,
    run_cycle,
    set_analysis,
    set_draft_next,
    set_story_text,
):
    # run when clicked
    mo.stop(not generate_next_btn.value)  # run_button pattern :contentReference[oaicite:2]{index=2}

    _premise = (get_premise() or "").strip()
    mo.stop(not _premise)

    _story = (get_story_text() or "").strip()
    if not _story:
        _opening = (get_opening() or "").strip()
        if _opening:
            _story = _opening
            set_story_text(_opening)

    # If you want to use your orchestration:
    res = run_cycle(
        llm_client,
        model=llm_model,
        premise=_premise,
        story_text=_story,
    )

    set_analysis(res.analysis)
    set_draft_next(res.draft_next)

    mo.md("")
    return


@app.cell
def _(story_body):
    story_body.value
    return


@app.cell
def _(
    append_btn,
    discard_btn,
    draft_editor,
    mo,
    set_draft_next,
    set_story_text,
    story_body,
):

    if discard_btn.value:
        set_draft_next("")

    mo.stop(not append_btn.value)

    _current_story = (story_body.value or "").strip()
    _current_draft = (draft_editor.value or "").strip()

    if _current_draft:
        set_story_text(f"{_current_story}\n\n{_current_draft}" if _current_story else _current_draft)
        set_draft_next("")

    mo.md("")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
