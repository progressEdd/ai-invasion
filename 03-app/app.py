import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo

    llm_backend = mo.ui.radio(
        options=["ollama", "llamacpp", "openai", "azure"],
        value="ollama",
        label="LLM backend",
    )

    llm_backend  # ok to display it
    return llm_backend, mo


@app.cell
def _(mo):
    ollama_form = (
        mo.md("**Ollama**\n\nHost: {host}")
        .batch(host=mo.ui.text(value="http://localhost:11434", label="Host"))
        .form(label="Ollama settings", submit_button_label="Apply")
    )

    llamacpp_form = (
        mo.md("**llama.cpp**\n\nBase URL: {base_url}")
        .batch(base_url=mo.ui.text(value="http://localhost:8080/v1", label="base_url"))
        .form(label="llama.cpp settings", submit_button_label="Apply")
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
    return azure_form, llamacpp_form, ollama_form, openai_form


@app.cell
def _(azure_form, llamacpp_form, llm_backend, mo, ollama_form, openai_form):
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

    if backend == "ollama":
        cfg = ollama_form.value or {}
        host = (cfg.get("host") or "").strip() or "http://localhost:11434"
        base_url = _normalize_base_url(host)  # host -> host/v1
        llm_client = OpenAI(base_url=base_url, api_key="ollama")  # required, but unused

        # Your requested listing method:
        available_models, model_source_error = _try_ollama_list_models(host)

    elif backend == "llamacpp":
        cfg = llamacpp_form.value or {}
        base_url = _normalize_base_url(cfg.get("base_url") or "http://localhost:8080/v1")
        llm_client = OpenAI(base_url=base_url, api_key="sk-no-key-required")
        available_models, model_source_error = _try_client_models_list(llm_client)

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
    return available_models, llm_client


@app.cell
def _(available_models: list[str], mo):
    # --- CELL 4A: create widgets (no .value access) ---
    custom_model = mo.ui.text(
        label="Custom model/deployment",
        placeholder="e.g., gemma3:12b-it-q8_0 | qwen3:30b-a3b-instruct-2507-q4_K_M | gpt-4o-mini",
    )

    model_dropdown = None
    if available_models:
        model_dropdown = mo.ui.dropdown(
            options=available_models,
            value=available_models[0],
            label="Model",
            searchable=True,
        )

    mo.vstack([model_dropdown] if model_dropdown is not None else [custom_model])
    return (model_dropdown,)


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
    return (StoryStart,)


@app.cell
def _(mo):
    get_start, set_start = mo.state(None)
    get_start_err, set_start_err = mo.state("")

    get_premise, set_premise = mo.state("")
    get_opening, set_opening = mo.state("")
    return (
        get_opening,
        get_premise,
        get_start,
        get_start_err,
        set_opening,
        set_premise,
        set_start,
        set_start_err,
    )


@app.cell
def _(mo):
    prompt_form = mo.ui.text_area(
        label="Staring premise",
        placeholder="Enter an idea for a story… for example: I'm feeling lucky suggest me a premise",
        rows=4,
        full_width=True,
    ).form(submit_button_label="Start")

    prompt_form
    return (prompt_form,)


@app.cell
def _(
    StoryStart,
    get_start_err,
    llm_client,
    mo,
    model_dropdown,
    parse_structured,
    prompt_form,
    set_opening,
    set_premise,
    set_start,
    set_start_err,
):
    user_prompt = (prompt_form.value or "").strip()
    status_ui = mo.md("")
    llm_model = model_dropdown.value

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
            set_premise(_start.premise or "")
            set_opening(_start.opening_paragraph or "")
            status_ui = mo.md("Generated start.")
        except Exception as e:
            set_start_err(f"{type(e).__name__}: {e}")
            status_ui = mo.md(f"**Error:** `{get_start_err()}`")

    status_ui
    return


@app.cell
def _(get_opening, get_premise, get_start, mo, set_opening, set_premise):
    start_obj = get_start()

    if start_obj is None:
        editors_ui = mo.md("")
    else:
        premise_edit = mo.ui.text_area(
            value=get_premise(),
            on_change=set_premise,
            label="Premise",
            rows=4,
            full_width=True,
        )
        opening_edit = mo.ui.text_area(
            value=get_opening(),
            on_change=set_opening,
            label="Opening paragraph",
            rows=10,
            full_width=True,
        )
        editors_ui = mo.vstack([premise_edit, opening_edit]).style(width="100%")

    editors_ui
    return


if __name__ == "__main__":
    app.run()