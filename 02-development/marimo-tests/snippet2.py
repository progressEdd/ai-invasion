premise_md = mo.md(f"**Premise:** {get_premise() or ''}").style(
    {"margin": "0", "padding": "0", "lineHeight": "0"}
)

controls_row = mo.hstack(
    [generate_next_btn, append_btn, discard_btn],
    gap=10,
).style(
    {
        "width": "100%",
        "flexWrap": "wrap",
        "justifyContent": "flex-start",
        "alignItems": "center",
        "margin": "0",
        "padding": "0",
        "marginTop": "-8px",  # pull buttons up (tune -4px .. -10px)
    }
)

header = mo.vstack([premise_md, controls_row], gap=1).style(
    {"width": "100%", "margin": "0", "padding": "0"}
)

story_body = mo.ui.text_area(
    value=get_story_text(),
    on_change=set_story_text,
    label="Story",
    rows=25,          # rows becomes less important, try to sync up the row to the min height as that will scale the text box to the whitespace to the next element
    full_width=True,
).style({"margin": "0", "padding": "1", "flex": "1 1 auto", "minHeight": "25"})

draft_editor = mo.ui.text_area(
    value=get_draft_next(),
    on_change=set_draft_next,
    label="Next paragraph (draft)",
    rows=13,
    full_width=True,
).style({"margin": "0", "padding": "0"})

_analysis_obj = get_analysis()
analysis_preview = (
    mo.md(f"```json\n{_analysis_obj.model_dump_json(indent=2)}\n```").style(
        {"margin": "0", "padding": "0", "lineHeight": "1.1"}
    )
    if _analysis_obj is not None
    else mo.md("")
)

bottom_block = mo.vstack(
    [
        draft_editor if (get_draft_next() or "").strip() else mo.md(""),
        analysis_preview if _analysis_obj is not None else mo.md(""),
    ],
    gap=0,  # tight spacing between draft + analysis
).style({"margin": "0", "padding": "0", "width": "100%"})

mo.vstack(
    [
        header,
        story_body,
        bottom_block,
    ],
    gap=0,  # tighter overall spacing
).style(
    {
        "width": "100%",
        "height": "100vh",   # fill viewport
        "display": "flex",
        "flexDirection": "column",
        "margin": "0",
        "padding": "0",
        "minHeight": "0",
    }
)