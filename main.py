import json
import traceback

import gradio as gr

from src.pipeline import process_video_url
from src.term_config import (
    get_category_names,
    get_default_categories,
    get_default_scene_categories,
    get_scene_category_names,
    load_term_config,
)


DEFAULT_CONFIG_PATH = "config/bad_terms_categories.json"


def _collect_scene_thresholds(scene_cfg: dict) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for key, value in scene_cfg.items():
        if isinstance(value, (int, float)) and ("threshold" in key or "interval" in key):
            thresholds[key] = float(value)
    return dict(sorted(thresholds.items()))


def _scene_group_name(category: str) -> str:
    if category.startswith("violence_"):
        return "Violence"
    lowered = category.lower()
    if any(token in lowered for token in ("nudity", "sex", "immodesty", "kissing")):
        return "Sex and Nudity"
    return "Other Scenes"


def _group_scene_categories(scene_categories: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "Violence": [],
        "Sex and Nudity": [],
        "Other Scenes": [],
    }
    for name in sorted(scene_categories):
        grouped[_scene_group_name(name)].append(name)
    return grouped


def _load_ui_snapshot(config_path: str) -> tuple[dict, str]:
    try:
        config = load_term_config(config_path)
        categories = get_category_names(config)
        defaults = get_default_categories(config)
        scene_categories = get_scene_category_names(config)
        scene_defaults = get_default_scene_categories(config)
        grouped_scene = _group_scene_categories(scene_categories)
        scene_thresholds = _collect_scene_thresholds(config.get("scene_detection", {}))
        return (
            {
                "categories": categories,
                "category_defaults": defaults,
                "scene_grouped": grouped_scene,
                "scene_defaults": scene_defaults,
                "thresholds": scene_thresholds,
            },
            "Config loaded",
        )
    except Exception as exc:
        return (
            {
                "categories": [],
                "category_defaults": [],
                "scene_grouped": {"Violence": [], "Sex and Nudity": [], "Other Scenes": []},
                "scene_defaults": [],
                "thresholds": {},
            },
            f"Config load failed: {exc}",
        )


def _apply_select_all(enabled: bool, choices: list[str], current: list[str]) -> list[str]:
    if enabled:
        return list(choices or [])
    return []


def _parse_threshold_overrides(raw_json: str) -> dict[str, float]:
    if not (raw_json or "").strip():
        return {}
    loaded = json.loads(raw_json)
    if not isinstance(loaded, dict):
        raise ValueError("Threshold overrides must be a JSON object, for example: {\"nudity_score_threshold\": 0.55}")

    parsed: dict[str, float] = {}
    for key, value in loaded.items():
        parsed[str(key)] = float(value)
    return parsed


def _merge_scene_selection(*groups: list[str]) -> list[str]:
    selected: set[str] = set()
    for group in groups:
        selected.update(group or [])
    return sorted(selected)


def _reload_ui(config_path: str):
    snapshot, status = _load_ui_snapshot(config_path)
    categories = snapshot["categories"]
    defaults = snapshot["category_defaults"]
    grouped = snapshot["scene_grouped"]
    scene_defaults = set(snapshot["scene_defaults"])
    thresholds = snapshot["thresholds"]

    return (
        gr.update(choices=categories, value=defaults),
        gr.update(choices=grouped["Violence"], value=[c for c in grouped["Violence"] if c in scene_defaults]),
        gr.update(
            choices=grouped["Sex and Nudity"],
            value=[c for c in grouped["Sex and Nudity"] if c in scene_defaults],
        ),
        gr.update(choices=grouped["Other Scenes"], value=[c for c in grouped["Other Scenes"] if c in scene_defaults]),
        json.dumps(thresholds, indent=2),
        gr.update(value=False),
        gr.update(value=False),
        gr.update(value=False),
        categories,
        grouped["Violence"],
        grouped["Sex and Nudity"],
        grouped["Other Scenes"],
        status,
    )


def run_pipeline(
    video_url: str,
    config_path: str,
    selected_categories: list[str],
    selected_scene_violence: list[str],
    selected_scene_sex_nudity: list[str],
    selected_scene_other: list[str],
    threshold_overrides_json: str,
    mute_scene_audio: bool,
    blur_scene_video: bool,
    blur_strength: float,
    model_size: str,
    mute_padding_seconds: float,
):
    try:
        selected_scene_categories = _merge_scene_selection(
            selected_scene_violence,
            selected_scene_sex_nudity,
            selected_scene_other,
        )
        threshold_overrides = _parse_threshold_overrides(threshold_overrides_json)

        output_video, report_json, counts_text, gallery_items = process_video_url(
            video_url=video_url,
            selected_categories=selected_categories,
            selected_scene_categories=selected_scene_categories,
            config_path=config_path,
            model_size=model_size,
            mute_padding_seconds=mute_padding_seconds,
            mute_scene_audio=mute_scene_audio,
            blur_scene_video=blur_scene_video,
            blur_strength=blur_strength,
            scene_threshold_overrides=threshold_overrides,
            artifacts_root="artifacts",
        )
        return output_video, report_json, counts_text, gallery_items
    except Exception as exc:
        debug_trace = traceback.format_exc(limit=4)
        return None, f"Error: {exc}\n\n{debug_trace}", "{}", []


def build_ui() -> gr.Blocks:
    snapshot, status = _load_ui_snapshot(DEFAULT_CONFIG_PATH)
    categories = snapshot["categories"]
    defaults = snapshot["category_defaults"]
    grouped = snapshot["scene_grouped"]
    scene_defaults = set(snapshot["scene_defaults"])
    thresholds_json = json.dumps(snapshot["thresholds"], indent=2)

    with gr.Blocks(title="Content Filtering POC") as app:
        audio_choices_state = gr.State(categories)
        violence_choices_state = gr.State(grouped["Violence"])
        sex_nudity_choices_state = gr.State(grouped["Sex and Nudity"])
        other_choices_state = gr.State(grouped["Other Scenes"])

        gr.Markdown("## Content Filtering\nPaste a video URL, choose categories, run, then review flagged frames.")

        video_url = gr.Textbox(
            label="Video URL",
            placeholder="https://... or /absolute/path/video.mp4",
            lines=1,
        )
        process_btn = gr.Button("Process video", variant="primary")

        with gr.Accordion("Audio Categories", open=False):
            select_all_audio = gr.Checkbox(label="Select all audio categories", value=False)
            selected_categories = gr.CheckboxGroup(
                label="Audio category checkboxes",
                choices=categories,
                value=defaults,
            )

        with gr.Accordion("Scene Categories", open=False):
            with gr.Accordion("Violence", open=False):
                select_all_violence = gr.Checkbox(label="Select all in Violence", value=False)
                selected_scene_violence = gr.CheckboxGroup(
                    label="Violence subcategories",
                    choices=grouped["Violence"],
                    value=[c for c in grouped["Violence"] if c in scene_defaults],
                )
            with gr.Accordion("Sex and Nudity", open=False):
                select_all_sex_nudity = gr.Checkbox(label="Select all in Sex and Nudity", value=False)
                selected_scene_sex_nudity = gr.CheckboxGroup(
                    label="Sex/Nudity subcategories",
                    choices=grouped["Sex and Nudity"],
                    value=[c for c in grouped["Sex and Nudity"] if c in scene_defaults],
                )
            with gr.Accordion("Other Scenes", open=False):
                select_all_other = gr.Checkbox(label="Select all in Other Scenes", value=False)
                selected_scene_other = gr.CheckboxGroup(
                    label="Other scene subcategories",
                    choices=grouped["Other Scenes"],
                    value=[c for c in grouped["Other Scenes"] if c in scene_defaults],
                )

        with gr.Accordion("Thresholds and Processing", open=False):
            with gr.Row():
                config_path = gr.Textbox(label="Config path", value=DEFAULT_CONFIG_PATH, lines=1)
                reload_btn = gr.Button("Reload from config", variant="secondary")

            threshold_overrides_json = gr.Code(
                label="Threshold overrides (JSON)",
                language="json",
                value=thresholds_json,
                lines=10,
            )
            config_status = gr.Textbox(label="Config status", value=status, interactive=False, lines=1)

            mute_scene_audio = gr.Checkbox(label="Mute detected scene intervals", value=False)
            blur_scene_video = gr.Checkbox(label="Blur detected scene intervals", value=True)
            blur_strength = gr.Slider(
                label="Scene blur strength",
                minimum=10.0,
                maximum=60.0,
                value=40.0,
                step=1.0,
            )
            model_size = gr.Dropdown(
                label="Whisper model size",
                choices=["tiny", "base", "small", "medium"],
                value="small",
            )
            mute_padding_seconds = gr.Slider(
                label="Mute padding per word (seconds)",
                minimum=0.0,
                maximum=0.25,
                value=0.08,
                step=0.01,
            )

        output_video = gr.Video(label="Censored video", width=640, height=480)
        report_json = gr.Textbox(label="Run report (JSON)", lines=16)
        counts_text = gr.Textbox(label="Filtered counts by term")
        scene_gallery = gr.Gallery(
            label="Flagged frame gallery (why this frame was filtered)",
            columns=3,
            height=420,
            object_fit="contain",
        )

        select_all_audio.change(
            fn=_apply_select_all,
            inputs=[select_all_audio, audio_choices_state, selected_categories],
            outputs=[selected_categories],
        )
        select_all_violence.change(
            fn=_apply_select_all,
            inputs=[select_all_violence, violence_choices_state, selected_scene_violence],
            outputs=[selected_scene_violence],
        )
        select_all_sex_nudity.change(
            fn=_apply_select_all,
            inputs=[select_all_sex_nudity, sex_nudity_choices_state, selected_scene_sex_nudity],
            outputs=[selected_scene_sex_nudity],
        )
        select_all_other.change(
            fn=_apply_select_all,
            inputs=[select_all_other, other_choices_state, selected_scene_other],
            outputs=[selected_scene_other],
        )

        reload_btn.click(
            fn=_reload_ui,
            inputs=[config_path],
            outputs=[
                selected_categories,
                selected_scene_violence,
                selected_scene_sex_nudity,
                selected_scene_other,
                threshold_overrides_json,
                select_all_audio,
                select_all_violence,
                select_all_sex_nudity,
                select_all_other,
                audio_choices_state,
                violence_choices_state,
                sex_nudity_choices_state,
                other_choices_state,
                config_status,
            ],
        )

        process_btn.click(
            fn=run_pipeline,
            inputs=[
                video_url,
                config_path,
                selected_categories,
                selected_scene_violence,
                selected_scene_sex_nudity,
                selected_scene_other,
                threshold_overrides_json,
                mute_scene_audio,
                blur_scene_video,
                blur_strength,
                model_size,
                mute_padding_seconds,
            ],
            outputs=[output_video, report_json, counts_text, scene_gallery],
        )

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)
