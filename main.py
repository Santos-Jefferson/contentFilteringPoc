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


def _load_ui_categories(config_path: str) -> tuple[list[str], list[str]]:
    try:
        config = load_term_config(config_path)
        return get_category_names(config), get_default_categories(config)
    except Exception:
        return [], []


def _load_ui_scene_categories(config_path: str) -> tuple[list[str], list[str]]:
    try:
        config = load_term_config(config_path)
        return get_scene_category_names(config), get_default_scene_categories(config)
    except Exception:
        return [], []


def run_pipeline(
    video_url: str,
    config_path: str,
    selected_categories: list[str],
    selected_scene_categories: list[str],
    mute_scene_audio: bool,
    blur_scene_video: bool,
    blur_strength: float,
    model_size: str,
    mute_padding_seconds: float,
):
    try:
        output_video, report_json, counts_text = process_video_url(
            video_url=video_url,
            selected_categories=selected_categories,
            selected_scene_categories=selected_scene_categories,
            config_path=config_path,
            model_size=model_size,
            mute_padding_seconds=mute_padding_seconds,
            mute_scene_audio=mute_scene_audio,
            blur_scene_video=blur_scene_video,
            blur_strength=blur_strength,
            artifacts_root="artifacts",
        )
        return output_video, report_json, counts_text
    except Exception as exc:
        debug_trace = traceback.format_exc(limit=3)
        return None, f"Error: {exc}\n\n{debug_trace}", "{}"


def build_ui() -> gr.Blocks:
    categories, defaults = _load_ui_categories(DEFAULT_CONFIG_PATH)
    scene_categories, scene_defaults = _load_ui_scene_categories(DEFAULT_CONFIG_PATH)

    with gr.Blocks(title="Content Filtering POC") as app:
        gr.Markdown(
            "## Video Content Filtering POC\n"
            "- Language: English\n"
            "- Audio censor: mute only filtered words\n"
            "- Captions: bad terms masked with ***\n"
            "- Category-based filters loaded from config file"
        )

        with gr.Row():
            video_url = gr.Textbox(
                label="Video URL",
                placeholder="https://... or /absolute/path/video.mp4",
                lines=1,
            )
            model_size = gr.Dropdown(
                label="Whisper model size",
                choices=["tiny", "base", "small", "medium"],
                value="small",
            )

        config_path = gr.Textbox(
            label="Filter config path",
            value=DEFAULT_CONFIG_PATH,
            lines=1,
        )
        selected_categories = gr.CheckboxGroup(
            label="Filter categories",
            choices=categories,
            value=defaults,
        )
        selected_scene_categories = gr.CheckboxGroup(
            label="Scene detection categories",
            choices=scene_categories,
            value=scene_defaults,
        )
        mute_scene_audio = gr.Checkbox(
            label="Mute detected scene intervals",
            value=False,
        )
        blur_scene_video = gr.Checkbox(
            label="Apply strong blur on detected scene intervals",
            value=True,
        )
        blur_strength = gr.Slider(
            label="Scene blur strength (sigma)",
            minimum=10.0,
            maximum=60.0,
            value=40.0,
            step=1.0,
            info="Higher values produce stronger blur on scene intervals.",
        )
        mute_padding_seconds = gr.Slider(
            label="Mute padding per word (seconds)",
            minimum=0.0,
            maximum=0.25,
            value=0.08,
            step=0.01,
            info="Expands mute window before/after each filtered word.",
        )

        process_btn = gr.Button("Process video", variant="primary")

        output_video = gr.Video(label="Censored video")
        report_json = gr.Textbox(label="Run report (JSON)", lines=12)
        counts_text = gr.Textbox(label="Filtered counts by term")

        process_btn.click(
            fn=run_pipeline,
            inputs=[
                video_url,
                config_path,
                selected_categories,
                selected_scene_categories,
                mute_scene_audio,
                blur_scene_video,
                blur_strength,
                model_size,
                mute_padding_seconds,
            ],
            outputs=[output_video, report_json, counts_text],
        )

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)
