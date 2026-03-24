import json
import os
import sys
import traceback
from pathlib import Path

import gradio as gr

from src.pipeline import process_video_url
from src.term_config import (
    MANUAL_PROFILE_NAME,
    get_category_names,
    get_default_processing_profile_name,
    get_processing_profile_names,
    get_scene_category_names,
    load_term_config,
    resolve_processing_profile,
)


DEFAULT_CONFIG_PATH = "config/bad_terms_categories.json"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _supported_browser_cookie_choices() -> list[str]:
    if _is_macos():
        return ["", "chrome", "firefox", "safari"]
    return ["", "chrome", "firefox"]


def _is_download_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("yt-dlp", "sign in", "cookies", "not a bot"))


def _download_auth_diagnostics() -> str:
    cookies_file_raw = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
    cookies_file_exists = False
    if cookies_file_raw:
        try:
            cookies_file_exists = Path(cookies_file_raw).expanduser().exists()
        except Exception:
            cookies_file_exists = False

    browser_raw = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER", "").strip().lower()
    browser_value = browser_raw if browser_raw else "<unset>"
    return (
        "Auth diagnostics: "
        f"platform={sys.platform}, "
        f"YT_DLP_COOKIES_FILE_set={bool(cookies_file_raw)}, "
        f"YT_DLP_COOKIES_FILE_exists={cookies_file_exists}, "
        f"YT_DLP_COOKIES_FROM_BROWSER_set={bool(browser_raw)}, "
        f"YT_DLP_COOKIES_FROM_BROWSER_value={browser_value}"
    )


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


def _split_scene_selection(
    grouped_scene: dict[str, list[str]],
    selected_scene_categories: list[str],
) -> tuple[list[str], list[str], list[str]]:
    selected = set(selected_scene_categories or [])
    return (
        [c for c in grouped_scene["Violence"] if c in selected],
        [c for c in grouped_scene["Sex and Nudity"] if c in selected],
        [c for c in grouped_scene["Other Scenes"] if c in selected],
    )


def _load_ui_snapshot(config_path: str, selected_profile: str | None = None) -> tuple[dict, str]:
    try:
        config = load_term_config(config_path)
        categories = get_category_names(config)
        scene_categories = get_scene_category_names(config)
        grouped_scene = _group_scene_categories(scene_categories)
        scene_thresholds = _collect_scene_thresholds(config.get("scene_detection", {}))
        profile_names = [MANUAL_PROFILE_NAME, *get_processing_profile_names(config)]
        default_profile = selected_profile or get_default_processing_profile_name(config)
        resolved_profile = resolve_processing_profile(config, default_profile)
        return (
            {
                "categories": categories,
                "category_defaults": resolved_profile.selected_categories,
                "scene_grouped": grouped_scene,
                "scene_defaults": resolved_profile.selected_scene_categories,
                "thresholds": scene_thresholds,
                "processing_profiles": profile_names,
                "selected_profile": resolved_profile.name,
                "profile_description": resolved_profile.description,
                "profile_mute_scene_audio": resolved_profile.mute_scene_audio,
                "profile_blur_scene_video": resolved_profile.blur_scene_video,
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
                "processing_profiles": [MANUAL_PROFILE_NAME],
                "selected_profile": MANUAL_PROFILE_NAME,
                "profile_description": "",
                "profile_mute_scene_audio": False,
                "profile_blur_scene_video": True,
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


def _reload_ui(config_path: str, processing_profile: str):
    snapshot, status = _load_ui_snapshot(config_path, processing_profile)
    categories = snapshot["categories"]
    defaults = snapshot["category_defaults"]
    grouped = snapshot["scene_grouped"]
    thresholds = snapshot["thresholds"]
    selected_profile = snapshot["selected_profile"]
    selected_scene_violence, selected_scene_sex_nudity, selected_scene_other = _split_scene_selection(
        grouped,
        snapshot["scene_defaults"],
    )

    return (
        gr.update(choices=snapshot["processing_profiles"], value=selected_profile),
        snapshot["profile_description"],
        gr.update(choices=categories, value=defaults),
        gr.update(choices=grouped["Violence"], value=selected_scene_violence),
        gr.update(choices=grouped["Sex and Nudity"], value=selected_scene_sex_nudity),
        gr.update(choices=grouped["Other Scenes"], value=selected_scene_other),
        json.dumps(thresholds, indent=2),
        gr.update(value=snapshot["profile_mute_scene_audio"]),
        gr.update(value=snapshot["profile_blur_scene_video"]),
        gr.update(value=False),
        gr.update(value=False),
        gr.update(value=False),
        gr.update(value=False),
        categories,
        grouped["Violence"],
        grouped["Sex and Nudity"],
        grouped["Other Scenes"],
        status,
    )


def _apply_processing_profile(config_path: str, processing_profile: str):
    snapshot, status = _load_ui_snapshot(config_path, processing_profile)
    grouped = snapshot["scene_grouped"]
    selected_scene_violence, selected_scene_sex_nudity, selected_scene_other = _split_scene_selection(
        grouped,
        snapshot["scene_defaults"],
    )
    return (
        snapshot["profile_description"],
        gr.update(value=snapshot["category_defaults"]),
        gr.update(value=selected_scene_violence),
        gr.update(value=selected_scene_sex_nudity),
        gr.update(value=selected_scene_other),
        gr.update(value=snapshot["profile_mute_scene_audio"]),
        gr.update(value=snapshot["profile_blur_scene_video"]),
        gr.update(value=False),
        gr.update(value=False),
        gr.update(value=False),
        gr.update(value=False),
        status,
    )


def run_pipeline(
    video_url: str,
    yt_dlp_cookies_file: str,
    yt_dlp_cookies_from_browser: str,
    config_path: str,
    processing_profile: str,
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
        cookies_file = (yt_dlp_cookies_file or "").strip()
        cookies_browser = (yt_dlp_cookies_from_browser or "").strip().lower()

        if cookies_file:
            cookie_path = Path(cookies_file).expanduser()
            if not cookie_path.exists():
                raise ValueError(
                    "Cookies file path does not exist: "
                    f"{cookie_path}. Provide a valid mounted cookies.txt path "
                    "(Netscape format) or clear the field."
                )

        if cookies_browser == "safari" and not _is_macos():
            raise ValueError(
                "YT_DLP_COOKIES_FROM_BROWSER=safari is only supported on macOS. "
                "For Linux/Kubernetes, provide a Netscape cookies file via the "
                "Cookies file path field (YT_DLP_COOKIES_FILE)."
            )

        if cookies_file:
            os.environ["YT_DLP_COOKIES_FILE"] = cookies_file
        else:
            os.environ.pop("YT_DLP_COOKIES_FILE", None)
        if cookies_browser:
            os.environ["YT_DLP_COOKIES_FROM_BROWSER"] = cookies_browser
        else:
            os.environ.pop("YT_DLP_COOKIES_FROM_BROWSER", None)

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
            processing_profile=processing_profile,
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
        diagnostics = ""
        if _is_download_auth_error(exc):
            diagnostics = _download_auth_diagnostics() + "\n\n"
        return None, f"Error: {exc}\n\n{diagnostics}{debug_trace}", "{}", []


def build_ui() -> gr.Blocks:
    snapshot, status = _load_ui_snapshot(DEFAULT_CONFIG_PATH)
    categories = snapshot["categories"]
    defaults = snapshot["category_defaults"]
    grouped = snapshot["scene_grouped"]
    scene_defaults = snapshot["scene_defaults"]
    thresholds_json = json.dumps(snapshot["thresholds"], indent=2)
    cookies_file_default = os.environ.get("YT_DLP_COOKIES_FILE", "")
    cookies_browser_default = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER", "")
    browser_choices = _supported_browser_cookie_choices()
    if cookies_browser_default not in browser_choices:
        cookies_browser_default = ""
    selected_scene_violence, selected_scene_sex_nudity, selected_scene_other = _split_scene_selection(
        grouped,
        scene_defaults,
    )

    with gr.Blocks(title="Content Filtering POC") as app:
        audio_choices_state = gr.State(categories)
        violence_choices_state = gr.State(grouped["Violence"])
        sex_nudity_choices_state = gr.State(grouped["Sex and Nudity"])
        other_choices_state = gr.State(grouped["Other Scenes"])

        gr.Markdown("## Content Filtering\nPaste a video URL, choose categories, run, then review flagged frames.")

        processing_profile = gr.Dropdown(
            label="Processing profile",
            choices=snapshot["processing_profiles"],
            value=snapshot["selected_profile"],
        )
        profile_description = gr.Textbox(
            label="Profile summary",
            value=snapshot["profile_description"],
            interactive=False,
            lines=2,
        )

        video_url = gr.Textbox(
            label="Video URL",
            placeholder="https://... or /absolute/path/video.mp4",
            lines=1,
        )

        with gr.Accordion("Download Authentication (yt-dlp)", open=False):
            yt_dlp_cookies_file = gr.Textbox(
                label="Cookies file path (optional)",
                value=cookies_file_default,
                placeholder="/absolute/path/cookies.txt",
                lines=1,
            )
            yt_dlp_cookies_from_browser = gr.Dropdown(
                label="Read cookies from browser (optional)",
                choices=browser_choices,
                value=cookies_browser_default,
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
                    value=selected_scene_violence,
                )
            with gr.Accordion("Sex and Nudity", open=False):
                select_all_sex_nudity = gr.Checkbox(label="Select all in Sex and Nudity", value=False)
                selected_scene_sex_nudity = gr.CheckboxGroup(
                    label="Sex/Nudity subcategories",
                    choices=grouped["Sex and Nudity"],
                    value=selected_scene_sex_nudity,
                )
            with gr.Accordion("Other Scenes", open=False):
                select_all_other = gr.Checkbox(label="Select all in Other Scenes", value=False)
                selected_scene_other = gr.CheckboxGroup(
                    label="Other scene subcategories",
                    choices=grouped["Other Scenes"],
                    value=selected_scene_other,
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

            mute_scene_audio = gr.Checkbox(
                label="Mute detected scene intervals",
                value=snapshot["profile_mute_scene_audio"],
            )
            blur_scene_video = gr.Checkbox(
                label="Blur detected scene intervals",
                value=snapshot["profile_blur_scene_video"],
            )
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

        processing_profile.change(
            fn=_apply_processing_profile,
            inputs=[config_path, processing_profile],
            outputs=[
                profile_description,
                selected_categories,
                selected_scene_violence,
                selected_scene_sex_nudity,
                selected_scene_other,
                mute_scene_audio,
                blur_scene_video,
                select_all_audio,
                select_all_violence,
                select_all_sex_nudity,
                select_all_other,
                config_status,
            ],
        )

        reload_btn.click(
            fn=_reload_ui,
            inputs=[config_path, processing_profile],
            outputs=[
                processing_profile,
                profile_description,
                selected_categories,
                selected_scene_violence,
                selected_scene_sex_nudity,
                selected_scene_other,
                threshold_overrides_json,
                mute_scene_audio,
                blur_scene_video,
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
                yt_dlp_cookies_file,
                yt_dlp_cookies_from_browser,
                config_path,
                processing_profile,
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
