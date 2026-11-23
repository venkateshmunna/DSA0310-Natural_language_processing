from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

import answer_generator
import evaluation
import media_processor
import retrieval


PROJECT_TITLE = "trustRAG — Multimodal Media Retrieval and Grounded Answers"

SECTIONS = {
    "Dashboard": "⌂",
    "Upload Media": "＋",
    "Knowledge Base": "▤",
    "Chat": "◉",
    "Evaluation": "▥",
}


def apply_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #262626;
            --muted: #737373;
            --line: #efefef;
            --pink: #ff3040;
            --purple: #833ab4;
            --canvas: #fafafa;
        }
        * { box-sizing: border-box; }
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: "Segoe UI", Inter, Arial, sans-serif;
            font-size: 16px;
            color: var(--ink);
        }
        .stApp { background: var(--canvas); }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
            min-width: 230px;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 4px 8px 26px;
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            color: #fff;
            background: radial-gradient(circle at 30% 110%, #fdf497 0 20%, #fd5949 45%, #d6249f 65%, #285aeb 100%);
            font-size: 17px;
            font-weight: 800;
            box-shadow: 0 6px 18px #d6249f26;
        }
        .brand-name {
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -0.8px;
            background: linear-gradient(90deg, #262626, #833ab4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .brand-caption {
            margin-top: 1px;
            color: var(--muted);
            font-size: 13px;
        }
        .eyebrow {
            color: #c13584;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        .page-title {
            margin: 3px 0 4px;
            color: var(--ink);
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -1.2px;
        }
        .page-subtitle {
            margin: 0 0 25px;
            color: var(--muted);
            font-size: 17px;
            line-height: 1.5;
        }
        .metric-card, .content-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: #fff;
            box-shadow: 0 8px 24px #00000008;
        }
        .metric-card { min-height: 138px; padding: 22px; }
        .metric-label { color: var(--muted); font-size: 15px; font-weight: 700; }
        .metric-value {
            margin-top: 11px;
            color: var(--ink);
            font-size: 32px;
            font-weight: 800;
        }
        .metric-note { margin-top: 4px; color: #8e8e8e; font-size: 13px; }
        .content-card { padding: 22px 24px; }
        .card-title {
            margin: 0 0 6px;
            color: var(--ink);
            font-size: 18px;
            font-weight: 800;
        }
        .card-copy { color: var(--muted); font-size: 15px; line-height: 1.65; }
        .status-pill {
            display: inline-block;
            margin-top: 12px;
            padding: 5px 10px;
            border-radius: 20px;
            background: #f0f3f8;
            color: #64708a;
            font-size: 13px;
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            border: 0;
            border-radius: 10px;
            min-height: 44px;
            padding: 8px 18px;
            background: linear-gradient(90deg, #833ab4, #fd1d1d, #fcb045);
            color: #fff;
            font-size: 16px;
            font-weight: 700;
        }
        .stButton > button[kind="secondary"], .stButton > button {
            min-height: 44px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 650;
        }
        [data-testid="stFileUploader"] { font-size: 16px; }
        [data-testid="stTextInput"] input, [data-baseweb="textarea"] textarea,
        [data-testid="stSelectbox"] div, [data-testid="stSlider"] label {
            font-size: 16px;
        }
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
            font-size: 16px;
        }
        [data-testid="stTable"], [data-testid="stDataFrame"] {
            font-size: 15px;
        }
        .stMarkdown, .stCaption { font-size: 16px; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            min-height: 48px;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 650;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #f2f2f2;
            color: #262626;
        }
        .block-container { max-width: 1180px; padding-top: 2.2rem; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_heading(section, subtitle):
    st.markdown(
        f'<div class="eyebrow">trustRAG · MULTIMODAL KNOWLEDGE</div>'
        f'<h1 class="page-title">{section}</h1>'
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_dashboard():
    page_heading(
        "Your media workspace",
        "A clear view of your indexed visual knowledge and retrieval activity.",
    )

    statistics = media_processor.get_statistics()
    index_status = retrieval.get_index_status()
    metrics = st.columns(4)
    with metrics[0]:
        metric_card(
            "Media files",
            str(statistics["media_files"]),
            "Images and videos saved locally",
        )
    with metrics[1]:
        metric_card(
            "Knowledge records",
            str(statistics["assets"]),
            "Images and extracted video frames",
        )
    with metrics[2]:
        metric_card(
            "Processed files",
            f'{statistics["processed_files"]} / {statistics["media_files"]}',
            "OCR and image descriptions completed",
        )
    with metrics[3]:
        metric_card(
            "Indexed records",
            f'{index_status["indexed_assets"]} / {index_status["available_assets"]}',
            "Current" if index_status["is_current"] else "Refresh required",
        )

    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        files = media_processor.list_media_files()
        if files:
            st.markdown("#### Recent media")
            st.dataframe(
                [
                    {
                        "File": item["filename"],
                        "Type": item["media_type"].title(),
                        "Status": item["status"].title(),
                    }
                    for item in files[:8]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown(
                """
                <div class="content-card">
                  <div class="card-title">Start building your knowledge base</div>
                  <div class="card-copy">
                    Add images and videos to save them locally, extract video frames,
                    and generate OCR text and visual descriptions.
                  </div>
                  <span class="status-pill">Workspace is ready for media</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            """
            <div class="content-card">
              <div class="card-title">Project pipeline</div>
              <div class="card-copy">
                <b>01</b> Workspace and interface<br>
                <b>02</b> Image, video and OCR processing<br>
                <b>03</b> Visual and keyword retrieval<br>
                <b>04</b> Evidence-grounded answers<br>
                <b>05</b> Evaluation and documentation
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_upload():
    page_heading(
        "Upload media",
        "Choose image and video files to add to your local knowledge base.",
    )
    st.markdown(
        """
        <div class="content-card">
          <div class="card-title">Add images and videos</div>
          <div class="card-copy">
            Supported formats: JPG, JPEG, PNG, WEBP, MP4, MOV, AVI and MKV.
            Uploaded originals, extracted frames and processing results are stored
            in the local <code>data</code> folder.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    interval = st.slider(
        "Extract a video frame every (seconds)",
        min_value=1,
        max_value=60,
        value=5,
        step=1,
    )
    fast_column, text_column, caption_column = st.columns(3)
    with fast_column:
        fast_mode = st.checkbox(
            "Fast mode",
            value=False,
            help="Skips BLIP captions and resizes frames for faster OCR.",
        )
    with text_column:
        extract_ocr = st.checkbox("Extract visible text", value=True)
    with caption_column:
        generate_captions = st.checkbox(
            "Generate image descriptions",
            value=not fast_mode,
            disabled=fast_mode,
        )
    uploaded_files = st.file_uploader(
        "Select one or more media files",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        help="Image and video analysis may take time on the first run while models are downloaded.",
        key="stage2_media_upload",
    )
    if uploaded_files and st.button("Save and process media", type="primary"):
        progress = st.progress(0)
        total_files = len(uploaded_files)
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                media_id = media_processor.save_uploaded_media(
                    uploaded_file.name, uploaded_file.getvalue()
                )
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    media_processor.process_media(
                        media_id,
                        interval,
                        extract_ocr=extract_ocr,
                        generate_captions=generate_captions and not fast_mode,
                    )
                st.success(f"Saved and analyzed {uploaded_file.name}.")
            except Exception as error:
                st.error(f"Could not process {uploaded_file.name}: {error}")
            progress.progress((index + 1) / total_files)
        st.rerun()

    st.caption(
        "Processing is CPU-based. Fast mode reduces work per frame; the first run "
        "may take longer while OCR or BLIP model files download. Processing time "
        "depends on frame count, image size and enabled models."
    )

    failed_files = [
        item for item in media_processor.list_media_files() if item["status"] == "failed"
    ]
    if failed_files and st.button("Retry failed media"):
        progress = st.progress(0)
        for index, item in enumerate(failed_files):
            try:
                with st.spinner(f"Retrying {item['filename']}..."):
                    media_processor.process_media(
                        item["id"],
                        interval,
                        extract_ocr=extract_ocr,
                        generate_captions=generate_captions and not fast_mode,
                    )
                st.success(f"Processed {item['filename']}.")
            except Exception as error:
                st.error(f"Could not process {item['filename']}: {error}")
            progress.progress((index + 1) / len(failed_files))
        st.rerun()


def show_knowledge_base():
    page_heading(
        "Knowledge base",
        "Browse your source files, extracted frames, OCR text and visual descriptions.",
    )
    files = media_processor.list_media_files()
    if not files:
        st.info("Your knowledge base is empty. Add images or videos from Upload Media.")
        return

    st.markdown("#### Uploaded files")
    st.dataframe(
        [
            {
                "File": item["filename"],
                "Type": item["media_type"].title(),
                "Size (MB)": round(item["size_bytes"] / (1024 * 1024), 2),
                "Status": item["status"].title(),
                "Error": item["error"],
                "Added": item["created_at"],
            }
            for item in files
        ],
        use_container_width=True,
        hide_index=True,
    )

    assets = media_processor.list_media_assets()
    if not assets:
        st.info("No images or video frames have been analyzed yet.")
        return

    index_status = retrieval.get_index_status()
    st.markdown("#### Visual retrieval index")
    st.caption(
        f'CLIP/FAISS: {index_status["indexed_assets"]} of '
        f'{index_status["available_assets"]} assets · {index_status["message"]}'
    )
    if st.button("Build / refresh CLIP index", type="primary"):
        try:
            with st.spinner(
                "Creating visual embeddings. The first run downloads the CLIP model."
            ):
                index_result = retrieval.build_index()
            st.success(
                f'Indexed {index_result["indexed_assets"]} images and frames '
                f'({index_result["dimension"]}-dimensional CLIP vectors).'
            )
            st.rerun()
        except Exception as error:
            st.error(f"Could not build the visual index: {error}")

    st.markdown("#### Extracted images and frames")
    for start in range(0, len(assets), 3):
        columns = st.columns(3)
        for column, asset in zip(columns, assets[start : start + 3]):
            with column:
                if Path(asset["path"]).is_file():
                    st.image(str(asset["path"]), use_container_width=True)
                else:
                    st.warning("The saved image or frame file is missing.")
                source = asset["filename"]
                if asset["timestamp_seconds"] is not None:
                    timestamp = int(asset["timestamp_seconds"])
                    minutes, seconds = divmod(timestamp, 60)
                    source += f" · {minutes:02d}:{seconds:02d}"
                st.markdown(f"**{source}**")
                st.write(asset["caption"] or "No description was generated.")
                if asset["ocr_text"]:
                    with st.expander("Extracted text"):
                        st.text(asset["ocr_text"])
                else:
                    st.caption("No text detected.")


def show_chat():
    page_heading(
        "Ask your media",
        "Retrieve evidence and generate an answer grounded in uploaded media.",
    )
    index_status = retrieval.get_index_status()
    if not index_status["is_current"]:
        st.warning(
            "The visual index is not current. You can use BM25 now, or rebuild "
            "the CLIP/FAISS index before using Dense or Hybrid retrieval."
        )
        rebuild_column, status_column = st.columns([1, 3])
        with rebuild_column:
            if st.button("Rebuild visual index", type="primary"):
                try:
                    with st.spinner(
                        "Rebuilding visual embeddings. This may take time."
                    ):
                        index_result = retrieval.build_index()
                    st.success(
                        f'Indexed {index_result["indexed_assets"]} images and frames.'
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not rebuild the visual index: {error}")
        with status_column:
            st.caption(index_status["message"])
    else:
        st.caption(
            f'Visual index ready: {index_status["indexed_assets"]} assets indexed.'
        )
    query = st.text_input(
        "Search your media",
        placeholder="For example: What text appears in the presentation?",
    )
    assets = media_processor.list_media_assets()
    media_ids_with_assets = {str(asset["media_id"]) for asset in assets}
    searchable_media = [
        item
        for item in media_processor.list_media_files()
        if item["status"] == "processed" and str(item["id"]) in media_ids_with_assets
    ]
    media_by_id = {str(item["id"]): item for item in searchable_media}
    selected_media_ids = st.multiselect(
        "Files to search",
        list(media_by_id),
        format_func=lambda media_id: (
            f'{media_by_id[media_id]["filename"]} · {media_id[:8]}'
        ),
        help="Choose one or more files to search. Leave empty to search all processed files.",
        key="chat_media_scope",
    )
    method_label = st.selectbox(
        "Retrieval method",
        ["Hybrid", "BM25", "Dense (CLIP + FAISS)"],
    )
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    plate_query = re.search(
        r"(license|number\s*plate|registration\s*plate)",
        query,
        re.IGNORECASE,
    )
    force_plate_rescan = False
    if plate_query:
        force_plate_rescan = st.checkbox(
            "Re-scan plate frames (slower)",
            value=False,
            help="Use this after replacing video frames or if you want to refresh cached OCR reads.",
            key="force_plate_rescan",
        )
    try:
        available_models = answer_generator.list_models()
    except RuntimeError as error:
        available_models = []
        st.warning(str(error))

    default_model = answer_generator.DEFAULT_MODEL
    if default_model not in available_models and available_models:
        default_model = available_models[0]
    model_index = (
        available_models.index(default_model) if available_models else 0
    )
    model_name = st.selectbox(
        "Ollama answer model",
        available_models or [answer_generator.DEFAULT_MODEL],
        index=model_index,
        disabled=not available_models,
    )

    dense_weight = 0.5
    sparse_weight = 0.5
    if method_label == "Hybrid":
        dense_column, sparse_column = st.columns(2)
        with dense_column:
            dense_weight = st.slider(
                "Visual similarity weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
        with sparse_column:
            sparse_weight = st.slider(
                "Keyword match weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

    if st.button("Search knowledge base", type="primary"):
        if plate_query:
            first_count_match = re.search(
                r"(?:first|give)\s+(\d+)",
                query,
                re.IGNORECASE,
            )
            if first_count_match:
                count = int(first_count_match.group(1))
            else:
                word_counts = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                }
                count_match = re.search(
                    r"(?:first|give)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b",
                    query,
                    re.IGNORECASE,
                )
                count = word_counts[count_match.group(1).lower()] if count_match else 5
            count = max(1, min(count, 25))
            scope_media = [
                item
                for item in searchable_media
                if not selected_media_ids or str(item["id"]) in selected_media_ids
            ]
            selected_videos = [
                item for item in scope_media if item["media_type"] == "video"
            ]
            if not selected_videos:
                st.warning(
                    "Select or process at least one video to ask about license plates."
                )
                return

            plate_progress = st.progress(0.0, text="Reading selected video frames...")
            plate_candidates = []
            for video_index, video in enumerate(selected_videos):
                def update_plate_progress(
                    value,
                    offset=video_index,
                    filename=video["filename"],
                ):
                    progress_value = (offset + value) / len(selected_videos)
                    plate_progress.progress(
                        progress_value,
                        text=f"Reading {filename}...",
                    )

                try:
                    with st.spinner(f"Checking frames in {video['filename']}..."):
                        plate_candidates.extend(
                            media_processor.extract_license_plate_candidates(
                                video["id"],
                                limit=count,
                                progress_callback=update_plate_progress,
                                force_refresh=force_plate_rescan,
                            )
                        )
                except Exception as error:
                    st.error(f"Could not read plates from {video['filename']}: {error}")
            plate_progress.empty()
            if not plate_candidates:
                st.warning(
                    "No UK-format license plates passed the OCR confidence checks "
                    "in the selected videos' sampled frames."
                )
                return

            st.markdown("#### License plate results")
            st.success(
                f"Found {len(plate_candidates)} confidence-filtered plate reads "
                f"across {len(selected_videos)} selected video(s), up to {count} "
                "per video."
            )
            st.caption(
                "These are OCR readings, not verified vehicle identities. The result "
                "is limited to selected videos' UK-format plates that passed OCR "
                "checks. Timestamps are local to each video; nearby repeated reads "
                "are merged, and obscured or missed plates may be absent. Verify "
                "each frame."
            )
            plate_rows = [
                {
                    "Plate OCR": candidate["plate"],
                    "Raw OCR": candidate["raw_ocr"],
                    "Filename": candidate["filename"],
                    "Timestamp": candidate["timestamp"],
                    "OCR confidence": candidate["confidence"],
                    "Repeated reads": candidate["support_count"],
                }
                for candidate in plate_candidates
            ]
            st.dataframe(
                plate_rows,
                use_container_width=True,
                hide_index=True,
            )
            for candidate in plate_candidates:
                if Path(candidate["path"]).is_file():
                    st.image(
                        candidate["path"],
                        caption=(
                            f'{candidate["plate"]} · {candidate["filename"]} '
                            f'· {candidate["timestamp"]}'
                        ),
                        width=600,
                    )
            return

        try:
            with st.spinner("Searching indexed media..."):
                results = retrieval.search(
                    query,
                    method="Dense" if method_label.startswith("Dense") else method_label,
                    top_k=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    media_ids=selected_media_ids or None,
                )
        except Exception as error:
            st.error(f"Search could not be completed: {error}")
            return

        if not results:
            st.warning("No matching media evidence was found for this query.")
            return

        st.markdown("#### Answer")
        if not answer_generator.has_sufficient_evidence(results):
            st.warning(
                "Insufficient evidence: the uploaded media does not provide enough "
                "information to answer this question."
            )
        elif not available_models:
            st.warning(
                "Retrieval succeeded, but Ollama is unavailable. Start Ollama and "
                "install at least one model."
            )
        else:
            try:
                with st.spinner(f"Generating an evidence-grounded answer with {model_name}..."):
                    answer, cited_sources = answer_generator.generate_answer(
                        query,
                        results,
                        model=model_name,
                    )
                st.success("Answer generated")
                st.write(answer)
                if cited_sources:
                    st.caption(
                        "Cited evidence: "
                        + " · ".join(
                            source["source"] for source in cited_sources
                        )
                    )
            except Exception as error:
                st.error(str(error))

        st.markdown(f"#### Top {len(results)} results")
        for result in results:
            with st.container():
                image_column, evidence_column = st.columns([1, 2])
                with image_column:
                    if Path(result["path"]).is_file():
                        st.image(result["path"], use_container_width=True)
                    else:
                        st.warning("The indexed image or frame file is missing.")
                with evidence_column:
                    st.markdown(f'**{result["source"]}**')
                    st.caption(f'Retrieval score: {result["score"]:.3f}')
                    if result["similarity"] is not None:
                        st.caption(
                            f'CLIP cosine similarity: {result["similarity"]:.3f}'
                        )
                    st.write(result["caption"] or "No image description.")
                    if result["ocr_text"]:
                        with st.expander("Extracted text"):
                            st.text(result["ocr_text"])


def show_evaluation():
    page_heading(
        "Retrieval evaluation",
        "Compare keyword, visual-semantic and hybrid retrieval with measured results.",
    )

    st.info(
        "trustRAG does not contain sample benchmark scores. Add a labeled CSV "
        "and run retrieval to calculate metrics from the current media index."
    )
    st.markdown("#### Evaluation dataset")
    st.caption(
        "CSV columns: query, relevant_asset_ids, and optional reference_answer. "
        "Separate multiple asset IDs with |, ; or comma."
    )

    template = pd.DataFrame(
        [
            {
                "query": "Example: What revenue is shown?",
                "relevant_asset_ids": "replace-with-media-asset-id",
                "reference_answer": "Example: Revenue increased by 25 percent.",
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "trustrag_evaluation_template.csv",
        "text/csv",
    )

    uploaded_dataset = st.file_uploader(
        "Upload evaluation CSV",
        type=["csv"],
        key="evaluation_dataset",
    )
    if uploaded_dataset is not None:
        try:
            frame = pd.read_csv(uploaded_dataset)
            if not {"query", "relevant_asset_ids"}.issubset(frame.columns):
                st.error(
                    "The CSV must contain 'query' and 'relevant_asset_ids' columns."
                )
            elif frame.empty:
                st.error("The evaluation CSV contains no query rows.")
            else:
                if "reference_answer" not in frame.columns:
                    frame["reference_answer"] = ""
                frame = frame.fillna("")
                evaluation.save_dataset(frame.to_dict("records"))
                st.success(f"Saved {len(frame)} evaluation queries.")
        except Exception as error:
            st.error(f"Could not read the evaluation CSV: {error}")

    rows = evaluation.load_dataset()
    if rows:
        st.markdown("#### Saved queries")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No evaluation dataset has been saved.")

    st.markdown("#### Run measured evaluation")
    k = st.slider("K for ranking metrics", min_value=1, max_value=10, value=5)
    dense_column, sparse_column = st.columns(2)
    with dense_column:
        dense_weight = st.slider(
            "Hybrid visual weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="evaluation_dense_weight",
        )
    with sparse_column:
        sparse_weight = st.slider(
            "Hybrid keyword weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="evaluation_sparse_weight",
        )

    faithfulness_method = st.radio(
        "Faithfulness method",
        ["Text overlap", "Ollama classifier"],
        horizontal=True,
        help=(
            "Text overlap compares reference answers with evidence. "
            "Ollama classifier asks a local model whether an answer is supported."
        ),
    )
    model_name = None
    if faithfulness_method == "Ollama classifier":
        try:
            models = answer_generator.list_models()
            model_name = st.selectbox(
                "Ollama faithfulness model",
                models,
                index=(
                    models.index(answer_generator.DEFAULT_MODEL)
                    if answer_generator.DEFAULT_MODEL in models
                    else 0
                ),
            )
        except RuntimeError as error:
            st.warning(str(error))

    if st.button("Run evaluation", type="primary"):
        if not rows:
            st.error("Upload or save an evaluation dataset before running.")
        elif faithfulness_method == "Ollama classifier" and not model_name:
            st.error("A reachable Ollama model is required for Ollama faithfulness.")
        else:
            try:
                with st.spinner("Running BM25, Dense and Hybrid retrieval..."):
                    measured_results = evaluation.evaluate(
                        k=k,
                        dense_weight=dense_weight,
                        sparse_weight=sparse_weight,
                        model=model_name,
                        faithfulness_method=faithfulness_method,
                    )
                st.success(
                    f"Evaluation completed for "
                    f'{measured_results["methods"]["BM25"]["Query count"]} queries.'
                )
            except Exception as error:
                st.error(f"Evaluation could not be completed: {error}")
                return

    measured_results = evaluation.load_results()
    if not measured_results:
        st.caption("No completed evaluation run is stored.")
        return

    st.markdown("#### Measured results")
    summary = pd.DataFrame(measured_results["methods"]).T.reset_index()
    summary = summary.rename(columns={"index": "Method"})
    st.dataframe(summary, use_container_width=True, hide_index=True)

    metrics = [
        f'Precision@{measured_results["k"]}',
        f'Recall@{measured_results["k"]}',
        "MRR",
        "Faithfulness",
    ]
    chart_methods = list(measured_results["methods"].keys())
    chart = pd.DataFrame(
        [
            {
                "Method": method,
                **{
                    metric: measured_results["methods"][method][metric]
                    for metric in metrics
                },
            }
            for method in chart_methods
        ]
    )
    chart_column, detail_column = st.columns([1.35, 1])
    with chart_column:
        fig, ax = plt.subplots(figsize=(8, 4))
        chart.set_index("Method")[metrics].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
    with detail_column:
        st.markdown("#### Run information")
        st.write(
            {
                "K": measured_results["k"],
                "Hybrid dense weight": measured_results["dense_weight"],
                "Hybrid sparse weight": measured_results["sparse_weight"],
                "Faithfulness method": measured_results["faithfulness_method"],
            }
        )

    st.markdown("#### Per-query results")
    if measured_results["details"]:
        st.dataframe(
            measured_results["details"],
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(
        page_title="trustRAG",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">TR</div>
              <div>
                <div class="brand-name">trustRAG</div>
                <div class="brand-caption">Visual knowledge, grounded answers</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("WORKSPACE")
        selected = st.radio(
            "Main navigation",
            options=list(SECTIONS),
            format_func=lambda section: f"{SECTIONS[section]}   {section}",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("CAPSTONE PROJECT")
        st.caption("Multimodal retrieval across images and videos.")

    page_by_section = {
        "Dashboard": show_dashboard,
        "Upload Media": show_upload,
        "Knowledge Base": show_knowledge_base,
        "Chat": show_chat,
        "Evaluation": show_evaluation,
    }
    page_by_section[selected]()
    st.caption(PROJECT_TITLE)


if __name__ == "__main__":
    main()
