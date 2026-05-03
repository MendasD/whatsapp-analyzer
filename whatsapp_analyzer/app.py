"""
Streamlit web interface for whatsapp-analyser.

Runs the full pipeline step-by-step with a live progress bar,
then displays results in themed tabs. Works without the [media] extra.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the package importable when Streamlit runs this file directly
# (i.e. without `pip install -e .` being active in the current env).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# Must be the very first Streamlit call
st.set_page_config(
    page_title="WhatsApp Analyser",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS

_CSS = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #075e54 0%, #128c7e 100%);
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stFileUploader > label,
[data-testid="stSidebar"] .stCheckbox > label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
[data-testid="stSidebar"] .stButton > button {
    background: #25d366;
    color: #fff !important;
    border: none;
    border-radius: 8px;
    width: 100%;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 0;
    letter-spacing: 0.04em;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #1ebe59; }

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #25d366;
    margin-bottom: 1rem;
}
.metric-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #075e54;
    line-height: 1.1;
}
.metric-card .sub { font-size: 0.8rem; color: #aaa; margin-top: 0.2rem; }

.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #075e54;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e8f5e9;
}

.hero { text-align: center; padding: 4rem 2rem 3rem; }
.hero h1 { font-size: 2.6rem; font-weight: 800; color: #075e54; margin-bottom: 0.5rem; }
.hero p { font-size: 1.1rem; color: #666; max-width: 560px; margin: 0 auto 2rem; }
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 1rem;
    max-width: 860px;
    margin: 0 auto;
}
.feature-tile {
    background: #fff;
    border-radius: 12px;
    padding: 1.4rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    text-align: center;
}
.feature-tile .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-tile .name { font-weight: 700; color: #333; font-size: 0.95rem; }
.feature-tile .desc { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

.app-header {
    background: linear-gradient(90deg, #075e54, #128c7e);
    border-radius: 12px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
}
.app-header h2 { color: #fff; font-size: 1.5rem; font-weight: 800; margin: 0; }
.app-header p { color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.2rem 0 0; }

.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #075e54 !important; border-bottom-color: #25d366 !important; }
</style>
"""


# Helpers

def _whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"{sub_html}</div>"
    )


def _section(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header"><span>{icon}</span>{title}</div>',
        unsafe_allow_html=True,
    )


# Sidebar

def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 💬 WhatsApp Analyser")
        st.caption("Analyse your chats locally — nothing leaves your device.")
        st.divider()

        uploaded = st.file_uploader(
            "Chat export",
            type=["zip", "txt"],
            help="Upload a .zip (native export) or _chat.txt file.",
        )

        st.divider()
        st.markdown("**⚙️ Analysis settings**")

        n_topics = st.slider("Number of topics", min_value=2, max_value=15, value=5)
        min_words = st.slider("Min words per message", min_value=1, max_value=10, value=3)
        language = st.selectbox("Language", ["Auto", "French", "English"], index=0)
        anonymise = st.checkbox("Anonymise authors", value=False)

        if _whisper_available():
            enable_media = st.checkbox("Enable media transcription", value=False)
        else:
            enable_media = False
            st.caption("🔇 *Media transcription unavailable — install Whisper to enable*")

        st.divider()
        run = st.button("▶  Run analysis", use_container_width=True)

    return {
        "uploaded": uploaded,
        "n_topics": n_topics,
        "min_words": min_words,
        "language": language,
        "anonymise": anonymise,
        "enable_media": enable_media,
        "run": run,
    }


# Welcome screen

def _render_welcome() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>💬 WhatsApp Analyser</h1>
          <p>Upload your chat export, click <strong>Run analysis</strong>
             and explore your conversations in seconds — 100 % local.</p>
        </div>
        <div class="feature-grid">
          <div class="feature-tile">
            <div class="icon">🗂️</div>
            <div class="name">Topic modelling</div>
            <div class="desc">Discover the main themes discussed in your group</div>
          </div>
          <div class="feature-tile">
            <div class="icon">😊</div>
            <div class="name">Sentiment analysis</div>
            <div class="desc">Track the mood of conversations over time</div>
          </div>
          <div class="feature-tile">
            <div class="icon">📅</div>
            <div class="name">Activity heatmaps</div>
            <div class="desc">See when your group is most active by hour and day</div>
          </div>
          <div class="feature-tile">
            <div class="icon">👤</div>
            <div class="name">User profiles</div>
            <div class="desc">Compare participation and communication style</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


#  Pipeline

def _run_pipeline(uploaded_file, config: dict) -> dict:
    """Write upload to tmp dir, chain pipeline steps, return results dict."""
    from whatsapp_analyzer.loader import Loader
    from whatsapp_analyzer.parser import Parser
    from whatsapp_analyzer.cleaner import Cleaner
    from whatsapp_analyzer.topic_classifier import TopicClassifier
    from whatsapp_analyzer.sentiment_analyzer import SentimentAnalyzer
    from whatsapp_analyzer.temporal_analyzer import TemporalAnalyzer
    from whatsapp_analyzer.user_analyzer import UserAnalyzer
    from whatsapp_analyzer.visualizer import Visualizer

    lang_map = {"Auto": None, "French": "fr", "English": "en"}
    forced_lang = lang_map[config["language"]]

    progress = st.progress(0, text="💾 Saving file…")
    status = st.empty()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getvalue())
        output_dir = Path(tmp_dir) / "reports"

        def _step(pct: int, msg: str) -> None:
            progress.progress(pct, text=msg)
            status.caption(msg)

        _step(10, "📂 Loading export…")
        loaded = Loader().load(tmp_path)

        _step(25, "📝 Parsing messages…")
        df_raw = Parser().parse(loaded.chat_path)

        if config["anonymise"]:
            from whatsapp_analyzer.parser import Parser as _P
            from whatsapp_analyzer.utils import anonymize_author
            df_raw["author"] = df_raw["author"].apply(anonymize_author)

        _step(40, "🧹 Cleaning & preprocessing…")
        df_clean = Cleaner(
            lang=forced_lang,
            min_words=config["min_words"],
        ).clean(df_raw)

        results: dict = {
            "group_name": loaded.group_name,
            "df_raw":   df_raw,
            "df_clean": df_clean,
            "topics": None,
            "sentiment": None,
            "temporal": None,
            "users": None,
            "report_bytes": None,
        }

        _step(52, "🗂️ Extracting topics…")
        try:
            results["topics"] = TopicClassifier(
                n_topics=config["n_topics"]
            ).fit_transform(df_clean)
        except Exception as exc:
            st.warning(f"Topic modelling skipped: {exc}")

        _step(65, "😊 Analysing sentiment…")
        try:
            results["sentiment"] = SentimentAnalyzer().analyze(df_clean)
        except Exception as exc:
            st.warning(f"Sentiment analysis skipped: {exc}")

        _step(76, "📅 Computing activity patterns…")
        try:
            results["temporal"] = TemporalAnalyzer().analyze(df_clean)
        except Exception as exc:
            st.warning(f"Temporal analysis skipped: {exc}")

        _step(86, "👤 Building user profiles…")
        try:
            results["users"] = UserAnalyzer().build_profiles(results)
        except Exception as exc:
            st.warning(f"User profiling skipped: {exc}")

        _step(94, "📄 Generating HTML report…")
        try:
            report_path = Visualizer().generate_report(results, output_dir)
            results["report_bytes"] = report_path.read_bytes()
        except Exception as exc:
            st.warning(f"Report generation skipped: {exc}")

        progress.progress(100, text="✅ Analysis complete!")
        status.empty()
        progress.empty()

    return results


# Tabs 

def _tab_overview(results: dict) -> None:
    import matplotlib.pyplot as plt

    df = results["df_clean"]
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    days = max((t_max - t_min).days, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_card("Messages", f"{len(df):,}",
                          f"avg {len(df)/days:.1f} / day"), unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Participants", str(int(df["author"].nunique()))),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(_card("Period start", t_min.strftime("%d %b %Y")),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(_card("Period end", t_max.strftime("%d %b %Y")),
                    unsafe_allow_html=True)

    temporal = results.get("temporal")
    if temporal:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown(_card("Peak hour", f"{temporal['peak_hour']:02d}:00"),
                        unsafe_allow_html=True)
        with c6:
            st.markdown(_card("Most active day", temporal["peak_day"]),
                        unsafe_allow_html=True)

    _section("📈", "Daily message timeline")
    if temporal and temporal.get("timeline") is not None:
        tl = temporal["timeline"]
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(tl.index, tl["count"], alpha=0.25, color="#25d366")
        ax.plot(tl.index, tl["count"], color="#075e54", linewidth=2)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Messages / day", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Timeline data not available.")


def _tab_topics(results: dict) -> None:
    import matplotlib.pyplot as plt
    from whatsapp_analyzer.visualizer import Visualizer

    topics = results.get("topics")
    if topics is None:
        st.info("Topic modelling did not produce results for this export.")
        return

    _section("📊", "Topic distribution")
    fig = Visualizer().plot_topic_distribution(results)
    st.pyplot(fig)
    plt.close(fig)

    _section("☁️", "Word clouds")
    gt = topics["group_topics"]
    n_cols = min(len(gt), 3)
    cols = st.columns(n_cols)
    for i, (_, row) in enumerate(gt.iterrows()):
        with cols[i % n_cols]:
            st.caption(f"**Topic {int(row['topic_id'])}** · {row['weight']:.2f}")
            fig = Visualizer().plot_wordcloud(results, topic_id=int(row["topic_id"]))
            st.pyplot(fig)
            plt.close(fig)


def _tab_sentiment(results: dict) -> None:
    import matplotlib.pyplot as plt
    from whatsapp_analyzer.visualizer import Visualizer

    sentiment = results.get("sentiment")
    if sentiment is None:
        st.info("Sentiment analysis did not produce results for this export.")
        return

    g = sentiment.get("global", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_card("Mean sentiment", f"{g.get('mean', 0):.2f}"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Positive", f"{g.get('pos_pct', 0)*100:.0f} %"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(_card("Negative", f"{g.get('neg_pct', 0)*100:.0f} %"),
                    unsafe_allow_html=True)

    _section("📈", "Sentiment over time")
    fig = Visualizer().plot_sentiment_timeline(results)
    st.pyplot(fig)
    plt.close(fig)

    by_user = sentiment.get("by_user")
    if by_user is not None and not by_user.empty:
        _section("👤", "Sentiment by participant")
        by_user_s = by_user.set_index("author")["mean_score"].sort_values()
        fig2, ax = plt.subplots(figsize=(10, max(3, len(by_user_s) * 0.45)))
        colors = ["#e74c3c" if v < 0 else "#25d366" for v in by_user_s.values]
        ax.barh(by_user_s.index, by_user_s.values, color=colors, edgecolor="none")
        ax.axvline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean sentiment score")
        ax.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def _tab_activity(results: dict) -> None:
    import matplotlib.pyplot as plt
    from whatsapp_analyzer.visualizer import Visualizer

    _section("👥", "Messages per participant")
    fig = Visualizer().plot_user_activity(results)
    st.pyplot(fig)
    plt.close(fig)

    temporal = results.get("temporal")
    if temporal is None:
        st.info("Temporal data not available.")
        return

    _section("🕐", "Activity heatmap — hour × weekday")
    fig = Visualizer().plot_hourly_heatmap(results)
    st.pyplot(fig)
    plt.close(fig)

    weekly = temporal.get("weekly_activity")
    if weekly is not None:
        _section("📅", "Activity by day of week")
        fig2, ax = plt.subplots(figsize=(9, 3))
        ax.bar(weekly.index, weekly.values, color="#128c7e", alpha=0.85, edgecolor="none")
        ax.set_ylabel("Messages")
        ax.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def _tab_report(results: dict) -> None:
    report_bytes = results.get("report_bytes")
    if report_bytes is None:
        st.warning("Report generation failed — check the warnings above.")
        return

    group = results.get("group_name", "group")
    st.success(
        "Your report is ready. It is fully self-contained — no internet required to open it."
    )
    st.download_button(
        label="⬇️  Download HTML report",
        data=report_bytes,
        file_name=f"{group}_report.html",
        mime="text/html",
        use_container_width=True,
    )


# Results layout 

def _render_results(results: dict) -> None:
    group = results.get("group_name", "Unknown group")
    st.markdown(
        f'<div class="app-header">'
        f'<h2>💬 {group}</h2>'
        f'<p>Analysis complete — explore the tabs below</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5 = st.tabs(
        ["📊 Overview", "🗂️ Topics", "😊 Sentiment", "📅 Activity", "📄 Report"]
    )
    with t1:
        _tab_overview(results)
    with t2:
        _tab_topics(results)
    with t3:
        _tab_sentiment(results)
    with t4:
        _tab_activity(results)
    with t5:
        _tab_report(results)


# Entry point

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    config = _render_sidebar()

    if config["run"]:
        if config["uploaded"] is None:
            st.error("Please upload a .zip or .txt chat export first.")
        else:
            try:
                results = _run_pipeline(config["uploaded"], config)
                st.session_state["results"] = results
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.session_state.pop("results", None)

    if "results" in st.session_state:
        _render_results(st.session_state["results"])
    else:
        _render_welcome()


if __name__ == "__main__":
    main()
