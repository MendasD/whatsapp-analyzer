"""
Visualisation module for WhatsApp conversation analysis.

Generates individual charts and self-contained HTML reports from the
pipeline results dict. All heavy imports (matplotlib, seaborn, wordcloud,
jinja2) are lazy so the module loads instantly even when extras are absent.

Output: matplotlib Figure objects for individual plots; Path to HTML file
        for report methods.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

# Primary accent colour used throughout charts
_ACCENT = "#128C7E"


def _truncate_label(label: str, n_words: int = 3) -> str:
    """Shorten a slash-separated topic label to the first n_words."""
    parts = [w.strip() for w in label.split("/")]
    short = " / ".join(parts[:n_words])
    return short + ("…" if len(parts) > n_words else "")


def _clean_label(name: str) -> str:
    """
    Strip emoji characters from a display name.

    Returns the original name unchanged when stripping would produce an empty
    string (i.e. the name is composed entirely of emoji).
    """
    try:
        import emoji
        cleaned = emoji.replace_emoji(name, replace="").strip()
        return cleaned if cleaned else name
    except ImportError:
        cleaned = name.encode("ascii", errors="ignore").decode("ascii").strip()
        return cleaned if cleaned else name


_COMPARISON_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Group Comparison Report</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --teal:       #128C7E;
      --teal-dark:  #075E54;
      --teal-light: #25D366;
      --bg:         #F0F2F5;
      --card:       #FFFFFF;
      --text:       #111B21;
      --muted:      #667781;
      --border:     #E9EDEF;
      --radius:     12px;
      --shadow:     0 2px 8px rgba(0,0,0,.08);
    }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                        Helvetica, Arial, sans-serif;
           background: var(--bg); color: var(--text); line-height: 1.5; }
    .header { background: linear-gradient(135deg, var(--teal-dark) 0%, var(--teal) 100%);
              color: #fff; padding: 2.5rem 2rem 2rem; }
    .header-inner { max-width: 1200px; margin: 0 auto; }
    .header .brand { font-size: .8rem; letter-spacing: .12em; text-transform: uppercase;
                     opacity: .75; margin-bottom: .6rem; }
    .header h1 { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .header .subtitle { margin-top: .4rem; opacity: .8; font-size: .95rem; }
    .header .groups-list { margin-top: .75rem; display: flex; flex-wrap: wrap; gap: .5rem; }
    .group-pill { background: rgba(255,255,255,.18); border-radius: 20px;
                  padding: .25rem .75rem; font-size: .85rem; font-weight: 500; }
    .container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 3rem; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                  gap: 1rem; margin-bottom: 2.5rem; }
    .stat-card { background: var(--card); border-radius: var(--radius);
                 box-shadow: var(--shadow); padding: 1.25rem 1.5rem;
                 border-left: 4px solid var(--teal); }
    .stat-card .stat-value { font-size: 1.75rem; font-weight: 700;
                              color: var(--teal-dark); line-height: 1; }
    .stat-card .stat-label { margin-top: .35rem; font-size: .8rem;
                              text-transform: uppercase; letter-spacing: .08em;
                              color: var(--muted); }
    .chart-section { background: var(--card); border-radius: var(--radius);
                     box-shadow: var(--shadow); padding: 1.75rem; margin-bottom: 1.5rem; }
    .chart-section h2 { font-size: 1.05rem; font-weight: 600; color: var(--teal-dark);
                        border-bottom: 2px solid var(--border); padding-bottom: .6rem;
                        margin-bottom: 1.25rem; }
    .chart-section .description { font-size: .88rem; color: var(--muted); margin-bottom: 1rem; }
    .chart-section img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    thead th { background: var(--teal-dark); color: #fff; padding: .6rem 1rem;
               text-align: left; font-weight: 600; }
    tbody tr:nth-child(even) { background: var(--bg); }
    tbody td { padding: .55rem 1rem; border-bottom: 1px solid var(--border); }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .badge-positive { color: #0b7a3e; font-weight: 600; }
    .badge-negative { color: #c0392b; font-weight: 600; }
    .group-section { background: var(--card); border-radius: var(--radius);
                     box-shadow: var(--shadow); padding: 1.75rem; margin-bottom: 1.5rem; }
    .group-heading { display: flex; align-items: center; flex-wrap: wrap; gap: .75rem;
                     border-bottom: 2px solid var(--border); padding-bottom: .75rem;
                     margin-bottom: 1.25rem; }
    .group-heading h2 { font-size: 1.1rem; font-weight: 700; color: var(--teal-dark); }
    .group-mini-stats { display: flex; flex-wrap: wrap; gap: .5rem; margin-left: auto; }
    .mini-pill { background: var(--bg); border-radius: 20px; padding: .2rem .65rem;
                 font-size: .8rem; color: var(--muted); }
    .group-charts-grid { display: grid;
                         grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
                         gap: 1.25rem; }
    .group-chart h3 { font-size: .9rem; color: var(--muted); margin-bottom: .6rem; }
    .group-chart img { display: block; width: 100%; height: auto;
                       border: 1px solid var(--border); border-radius: 8px; }
    footer { text-align: center; padding: 1.5rem; color: var(--muted);
             font-size: .82rem; border-top: 1px solid var(--border); margin-top: 1rem; }
    @media (max-width: 600px) {
      .header h1 { font-size: 1.4rem; }
      .stat-card .stat-value { font-size: 1.4rem; }
      .group-mini-stats { margin-left: 0; }
    }
  </style>
</head>
<body>

<header class="header">
  <div class="header-inner">
    <div class="brand">WhatsApp Analyser &mdash; Comparison</div>
    <h1>Group Comparison Report</h1>
    <p class="subtitle">{{ generated_at }}</p>
    <div class="groups-list">
      {%- for g in group_names %}
      <span class="group-pill">{{ g }}</span>
      {%- endfor %}
    </div>
  </div>
</header>

<div class="container">

  {%- if stat_cards %}
  <div class="stats-grid">
    {%- for card in stat_cards %}
    <div class="stat-card">
      <div class="stat-value">{{ card.value }}</div>
      <div class="stat-label">{{ card.label }}</div>
    </div>
    {%- endfor %}
  </div>
  {%- endif %}

  {%- if activity_rows %}
  <section class="chart-section">
    <h2>Activity Summary</h2>
    <p class="description">Key engagement metrics for each group.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Group</th>
            <th>Messages</th>
            <th>Participants</th>
            <th>Msgs / day</th>
            <th>Start</th>
            <th>End</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>
          {%- for row in activity_rows %}
          <tr>
            <td><strong>{{ row.group }}</strong></td>
            <td class="num">{{ row.nb_messages }}</td>
            <td class="num">{{ row.nb_participants }}</td>
            <td class="num">{{ row.msgs_per_day }}</td>
            <td>{{ row.period_start }}</td>
            <td>{{ row.period_end }}</td>
            <td class="num">{{ row.duration_days }} days</td>
          </tr>
          {%- endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {%- endif %}

  {%- for section in sections %}
  <section class="chart-section">
    <h2>{{ section.title }}</h2>
    {%- if section.description %}
    <p class="description">{{ section.description }}</p>
    {%- endif %}
    <img src="data:image/png;base64,{{ section.image }}" alt="{{ section.title }}">
  </section>
  {%- endfor %}

  {%- for group in group_sections %}
  <section class="group-section">
    <div class="group-heading">
      <h2>{{ group.name }}</h2>
      <div class="group-mini-stats">
        {%- if group.nb_messages %}<span class="mini-pill">{{ group.nb_messages }} messages</span>{%- endif %}
        {%- if group.nb_participants %}<span class="mini-pill">{{ group.nb_participants }} participants</span>{%- endif %}
        {%- if group.period %}<span class="mini-pill">{{ group.period }}</span>{%- endif %}
      </div>
    </div>
    <div class="group-charts-grid">
      {%- for chart in group.charts %}
      <div class="group-chart">
        <h3>{{ chart.title }}</h3>
        <img src="data:image/png;base64,{{ chart.image }}" alt="{{ chart.title }}">
      </div>
      {%- endfor %}
    </div>
  </section>
  {%- endfor %}

  {%- if common_users %}
  <section class="chart-section">
    <h2>Common Participants ({{ common_users | length }})</h2>
    <p class="description">Members present in more than one of the compared groups.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Participant</th><th>Groups</th><th># Groups</th></tr>
        </thead>
        <tbody>
          {%- for row in common_users %}
          <tr>
            <td>{{ row.author }}</td>
            <td>{{ row.groups }}</td>
            <td class="num">{{ row.n_groups }}</td>
          </tr>
          {%- endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {%- endif %}

</div>

<footer>Generated by whatsapp-analyzer &nbsp;·&nbsp; {{ generated_at }}</footer>

</body>
</html>
"""


def _fig_to_base64(fig: Any) -> str:
    """Encode a matplotlib Figure as a base64 PNG string and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


# Self-contained HTML report — no external CDN, fully responsive.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ group_name }} — WhatsApp Analysis</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --teal:       #128C7E;
      --teal-dark:  #075E54;
      --teal-light: #25D366;
      --bg:         #F0F2F5;
      --card:       #FFFFFF;
      --text:       #111B21;
      --muted:      #667781;
      --border:     #E9EDEF;
      --radius:     12px;
      --shadow:     0 2px 8px rgba(0,0,0,.08);
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }

    /* ── Header ─────────────────────────────────────────── */
    .header {
      background: linear-gradient(135deg, var(--teal-dark) 0%, var(--teal) 100%);
      color: #fff;
      padding: 2.5rem 2rem 2rem;
    }
    .header-inner {
      max-width: 1100px;
      margin: 0 auto;
    }
    .header .brand {
      font-size: .8rem;
      letter-spacing: .12em;
      text-transform: uppercase;
      opacity: .75;
      margin-bottom: .6rem;
    }
    .header h1 {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.2;
    }
    .header .subtitle {
      margin-top: .4rem;
      opacity: .8;
      font-size: .95rem;
    }

    /* ── Main container ──────────────────────────────────── */
    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1.5rem 3rem;
    }

    /* ── Stat cards ──────────────────────────────────────── */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2.5rem;
    }
    .stat-card {
      background: var(--card);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 1.25rem 1.5rem;
      border-left: 4px solid var(--teal);
    }
    .stat-card .stat-value {
      font-size: 1.75rem;
      font-weight: 700;
      color: var(--teal-dark);
      line-height: 1;
    }
    .stat-card .stat-label {
      margin-top: .35rem;
      font-size: .8rem;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
    }

    /* ── Section cards ───────────────────────────────────── */
    .chart-section {
      background: var(--card);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 1.75rem;
      margin-bottom: 1.5rem;
    }
    .chart-section h2 {
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--teal-dark);
      border-bottom: 2px solid var(--border);
      padding-bottom: .6rem;
      margin-bottom: 1.25rem;
    }
    .chart-section .description {
      font-size: .88rem;
      color: var(--muted);
      margin-bottom: 1rem;
    }
    .chart-section img {
      display: block;
      max-width: 100%;
      height: auto;
      margin: 0 auto;
    }

    /* ── Word-cloud grid ─────────────────────────────────── */
    .wc-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 1.25rem;
    }
    .wc-item h3 {
      font-size: .9rem;
      color: var(--muted);
      margin-bottom: .6rem;
    }
    .wc-item img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
    }

    /* ── Participant table ───────────────────────────────── */
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: .9rem;
    }
    thead th {
      background: var(--teal-dark);
      color: #fff;
      padding: .6rem 1rem;
      text-align: left;
      font-weight: 600;
    }
    tbody tr:nth-child(even) { background: var(--bg); }
    tbody td {
      padding: .55rem 1rem;
      border-bottom: 1px solid var(--border);
    }

    /* ── Footer ──────────────────────────────────────────── */
    footer {
      text-align: center;
      padding: 1.5rem;
      color: var(--muted);
      font-size: .82rem;
      border-top: 1px solid var(--border);
      margin-top: 1rem;
    }

    @media (max-width: 600px) {
      .header h1 { font-size: 1.4rem; }
      .stat-card .stat-value { font-size: 1.4rem; }
    }
  </style>
</head>
<body>

<header class="header">
  <div class="header-inner">
    <div class="brand">WhatsApp Analyser</div>
    <h1>{{ group_name }}</h1>
    <p class="subtitle">Conversation analysis report &mdash; {{ generated_at }}</p>
  </div>
</header>

<div class="container">

  {%- if stat_cards %}
  <div class="stats-grid">
    {%- for card in stat_cards %}
    <div class="stat-card">
      <div class="stat-value">{{ card.value }}</div>
      <div class="stat-label">{{ card.label }}</div>
    </div>
    {%- endfor %}
  </div>
  {%- endif %}

  {%- if top_authors %}
  <section class="chart-section">
    <h2>Top Participants</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Participant</th>
            <th>Messages</th>
            <th>Share</th>
          </tr>
        </thead>
        <tbody>
          {%- for row in top_authors %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ row.author }}</td>
            <td>{{ row.count }}</td>
            <td>{{ row.pct }}%</td>
          </tr>
          {%- endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {%- endif %}

  {%- for section in sections %}
  <section class="chart-section">
    <h2>{{ section.title }}</h2>
    {%- if section.description %}
    <p class="description">{{ section.description }}</p>
    {%- endif %}
    <img src="data:image/png;base64,{{ section.image }}" alt="{{ section.title }}">
  </section>
  {%- endfor %}

  {%- if wordclouds %}
  <section class="chart-section">
    <h2>Topic Word Clouds</h2>
    <p class="description">Each cloud shows the most representative words for that topic.</p>
    <div class="wc-grid">
      {%- for wc in wordclouds %}
      <div class="wc-item">
        <h3>{{ wc.title }}</h3>
        <img src="data:image/png;base64,{{ wc.image }}" alt="{{ wc.title }}">
      </div>
      {%- endfor %}
    </div>
  </section>
  {%- endif %}

</div>

<footer>Generated by whatsapp-analyzer &nbsp;·&nbsp; {{ generated_at }}</footer>

</body>
</html>
"""


class Visualizer:
    """Generate charts and self-contained HTML reports from pipeline results."""

    def plot_topic_distribution(self, results: dict) -> "Figure":
        """
        Return a horizontal bar chart of topic weights.

        Args:
            results: Pipeline results dict (must contain 'topics').

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        topics_result = results.get("topics")

        if topics_result is None:
            ax.text(0.5, 0.5, "No topic data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=12)
            ax.axis("off")
            return fig

        gt = topics_result["group_topics"].sort_values("weight", ascending=True)
        bars = ax.barh(gt["topic_label"], gt["weight"], color=_ACCENT, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color="gray")
        ax.set_xlabel("Weight", fontsize=10)
        ax.set_title("Topic Distribution", fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_wordcloud(self, results: dict, topic_id: int = 0) -> "Figure":
        """
        Return a word cloud (or bar chart fallback) for a single topic.

        Args:
            results:  Pipeline results dict (must contain 'topics').
            topic_id: Which topic to render.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")

        topics_result = results.get("topics")
        if topics_result is None:
            ax.text(0.5, 0.5, "No topic data available",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        gt = topics_result["group_topics"]
        row = gt[gt["topic_id"] == topic_id]
        if row.empty:
            ax.text(0.5, 0.5, f"Topic {topic_id} not found",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        label = row.iloc[0]["topic_label"]
        words = [w.strip() for w in label.split(" / ")]
        word_freq = {w: max(1, len(words) - i) for i, w in enumerate(words)}

        try:
            from wordcloud import WordCloud
            wc = WordCloud(
                width=800, height=400,
                background_color="white",
                colormap="GnBu",
                prefer_horizontal=0.9,
            )
            wc.generate_from_frequencies(word_freq)
            ax.imshow(wc, interpolation="bilinear")
        except ImportError:
            logger.warning("wordcloud not installed; rendering bar chart instead.")
            ax.axis("on")
            ax.bar(list(word_freq.keys()), list(word_freq.values()),
                   color=_ACCENT, alpha=0.85)
            ax.spines[["top", "right"]].set_visible(False)

        ax.set_title(f"Topic {topic_id}: {label}", fontsize=11, pad=8)
        fig.tight_layout()
        return fig

    def plot_sentiment_timeline(self, results: dict) -> "Figure":
        """
        Return a line chart of mean daily sentiment over time.

        Args:
            results: Pipeline results dict (must contain 'sentiment').

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        sentiment_result = results.get("sentiment")

        if sentiment_result is None:
            ax.text(0.5, 0.5, "No sentiment data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=12)
            ax.axis("off")
            return fig

        df = sentiment_result["df"].copy()
        df = df.sort_values("timestamp")
        daily = df.groupby(df["timestamp"].dt.normalize())["sentiment_score"].mean()

        ax.plot(daily.index, daily.values, linewidth=2, color=_ACCENT)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(daily.index, daily.values, 0,
                        where=(daily.values >= 0), alpha=0.25, color="#25D366",
                        label="Positive")
        ax.fill_between(daily.index, daily.values, 0,
                        where=(daily.values < 0), alpha=0.25, color="#E74C3C",
                        label="Negative")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Mean Sentiment Score", fontsize=10)
        ax.set_title("Sentiment Over Time", fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_user_activity(self, results: dict, top_n: int = 50) -> "Figure":
        """
        Return a horizontal bar chart of message count per author.

        Only the top ``top_n`` participants by message count are shown so the
        chart remains readable even for large groups.  Emoji are stripped from
        author name labels (display only).

        Args:
            results: Pipeline results dict (must contain 'df_clean').
            top_n:   Maximum number of participants to display (default 50).

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        df = results.get("df_clean")

        if df is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No message data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=12)
            ax.axis("off")
            return fig

        # Keep only the top_n most active participants
        counts = df["author"].value_counts()
        total_participants = len(counts)
        if total_participants > top_n:
            counts = counts.head(top_n)
        counts = counts.sort_values(ascending=True)
        display_names = counts.index.map(_clean_label)

        fig_height = max(4, min(len(counts) * 0.38, 22))
        fig, ax = plt.subplots(figsize=(10, fig_height))

        bars = ax.barh(display_names, counts.values, color=_ACCENT, alpha=0.85)
        ax.bar_label(bars, padding=4, fontsize=9, color="gray")
        ax.set_xlabel("Number of messages", fontsize=10)
        title = (
            "Messages per Participant"
            if total_participants <= top_n
            else f"Top {top_n} Participants by Message Count"
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_hourly_heatmap(self, results: dict) -> "Figure":
        """
        Return a heatmap of activity by weekday and hour.

        Args:
            results: Pipeline results dict (must contain 'temporal').

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(16, 5))
        temporal = results.get("temporal")

        if temporal is None:
            ax.text(0.5, 0.5, "No temporal data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=12)
            ax.axis("off")
            return fig

        sns.heatmap(
            temporal["hourly_heatmap"], ax=ax,
            cmap="YlOrRd", linewidths=0.2, linecolor="#f5f5f5",
            cbar_kws={"label": "Messages", "shrink": 0.7},
        )
        ax.set_title("Activity Heatmap — Day × Hour",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Hour of day", fontsize=10)
        ax.set_ylabel("Day of week", fontsize=10)
        fig.tight_layout()
        return fig

    def generate_report(self, results: dict, output_dir: Path) -> Path:
        """
        Generate a self-contained HTML report with all charts embedded as base64 PNG.

        Args:
            results:    Pipeline results dict.
            output_dir: Directory where report.html will be written.

        Returns:
            Path to the written report.html file.
        """
        import datetime

        from jinja2 import Template

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        group_name = results.get("group_name", "Unknown Group")
        generated_at = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

        # Stat cards
        stat_cards = self._build_stat_cards(results)

        # Top-participants table
        top_authors = self._build_top_authors(results)

        # Main charts (exclude word clouds — those go in the grid below)
        main_plots: list[tuple[str, str, Any]] = [
            ("Messages per Participant",
             "Number of messages sent by each participant.",
             self.plot_user_activity(results)),
            ("Hourly Activity Heatmap",
             "When is the group most active? Each cell shows the number of messages.",
             self.plot_hourly_heatmap(results)),
            ("Topic Distribution",
             "Relative weight of each extracted topic across the entire conversation.",
             self.plot_topic_distribution(results)),
            ("Sentiment Over Time",
             "Rolling mean sentiment score — positive values indicate upbeat exchanges.",
             self.plot_sentiment_timeline(results)),
        ]

        sections = [
            {"title": title, "description": desc, "image": _fig_to_base64(fig)}
            for title, desc, fig in main_plots
        ]

        # Word clouds — one per topic, displayed in a responsive grid
        wordclouds = []
        topics_result = results.get("topics")
        if topics_result is not None:
            for tid in topics_result["group_topics"]["topic_id"]:
                fig = self.plot_wordcloud(results, topic_id=int(tid))
                wordclouds.append({
                    "title": f"Topic {int(tid)}",
                    "image": _fig_to_base64(fig),
                })

        html = Template(_HTML_TEMPLATE).render(
            group_name=group_name,
            generated_at=generated_at,
            stat_cards=stat_cards,
            top_authors=top_authors,
            sections=sections,
            wordclouds=wordclouds,
        )

        output_path = output_dir / "report.html"
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path

    def generate_comparison_report(self, comparison_data: dict, output_dir: Path) -> Path:
        """
        Generate a self-contained HTML comparison report for multiple groups.

        Args:
            comparison_data: Dict produced by GroupComparator.report() containing
                             keys: analyzers, activity, topics, sentiment, common_users.
            output_dir:      Directory where comparison_report.html will be written.

        Returns:
            Path to the written comparison_report.html file.
        """
        import datetime

        import pandas as pd
        from jinja2 import Template

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_at = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

        analyzers     = comparison_data.get("analyzers", [])
        activity_df   = comparison_data.get("activity",     pd.DataFrame())
        topics_df     = comparison_data.get("topics",       pd.DataFrame())
        sentiment_df  = comparison_data.get("sentiment",    pd.DataFrame())
        common_u_df   = comparison_data.get("common_users", pd.DataFrame(columns=["author", "groups"]))

        if activity_df is None:
            activity_df = pd.DataFrame()
        if topics_df is None:
            topics_df = pd.DataFrame()
        if sentiment_df is None:
            sentiment_df = pd.DataFrame()
        if common_u_df is None:
            common_u_df = pd.DataFrame(columns=["author", "groups"])

        group_names = [
            az._results.get("group_name", f"Group {i + 1}")
            for i, az in enumerate(analyzers)
        ]

        # Stat cards
        stat_cards = self._build_comparison_stat_cards(activity_df, len(analyzers))

        # Activity table
        activity_rows = self._build_activity_table_rows(activity_df)

        # Comparison-level charts
        sections = []

        fig = self._plot_activity_overview(activity_df)
        if fig is not None:
            sections.append({
                "title": "Activity Overview",
                "description": "Total messages and daily message rate for each group.",
                "image": _fig_to_base64(fig),
            })

        if not sentiment_df.empty:
            fig = self._plot_sentiment_comparison(sentiment_df)
            if fig is not None:
                sections.append({
                    "title": "Sentiment Comparison",
                    "description": (
                        "Distribution of positive, neutral, and negative messages "
                        "per group. Neutral messages sit between the two thresholds."
                    ),
                    "image": _fig_to_base64(fig),
                })

        if not topics_df.empty:
            fig = self._plot_topic_heatmap(topics_df)
            if fig is not None:
                sections.append({
                    "title": "Topic Distribution Heatmap",
                    "description": (
                        "Topic weight per group — darker cells indicate stronger presence "
                        "of that topic in the conversation."
                    ),
                    "image": _fig_to_base64(fig),
                })

        # Per-group detail sections
        group_sections = []
        for az in analyzers:
            results  = az._results
            gname    = results.get("group_name", "Unknown")
            df_clean = results.get("df_clean")

            mini = {}
            if df_clean is not None and not df_clean.empty:
                mini["nb_messages"]    = f"{len(df_clean):,}"
                mini["nb_participants"] = str(df_clean["author"].nunique())
                ts = df_clean["timestamp"].dropna()
                if not ts.empty:
                    mini["period"] = (
                        f"{ts.min().strftime('%d %b %Y')} → {ts.max().strftime('%d %b %Y')}"
                    )

            charts = []
            fig_p = self.plot_user_activity(results, top_n=15)
            charts.append({"title": "Top 15 Participants", "image": _fig_to_base64(fig_p)})

            if results.get("temporal") is not None:
                fig_h = self.plot_hourly_heatmap(results)
                charts.append({"title": "Activity Heatmap", "image": _fig_to_base64(fig_h)})

            if results.get("sentiment") is not None:
                fig_s = self.plot_sentiment_timeline(results)
                charts.append({"title": "Sentiment Over Time", "image": _fig_to_base64(fig_s)})

            group_sections.append({
                "name":            gname,
                "nb_messages":     mini.get("nb_messages", ""),
                "nb_participants": mini.get("nb_participants", ""),
                "period":          mini.get("period", ""),
                "charts":          charts,
            })

        # Common participants
        common_users_rows = []
        if not common_u_df.empty:
            for _, row in common_u_df.iterrows():
                grps = row["groups"]
                common_users_rows.append({
                    "author":   _clean_label(str(row["author"])),
                    "groups":   ", ".join(grps) if isinstance(grps, list) else str(grps),
                    "n_groups": len(grps) if isinstance(grps, list) else 1,
                })
            common_users_rows.sort(key=lambda r: r["n_groups"], reverse=True)

        html = Template(_COMPARISON_HTML_TEMPLATE).render(
            generated_at=generated_at,
            group_names=group_names,
            stat_cards=stat_cards,
            activity_rows=activity_rows,
            sections=sections,
            group_sections=group_sections,
            common_users=common_users_rows,
        )

        output_path = output_dir / "comparison_report.html"
        output_path.write_text(html, encoding="utf-8")
        logger.info("Comparison report written to %s", output_path)
        return output_path

    # Private helpers

    @staticmethod
    def _build_stat_cards(results: dict) -> list[dict]:
        """Build summary stat-card data from pipeline results."""
        cards: list[dict] = []

        df_raw = results.get("df_raw")
        df_clean = results.get("df_clean")
        df = df_raw if df_raw is not None else df_clean

        if df is not None and not df.empty:
            cards.append({"value": f"{len(df):,}", "label": "Total messages"})

            if "author" in df.columns:
                n_participants = df["author"].nunique()
                cards.append({"value": str(n_participants), "label": "Participants"})

            if "timestamp" in df.columns:
                ts = df["timestamp"].dropna()
                if not ts.empty:
                    first = ts.min()
                    last = ts.max()
                    days = max(1, (last - first).days)
                    cards.append({
                        "value": first.strftime("%d %b %Y"),
                        "label": "First message",
                    })
                    cards.append({"value": f"{days:,}", "label": "Days of history"})

        return cards

    @staticmethod
    def _build_top_authors(results: dict, top_n: int = 10) -> list[dict]:
        """Return a list of dicts for the top-participants table."""
        _df_raw = results.get("df_raw")
        df = _df_raw if _df_raw is not None else results.get("df_clean")
        if df is None or "author" not in df.columns:
            return []

        counts = df["author"].value_counts().head(top_n)
        total = counts.sum()
        rows = []
        for author, count in counts.items():
            rows.append({
                "author": _clean_label(author),
                "count": f"{count:,}",
                "pct": f"{100 * count / total:.1f}",
            })
        return rows

    # Comparison chart helpers

    @staticmethod
    def _build_comparison_stat_cards(activity_df, n_groups: int) -> list[dict]:
        """Build summary stat cards for the comparison report header."""
        import pandas as pd

        cards = [{"value": str(n_groups), "label": "Groups compared"}]
        if activity_df is None or activity_df.empty:
            return cards
        total_msgs = activity_df["nb_messages"].sum()
        cards.append({"value": f"{total_msgs:,}", "label": "Total messages (all groups)"})
        total_p = activity_df["nb_participants"].sum()
        cards.append({"value": f"{total_p:,}", "label": "Participants (combined)"})
        starts = activity_df["period_start"].dropna()
        if not starts.empty:
            earliest = starts.min()
            if hasattr(earliest, "strftime"):
                cards.append({"value": earliest.strftime("%d %b %Y"), "label": "Earliest message"})
        return cards

    @staticmethod
    def _build_activity_table_rows(activity_df) -> list[dict]:
        """Return serialisable rows for the activity summary table."""
        import pandas as pd

        if activity_df is None or activity_df.empty:
            return []
        rows = []
        for group, row in activity_df.iterrows():
            start = row["period_start"]
            end   = row["period_end"]
            start_str = start.strftime("%d %b %Y") if hasattr(start, "strftime") else str(start)
            end_str   = end.strftime("%d %b %Y")   if hasattr(end,   "strftime") else str(end)
            days = max(1, (end - start).days) if hasattr(start, "days") else "—"
            rows.append({
                "group":          str(group),
                "nb_messages":    f"{int(row['nb_messages']):,}",
                "nb_participants": str(int(row["nb_participants"])),
                "msgs_per_day":   f"{row['msgs_per_day']:.1f}",
                "period_start":   start_str,
                "period_end":     end_str,
                "duration_days":  f"{days:,}" if isinstance(days, int) else days,
            })
        return rows

    def _plot_activity_overview(self, activity_df) -> "Figure | None":
        """Two-panel bar chart: total messages (left) and msgs/day (right)."""
        import matplotlib.pyplot as plt

        if activity_df is None or activity_df.empty:
            return None

        groups = [str(g) for g in activity_df.index.tolist()]
        n      = len(groups)
        colors = [_ACCENT, "#34B7F1"]

        fig, axes = plt.subplots(1, 2, figsize=(14, max(3, n * 0.55 + 1.5)))

        for ax, col, title, color, fmt in [
            (axes[0], "nb_messages",  "Total Messages",    colors[0], "%d"),
            (axes[1], "msgs_per_day", "Messages per Day",  colors[1], "%.1f"),
        ]:
            vals = activity_df[col].values
            bars = ax.barh(groups, vals, color=color, alpha=0.85)
            ax.bar_label(bars, fmt=fmt, padding=4, fontsize=9, color="gray")
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=9)

        fig.tight_layout(pad=2.5)
        return fig

    def _plot_sentiment_comparison(self, sentiment_df) -> "Figure | None":
        """Grouped bar chart: positive / neutral / negative % per group."""
        import matplotlib.pyplot as plt

        if sentiment_df is None or sentiment_df.empty:
            return None

        groups = [str(g) for g in sentiment_df.index.tolist()]
        n      = len(groups)
        x      = list(range(n))
        w      = 0.25

        pos = sentiment_df["pos_pct"].values * 100
        neg = sentiment_df["neg_pct"].values * 100
        neu = 100 - pos - neg

        fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 5))
        ax.bar([i - w for i in x], pos, w, label="Positive", color="#25D366", alpha=0.85)
        ax.bar([i     for i in x], neu, w, label="Neutral",  color="#B2BEC3", alpha=0.85)
        ax.bar([i + w for i in x], neg, w, label="Negative", color="#E74C3C", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.set_title("Sentiment Distribution per Group",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    def _plot_topic_heatmap(self, topics_df) -> "Figure | None":
        """Seaborn heatmap: groups as rows, truncated topic labels as columns."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if topics_df is None or topics_df.empty:
            return None

        short_cols = {c: _truncate_label(c) for c in topics_df.columns}
        plot_df    = topics_df.rename(columns=short_cols)

        n_groups = len(plot_df)
        n_topics = len(plot_df.columns)
        fig_h    = max(2.5, n_groups * 0.9 + 1.5)
        fig_w    = max(10,  n_topics * 2.2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            plot_df, ax=ax,
            cmap="YlOrRd", annot=True, fmt=".3f",
            linewidths=0.4, linecolor="#f0f0f0",
            cbar_kws={"label": "Weight", "shrink": 0.7},
        )
        ax.set_title("Topic Weight Heatmap",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Topic", fontsize=10)
        ax.set_ylabel("Group", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=9)
        fig.tight_layout()
        return fig
