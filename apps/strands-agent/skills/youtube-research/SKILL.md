---
name: youtube-research
description: >
  YouTube channel and content discovery methodology for research corpus building.
  Uses TranscriptAPI as the primary intelligence source — searching directly
  inside YouTube video transcripts to find which channels actually contain
  content on target topics. Extends the OSINT three-question framework:
  (1) What is the maximum set of channels covering these topics?
  (2) How do I discriminate which have the deepest, most unique content?
  (3) Where exactly is the knowledge (which videos, which timestamps)?
  Ground-truth-first: searches actual video content, not web opinions about
  channels. Produces quantified channel rankings backed by search evidence.
allowed-tools: >
  search_youtube search_channel_videos get_youtube_transcript
  get_channel_latest_videos list_channel_videos list_playlist_videos
  duckduckgo_search stract_search brave_web_search brave_video_search
  mojeek_search yandex_search jina_read_url firecrawl_scrape
  reddit_search reddit_get_subreddit_posts reddit_get_post_details
  grok_deep_research perplexity_deep_research tavily_deep_research
  exa_multi_search kagi_search kagi_enrich_web
  store_finding read_findings add_entity add_edge query_graph find_gaps
---

# YouTube Research — Ground-Truth Channel & Content Discovery

You MUST follow this protocol for any query about YouTube channels, YouTube
content discovery, or building research corpora from YouTube videos.
Do not shortcut. The goal is **GROUND-TRUTH discovery** — finding channels
based on what they ACTUALLY say in their videos, not what search engines
or LLMs think about them.

This methodology extends the OSINT exhaustive discovery protocol with
YouTube-specific intelligence tools. The same three core questions apply,
but the primary intelligence source is **direct transcript search**.

**QUALITY BAR**: The output must be better than what someone could find by
Googling "best YouTube channels for X." It must surface channels that web
search misses — smaller creators who produce deep content but lack SEO,
foreign-language channels, channels that cover the topic in a few key
videos rather than as their main focus.

**CRITICAL PRINCIPLE — GROUND TRUTH OVER OPINION**: TranscriptAPI lets you
search INSIDE actual YouTube video content. A channel that has 47 videos
mentioning "insulin protocol" in their transcripts is a better signal than
a blog post saying "this channel covers insulin." Always prefer transcript
search evidence over web search opinions.

---

## TOOL TIERS — YOUTUBE INTELLIGENCE SOURCING

### Tier 0 — Direct Content Search (ALWAYS START HERE)

TranscriptAPI tools search inside actual YouTube video transcripts and
metadata. This is your PRIMARY intelligence source for any YouTube query.

**Tools:**
- `search_youtube` — Search across ALL of YouTube for videos matching a
  query. Returns video titles, channels, descriptions, view counts. This
  is your most powerful discovery tool. Use it to find which channels
  produce content on specific topics.
- `search_channel_videos` — Search WITHIN a specific channel for videos
  matching a query. Use this to assess how deeply a channel covers a topic.
- `get_channel_latest_videos` — List recent videos from a channel. Use to
  assess current activity and topic focus.
- `list_channel_videos` — Browse all videos in a channel. Use for
  comprehensive channel assessment.
- `list_playlist_videos` — Browse playlist contents. Some channels organize
  content by topic in playlists.

**Why Tier 0 is special:**
- Returns ground truth: which channels ACTUALLY contain this content
- Quantifiable: count videos per channel per topic = coverage score
- Discovers unknown channels: a channel with 200 subscribers but 30 deep
  insulin protocol videos is more valuable than a 2M-sub channel that
  mentioned insulin once
- Language-agnostic: finds Russian, German, Portuguese channels that web
  search in English would never surface
- Current: reflects what's actually on YouTube right now, not what an LLM
  was trained on

**How to use for each core question:**
- **Q1 (maximum set)**: Run `search_youtube` for each target topic keyword.
  Collect ALL unique channel names from results. This is your raw channel
  universe.
- **Q2 (discrimination)**: For each channel in the universe, count how many
  topic searches it appeared in. Channels appearing across multiple topics
  have broader coverage. Use `search_channel_videos` to assess depth within
  each topic.
- **Q3 (sources)**: The search results themselves ARE the source map — each
  result links to a specific video where the topic is discussed.

### Tier 1 — Community Intelligence (forums, Reddit, comments)

Real people's opinions about YouTube channels are HIGH-VALUE intelligence.
They tell you things transcript search cannot: which channels are trusted
by practitioners, which have been caught spreading misinformation, which
have the best comment sections for anecdotal data.

**Tools:**
- `reddit_search` / `reddit_get_subreddit_posts` — Search subreddits for
  channel recommendations. r/steroids, r/PEDs, r/bodybuilding, r/fitness
  have active discussions about which YouTubers to trust.
- `grok_deep_research` — X/Twitter discussions about YouTube channels.
  Community sentiment, callouts, endorsements.
- `duckduckgo_search` with `site:reddit.com` — Reddit via web index.

**Why this tier matters for YouTube research:**
- Reveals trust signals that transcript content alone cannot show
- Surfaces "hidden gem" channels that the community recommends but that
  don't rank well in search
- Identifies channels to AVOID (misinformation, outdated advice)
- Comments on YouTube videos themselves are gold — when harvesting is
  available, prioritize channels with high-quality comment discussions

### Tier 2 — Web Search (opinion aggregation)

Web search finds curated lists, blog posts, and articles about YouTube
channels. This is SUPPLEMENTARY — use it to cross-validate and extend
what Tier 0 and Tier 1 found, not as your primary source.

**Tools:**
- `brave_web_search` / `brave_video_search` — Video-specific search
- `duckduckgo_search` — "best YouTube channels for [topic]" lists
- `perplexity_deep_research` — LLM-synthesized channel recommendations
- `kagi_search` / `kagi_enrich_web` — Indie/small-web content about channels
- `exa_multi_search` — Semantic search for channel recommendation articles

**When to use:**
- AFTER Tier 0 has established the ground-truth channel universe
- To discover channels that might use different terminology (so Tier 0
  keyword searches missed them)
- To find expert-curated lists that weight quality, not just keyword count
- To cross-validate: if a channel ranks high in Tier 0 (many topic matches)
  AND high in Tier 2 (recommended by experts), it's a strong signal

### Tier 3 — Deep Content Inspection

Once you've identified promising channels, inspect their actual content
to assess quality and depth.

**Tools:**
- `get_youtube_transcript` — Pull full transcript of a specific video.
  Use to verify that a channel's content is substantive, not clickbait.
- `jina_read_url` / `firecrawl_scrape` — Extract content from channel
  pages, community posts, linked resources.

---

## THE THREE CORE QUESTIONS (YouTube-Adapted)

Before doing ANY searching, explicitly answer these for the user's query:

### Q1 — "What is the maximum set of YouTube channels that could have content on these topics?"

The full universe of channels that might contain relevant content. For
bodybuilding research, this is potentially thousands of channels worldwide.
TranscriptAPI search is the fastest way to discover the actual universe —
every channel that has videos with the target keywords in their transcripts.

### Q2 — "How do I discriminate which channels have the deepest, most unique content?"

Scoring dimensions (all derivable from TranscriptAPI search results):
- **Topic breadth**: How many of the target topics does the channel cover?
  (A channel appearing in 5/6 topic searches > one appearing in 1/6)
- **Topic depth**: How many videos does the channel have on each topic?
  (30 videos on insulin > 2 videos on insulin)
- **Content uniqueness**: Does this channel cover things others don't?
  (Only channel discussing Andrei Smayev = high uniqueness signal)
- **Recency**: Are the videos recent or years old?
- **View count**: Higher views suggest community validation (but low-view
  channels with deep content are valuable discoveries)
- **Language**: Non-English channels may contain unique regional knowledge

### Q3 — "Where exactly is the knowledge?"

The TranscriptAPI search results themselves answer this — each result is
a specific video URL where the topic is discussed. The output should map:
channel → topics covered → specific videos for each topic.

---

## PHASE 0: TOPIC DECOMPOSITION & KEYWORD EXPANSION

**Purpose**: Break the user's research query into searchable keyword sets
that will maximize recall when searching YouTube transcripts.

### Step 1 — Decompose topics into search terms

For each topic the user wants researched, generate 3-5 keyword variations
that people would actually SAY in a YouTube video (not write in an article):

Example for "insulin protocols in bodybuilding":
- "insulin protocol" (direct term)
- "humalog bodybuilding" (brand name + context)
- "insulin dosing pre workout" (practical usage)
- "insulin sensitivity steroids" (interaction)
- "insulin gut growth hormone" (side effect discussion)

Example for "trenbolone":
- "trenbolone" (direct)
- "tren cycle" (community shorthand)
- "tren side effects" (common discussion)
- "trenbolone acetate dosage" (specific variant)
- "19-nortestosterone" (chemical name, for academic channels)

**Why multiple keywords matter**: Different channels use different
terminology. Academic channels say "exogenous growth hormone administration."
Bro-science channels say "pinning GH." Both are valuable — you need
keywords that catch both.

### Step 2 — Plan the search matrix

Create a topic × keyword matrix. You'll run `search_youtube` for each
keyword, then aggregate channels across all results.

---

## PHASE 1: GROUND-TRUTH CHANNEL DISCOVERY (Tier 0)

**Purpose**: Use TranscriptAPI to discover which channels actually contain
content on the target topics. This is the core of the methodology.

### Step 3 — Execute transcript searches

For EACH keyword from Step 1, run `search_youtube`:
- Record every unique channel that appears in results
- Note the video title, view count, and channel name for each result
- Track which keywords each channel appeared for

**CRITICAL**: Do NOT stop after the first search. Run ALL planned keywords.
A channel that only appears for "19-nortestosterone" but not "tren" might
be an academic channel with unique content — you'd miss it if you only
searched "tren."

### Step 4 — Build the channel frequency table

Aggregate results into a structured table:

| Channel | Topics Matched (of N) | Total Video Hits | Top Keywords |
|---------|----------------------|------------------|--------------|
| MPMD    | 6/6                  | 47               | insulin, tren, GH, ... |
| Leo     | 4/6                  | 23               | peptides, GH, harm... |
| ...     | ...                  | ...              | ... |

Channels with the MOST topic matches AND the MOST video hits are your
strongest candidates. But don't dismiss channels with few matches if they
cover a unique topic nobody else does.

### Step 5 — Drill into top candidates

For the top 10-15 channels, use `search_channel_videos` to assess depth:
- Search each channel for each target topic
- Count how many videos they have on each topic
- Note any unique topics they cover that others don't

This converts the raw frequency table into a coverage matrix:

| Channel | Insulin | GH | Tren | Nutrition | Extreme | Smayev |
|---------|---------|-----|------|-----------|---------|--------|
| MPMD    | 12      | 8   | 15   | 20        | 5       | 0      |
| Leo     | 3       | 15  | 2    | 5         | 0       | 0      |

---

## PHASE 2: COMMUNITY VALIDATION (Tier 1)

**Purpose**: Cross-validate Tier 0 findings with community intelligence.
Are the channels Tier 0 surfaced actually trusted by practitioners?

### Step 6 — Reddit and forum sweep

Search relevant subreddits and forums:
- "best YouTube channels [topic]" on r/steroids, r/PEDs, r/bodybuilding
- "who to watch for [topic] information" on relevant communities
- Channel-specific searches: "[channel name] trust" or "[channel name] review"

### Step 7 — X/Twitter sweep

Use `grok_deep_research` to check:
- Community sentiment about top-ranked channels
- Any callouts for misinformation or outdated advice
- Endorsements from known experts

### Step 8 — Reconcile Tier 0 and Tier 1

Channels that rank high in BOTH transcript search AND community trust are
highest priority for harvesting. Flag any discrepancies:
- High in Tier 0, low in Tier 1 → check if the channel is new/unknown
  or if the community has concerns about accuracy
- Low in Tier 0, high in Tier 1 → the channel may use different
  terminology than your keywords caught, or its value is in presentation
  quality rather than keyword density

---

## PHASE 3: WEB SEARCH EXTENSION (Tier 2)

**Purpose**: Find channels that Tier 0 might have missed due to keyword
limitations or channels that don't use standard terminology.

### Step 9 — Curated list discovery

Search the web for existing channel recommendation lists:
- "best bodybuilding science YouTube channels [year]"
- "top PED YouTube channels"
- "[topic] YouTube channel recommendations reddit"

### Step 10 — Cross-reference with Tier 0

For any channel found in Tier 2 that wasn't in Tier 0 results:
- Run `search_channel_videos` to check if they actually have relevant
  content (the recommendation might be outdated or inaccurate)
- If they DO have content, add them to the coverage matrix

---

## PHASE 4: SYNTHESIS & RANKING

**Purpose**: Produce the final ranked channel list with evidence.

### Step 11 — Compute composite scores

For each channel, compute:
- **Breadth score**: topics_matched / total_topics (0.0 - 1.0)
- **Depth score**: total_relevant_videos / max_videos_any_channel (0.0 - 1.0)
- **Uniqueness score**: does this channel cover topics no other channel does?
- **Trust score**: community validation from Tier 1 (high/medium/low/unknown)
- **Composite**: weighted combination (suggested: 30% breadth, 30% depth,
  20% uniqueness, 20% trust)

### Step 12 — Write the final report

For each recommended channel (ranked by composite score):

```
## [Rank]. [Channel Name] (Composite: X.XX)

**Coverage**: [N]/[M] topics | [V] relevant videos found
**Strongest topics**: [list with video counts]
**Unique coverage**: [topics only this channel covers, if any]
**Community trust**: [Tier 1 findings]
**Sample videos**: [2-3 specific video titles with view counts]
**Why harvest this channel**: [1-2 sentences on what unique knowledge
  this channel adds to the corpus that other channels don't]
```

### Step 13 — Identify gaps

After ranking, explicitly note:
- Topics with NO strong channel coverage (content gaps)
- Types of knowledge that YouTube doesn't cover well (e.g., may need
  podcasts, forums, or academic papers instead)
- Channels that were expected but not found (e.g., if a well-known
  channel doesn't actually have the content people assume it does)

---

## EXECUTION CHECKLIST

Before reporting results, verify:

- [ ] TranscriptAPI `search_youtube` was used for EVERY target topic keyword
- [ ] Channel frequency table was built from actual search results
- [ ] Top channels were drilled into with `search_channel_videos`
- [ ] Community validation was performed (Reddit, X/Twitter)
- [ ] Web search was used to find channels Tier 0 might have missed
- [ ] Final ranking includes evidence (video counts, search hits, community signals)
- [ ] Gaps and limitations are explicitly stated
- [ ] Every channel recommendation is backed by transcript search evidence,
      not just "this channel is popular" or "an LLM recommended it"

---

## ANTI-PATTERNS (what NOT to do)

1. **Do NOT start with web search.** Web search returns opinions about
   channels. TranscriptAPI returns ground truth about content. Always
   start with Tier 0.

2. **Do NOT rely on LLM knowledge about channels.** LLM training data is
   months or years old. Channels change focus, new channels emerge, old
   ones go inactive. TranscriptAPI reflects current YouTube state.

3. **Do NOT rank by subscriber count.** A 50K-subscriber channel with 40
   deep insulin protocol videos is more valuable for corpus building than
   a 5M-subscriber channel that mentioned insulin twice.

4. **Do NOT stop after finding "the obvious channels."** The methodology's
   value is in finding channels that simple web search WOULDN'T find —
   smaller creators, foreign-language channels, channels that cover the
   topic in a few key videos rather than as their main focus.

5. **Do NOT present results without evidence.** Every channel recommendation
   must cite specific TranscriptAPI search results (video counts, topic
   matches) — not just "this channel is good for X."
