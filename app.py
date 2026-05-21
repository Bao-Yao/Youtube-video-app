import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
import os
import re
from googleapiclient.discovery import build
from datetime import datetime

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize models
sia = SentimentIntensityAnalyzer()

@st.cache_resource
def load_keybert():
    return KeyBERT()

kw_model = load_keybert()

def extract_video_id(url):
    """
    Extracts the 11-character video ID from various YouTube URL formats.
    """
    if not url:
        return None
    url = url.strip()
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&\s]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([^?\s]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^?\s]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([^?\s]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
            
    # Fallback: Check if the input itself is directly an 11-char YouTube video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    return None

@st.cache_data(ttl=300)
def fetch_live_video_data(api_key, video_id):
    """
    Fetches real-time video details from the official YouTube Data API.
    """
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        
        if not response.get('items'):
            return pd.DataFrame()
            
        item = response['items'][0]
        snippet = item.get('snippet', {})
        stats = item.get('statistics', {})
        
        # Maps to the structure expected by the app
        video_data = {
            'videoId': video_id,
            'publishedAt': snippet.get('publishedAt', 'N/A'),
            'title': snippet.get('title', 'N/A'),
            'channelTitle': snippet.get('channelTitle', 'N/A'),
            'thumbnailUrl': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
            'viewCount': int(stats.get('viewCount', 0)),
            'likeCount': int(stats.get('likeCount', 0)),
            'commentCount': int(stats.get('commentCount', 0))
        }
        return pd.DataFrame([video_data])
    except Exception as e:
        st.error(f"Error fetching live video details: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_comments(api_key, video_id, max_results=200):
    """
    Fetches real-time comments for a video from the official YouTube Data API.
    """
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText"
        )
        
        while request and len(comments) < max_results:
            response = request.execute()
            for item in response.get('items', []):
                top_comment = item.get('snippet', {}).get('topLevelComment', {})
                snippet = top_comment.get('snippet', {})
                comments.append({
                    'videoId': video_id,
                    'textOriginal': snippet.get('textDisplay', ''),
                    'likeCount': int(snippet.get('likeCount', 0)),
                    'publishedAt': snippet.get('publishedAt', '')
                })
            
            if len(comments) < max_results and response.get('nextPageToken'):
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(max_results - len(comments), 100),
                    pageToken=response['nextPageToken'],
                    textFormat="plainText"
                )
            else:
                break
                
        return pd.DataFrame(comments)
    except Exception as e:
        st.error(f"Error fetching live comments: {str(e)}")
        return pd.DataFrame()


# Set Page Configuration
st.set_page_config(layout="wide", page_title="YouTube Marketing Dashboard", page_icon="📈")

# Custom Dark Blue CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0b132b;
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1c2541;
    }
    
    /* Metrics boxes */
    [data-testid="stMetricValue"] {
        color: #4ea8de;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1c2541 !important;
        color: white !important;
    }
    
    /* Cards for grouping */
    .card {
        background-color: #1c2541;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #3a506b;
    }

    /* Style st.container(border=True) to match our premium card design */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1c2541 !important;
        border: 1px solid #3a506b !important;
        border-radius: 10px !important;
        padding: 0 !important;
        margin-bottom: 20px !important;
    }
    
    /* Content wrapper inside container */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 20px !important;
    }
    
    /* Modern Streamlit table overrides */
    div[data-testid="stTable"] table {
        background-color: #1c2541 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a506b !important;
        border-collapse: collapse !important;
        width: 100% !important;
    }
    div[data-testid="stTable"] th {
        background-color: #0b132b !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: 1px solid #3a506b !important;
        padding: 10px !important;
    }
    div[data-testid="stTable"] td {
        background-color: #1c2541 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a506b !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# Removed local dataset loading functions

def analyze_sentiment(text):
    if not isinstance(text, str):
        return "Neutral"
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_grade(metric_name, ratio):
    if metric_name == "Like-to-View":
        # Average is around 4%
        if ratio >= 0.05: return "High 🟢"
        elif ratio >= 0.02: return "Average 🟡"
        else: return "Low 🔴"
    elif metric_name == "Comment-to-View":
        # Average is around 0.5%
        if ratio >= 0.01: return "High 🟢"
        elif ratio >= 0.002: return "Average 🟡"
        else: return "Low 🔴"

    return "N/A"


def render_dashboard_layout(video_info, vid_comments, key_suffix=""):
    # --- TOP SECTION ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        card_html = f"""
        <div class="card" style="height: 100%;">
            <h3 style="margin-top: 0; color: #ffffff; margin-bottom: 15px;">📌 Video Information</h3>
            {f'<img src="{video_info.get("thumbnailUrl")}" style="width:100%; border-radius:10px; margin-bottom:15px;"/>' if video_info.get("thumbnailUrl") else ''}
            <p style="margin: 10px 0; font-size: 1.05rem; line-height: 1.5; color: #e0e0e0;"><b>Title:</b> {video_info.get('title', 'N/A')}</p>
            <p style="margin: 10px 0; font-size: 1.05rem; color: #e0e0e0;"><b>Channel:</b> {video_info.get('channelTitle', 'N/A')}</p>
            <p style="margin: 10px 0; font-size: 1.05rem; color: #e0e0e0;"><b>Published Date:</b> {video_info.get('publishedAt', 'N/A')}</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        views = float(video_info.get('viewCount', 0))
        likes = float(video_info.get('likeCount', 0))
        comments_count = float(video_info.get('commentCount', 0))
        
        metrics_html = f"""
        <div class="card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #ffffff; margin-bottom: 20px;">📈 Basic Metrics</h3>
            <div style="display: grid; grid-template-columns: 1fr; gap: 20px;">
                <div>
                    <div style="font-size: 0.95rem; color: #a0aec0; margin-bottom: 4px;">Views</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #4ea8de; margin: 0;">{views:,.0f}</div>
                </div>
                <div>
                    <div style="font-size: 0.95rem; color: #a0aec0; margin-bottom: 4px;">Likes</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #4ea8de; margin: 0;">{likes:,.0f}</div>
                </div>
                <div>
                    <div style="font-size: 0.95rem; color: #a0aec0; margin-bottom: 4px;">Comments</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #4ea8de; margin: 0;">{comments_count:,.0f}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        with st.container(border=True):
            st.subheader("🎯 YouTube Metrics Tracker")
            
            l2v = likes / views if views > 0 else 0
            c2v = comments_count / views if views > 0 else 0
            
            tracker_data = {
                "Metric": ["Like-to-View Ratio", "Comment-to-View Ratio"],
                "Ratio": [f"{l2v:.2%}", f"{c2v:.2%}"],
                "Grade": [get_grade("Like-to-View", l2v), get_grade("Comment-to-View", c2v)]
            }
            st.table(pd.DataFrame(tracker_data))
    
    st.markdown("---")
    
    # --- MIDDLE SECTION ---
    if not vid_comments.empty:
        if 'Sentiment' not in vid_comments.columns:
            with st.spinner("Analyzing sentiments..."):
                vid_comments = vid_comments.copy()
                vid_comments['Sentiment'] = vid_comments['textOriginal'].apply(analyze_sentiment)
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            with st.container(border=True):
                st.subheader("💬 Comment Sentiment Analysis")
                sentiment_counts = vid_comments['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                             color='Sentiment', 
                             color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'},
                             hole=0.4)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_sentiment_{key_suffix}")
            
        with col4:
            with st.container(border=True):
                st.subheader("🏆 Top 5 Topics Leaderboard")
                with st.spinner("Extracting topics using KeyBERT..."):
                    text_corpus = " ".join(vid_comments['textOriginal'].dropna().tolist())
                    if len(text_corpus) > 50000:
                        text_corpus = text_corpus[:50000]
                        
                    if text_corpus.strip():
                        keywords = kw_model.extract_keywords(text_corpus, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                        if keywords:
                            kw_df = pd.DataFrame(keywords, columns=['Topic / Keyword', 'Relevance Score'])
                            kw_df.index = kw_df.index + 1
                            st.table(kw_df)
                        else:
                            st.write("No topics extracted.")
                    else:
                        st.write("No valid comment text available for topic extraction.")
        
        # --- BOTTOM SECTION ---
        st.markdown("---")
        with st.container(border=True):
            st.subheader("📋 Comment Analysis Dashboard")
            st.write("Detailed breakdown of recent comments.")
            
            display_df = vid_comments[['textOriginal', 'likeCount', 'publishedAt', 'Sentiment']].copy()
            display_df = display_df.rename(columns={
                'textOriginal': 'Comment Text', 
                'likeCount': 'Likes', 
                'publishedAt': 'Published Date'
            })
            display_df = display_df.sort_values(by='Published Date', ascending=False)
            
            st.dataframe(display_df, use_container_width=True, height=400, key=f"df_comments_{key_suffix}")
            
    else:
        st.info("No comments found for this video in the dataset.")


st.title("📊 YouTube Marketing Dashboard")

# Initialize session history
if 'query_history' not in st.session_state:
    st.session_state.query_history = {}
if 'history_page' not in st.session_state:
    st.session_state.history_page = 'list'
if 'selected_history_key' not in st.session_state:
    st.session_state.selected_history_key = None
if 'api_quota_used' not in st.session_state:
    st.session_state.api_quota_used = 0

# --- CLEAR HISTORY DIALOG ---
@st.dialog("🗑️ Clear Search History")
def confirm_clear_dialog():
    st.warning("Are you sure you want to clear all search history? This action cannot be undone.")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, Clear History", use_container_width=True, type="primary"):
            st.session_state.query_history = {}
            st.session_state.history_page = 'list'
            st.session_state.selected_history_key = None
            st.rerun()
    with col_no:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()

# --- SIDEBAR NAVIGATION & DATA LOADING ---
st.sidebar.header("Data Source Settings")
app_page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🕐 History"])
st.sidebar.markdown("---")

help_text = """Calculating quota usage
Google calculates your quota usage by assigning a cost to each request. Different types of operations have different quota costs. For example:

- A read operation that retrieves a list of resources -- channels, videos, playlists -- usually costs 1 unit.
- A write operation that creates, updates, or deletes a resource usually costs 50 units.
- A search request costs 100 units.
- A video upload costs 100 units."""

st.sidebar.markdown(f"**API Quota Used (Approx)**", help=help_text)
quota_used = st.session_state.api_quota_used
progress_val = min(quota_used / 10000.0, 1.0)
st.sidebar.progress(progress_val, text=f"{quota_used} / 10,000 credits")

if app_page == "📊 Dashboard":
    youtube_link = st.sidebar.text_input("Enter YouTube Video Link", placeholder="https://www.youtube.com/watch?v=...")
    api_key = st.sidebar.text_input("Enter YouTube API Key", type="password", help="Requires a free YouTube Data API v3 key from Google Cloud Console.")
    max_comments = st.sidebar.slider("Max Live Comments to Fetch", 50, 500, 200, 50, help="Fewer comments fetch faster and use less API quota.")
    
    if not api_key or not youtube_link:
        st.info("👈 Please enter your YouTube API Key and a YouTube Video Link in the sidebar to load live analysis.")
        st.stop()
        
    video_id_input = extract_video_id(youtube_link)
    if not video_id_input:
        st.warning("Could not extract a valid YouTube Video ID from the link. Please check the URL format.")
        st.stop()
        
    with st.spinner("Fetching live video details from YouTube API..."):
        video_data = fetch_live_video_data(api_key, video_id_input)
        
    if video_data.empty:
        st.warning("Video not found or API error. Please verify your API Key and ensure the video is public.")
        st.stop()
        
    video_info = video_data.iloc[0]
    
    with st.spinner("Fetching live comment threads from YouTube API..."):
        vid_comments = fetch_live_comments(api_key, video_id_input, max_results=max_comments)
        
    # Increment quota tracking for new successful queries
    if 'last_fetched_video' not in st.session_state or st.session_state.last_fetched_video != video_id_input:
        st.session_state.api_quota_used += 1 + max(1, (max_comments // 100))
        st.session_state.last_fetched_video = video_id_input

    # Save successful query to session history
    search_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_key = f"[Live API] {video_info.get('title', 'N/A')} ({video_info.get('videoId', 'N/A')})"
    st.session_state.query_history[history_key] = {
        'video_info': video_info,
        'vid_comments': vid_comments,
        'search_time': search_time,
        'title': video_info.get('title', 'N/A'),
        'video_id': video_info.get('videoId', 'N/A'),
        'url': youtube_link,
        'published_date': video_info.get('publishedAt', 'N/A')
    }

    st.subheader("📊 Current Video Analytics")
    render_dashboard_layout(video_info, vid_comments, key_suffix="live")

elif app_page == "🕐 History":
    if not st.session_state.query_history:
        st.subheader("🕐 History")
        st.info("No query history found yet. Perform a query in the Dashboard first!")
    elif st.session_state.history_page == 'list':
        head_col, btn_view_col, btn_clear_col = st.columns([2.5, 1, 1])
        with head_col:
            st.subheader("🕐 History")
        
        # Build display table
        history_list = []
        for key, item in st.session_state.query_history.items():
            history_list.append({
                'Last Search Time': item.get('search_time', 'N/A'),
                'Video Title': item.get('title', 'N/A'),
                'Video ID / Link': item.get('url', 'N/A'),
                'Published Date': item.get('published_date', 'N/A'),
                'key': key
            })
            
        history_df = pd.DataFrame(history_list)
        
        # Selection logic via table
        st.markdown("**Select a video from the table below, then click 'View Analytics' to load the full dashboard.**")
        
        selection_event = st.dataframe(
            history_df[['Last Search Time', 'Video Title', 'Video ID / Link', 'Published Date']],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="history_table_display"
        )
        
        selected_key = None
        selected_rows = []
        if hasattr(selection_event, "selection") and hasattr(selection_event.selection, "rows"):
            selected_rows = selection_event.selection.rows
        elif isinstance(selection_event, dict):
            selected_rows = selection_event.get("selection", {}).get("rows", [])
            
        if selected_rows:
            selected_index = selected_rows[0]
            selected_key = history_df.iloc[selected_index]['key']

        with btn_view_col:
            if st.button("🔍 View Analytics", use_container_width=True, type="primary"):
                if selected_key:
                    st.session_state.selected_history_key = selected_key
                    st.session_state.history_page = 'detail'
                    st.rerun()
                else:
                    st.toast("Please select a video from the table first!", icon="⚠️")
        with btn_clear_col:
            if st.button("🗑️ Clear History", use_container_width=True):
                confirm_clear_dialog()

    elif st.session_state.history_page == 'detail':
        if st.button("⬅️ Back to History List"):
            st.session_state.history_page = 'list'
            st.session_state.selected_history_key = None
            st.rerun()

        selected_key = st.session_state.selected_history_key
        if selected_key and selected_key in st.session_state.query_history:
            history_item = st.session_state.query_history[selected_key]
            h_video_info = history_item['video_info']
            h_vid_comments = history_item['vid_comments']

            st.info(f"Showing cached analytics for: **{h_video_info.get('title', 'N/A')}**  |  Searched at: {history_item.get('search_time', 'N/A')}")
            render_dashboard_layout(h_video_info, h_vid_comments, key_suffix="history")
        else:
            st.error("Selected history item not found. Please go back and try again.")
            st.session_state.history_page = 'list'

