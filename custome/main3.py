import streamlit as st
import json
import requests

OPENROUTER_API_KEY = "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"

def message_model(prompt, model="deepseek/deepseek-chat-v3.1:free"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 480,
        "temperature": 0.3
    }
    print("LOG: Sending prompt to LLM...")
    resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=90)
    print(f"LOG: OpenRouter response code: {resp.status_code}")
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    print("LOG: LLM answer:", text)
    return text

def load_profiles(path="profile_metrics2.json"):
    with open(path) as f:
        profiles = json.load(f)
        if isinstance(profiles, dict):
            profiles = [profiles]
        return profiles

def find_profile(profiles, username):
    for prof in profiles:
        if prof["username"] == username:
            return prof
    return None

profiles = load_profiles()

def get_er(content, followers):
    val = round(
        (
            content.get("avg_likes", 0)
            + content.get("avg_comments", 0)
            + content.get("avg_shares", 0)
            + content.get("avg_saves", 0)
        )
        / followers
        * 100,
        2,
    )
    print(
        f"LOG: Content ER calc: likes={content.get('avg_likes',0)}, comments={content.get('avg_comments',0)}, shares={content.get('avg_shares',0)}, saves={content.get('avg_saves',0)}, followers={followers}, ER={val}"
    )
    return val

def get_gr(new, last_week):
    val = round((new / last_week) * 100, 2)
    print(
        f"LOG: Growth Rate calc: new_followers={new}, last_week={last_week}, GR={val}"
    )
    return val

def get_virality(content):
    val = round((content.get("avg_shares", 0) / content.get("avg_impressions", 1)) * 100, 2)
    print(
        f"LOG: Virality calc: shares={content.get('avg_shares',0)}, impressions={content.get('avg_impressions',1)}, virality={val}"
    )
    return val

def log_audience_segments(aud):
    print(
        f"LOG: Audience breakdown - Age: {aud['age']}, Gender: {aud['gender']}, Locations: {aud['locations']}"
    )

SYSTEM_PROMPT_TEMPLATE = """
You are a sharp, data-driven social media coach for influencers.
For each user, summarize and answer questions using only their metrics. Respond in concise, highly actionable points.

Always use:
- ER = ((Likes + Comments + Shares + Saves)/Followers)*100
- GR = (New Followers this week / Followers last week)*100
- Virality Score = (Shares per Post / Total Impressions)*100

Follow these answer styles; each answer should be 3-5 punchy bullets, max one line each. End always with a standout tip, DM, or next action using an arrow (→):

Q: Am I doing well?
A:
• ER: 4.8% (above avg)
• GR: 3.1% weekly (strong!)
→ You're on track—keep pushing more Reels!

Q: What's working for me?
A:
• Q&A Reels: 7.2% ER, 80% positive sentiment
• Outperforming Posts/Stories
→ Post more Q&A Reels for faster growth!

Q: Give me an idea to grab attention:
A:
• Best: Short Reels w/ trending audio (ER 7.5%)
• Virality up 7.5%
• Try: 15-sec Q&A Reel with top trending audio
→ Move fast—jump on this trend today!

Q: Who's my top collab?
A:
• 'healthy_sam' (48% overlap)
• Latest: HIIT quick tips Reel
• DM: "Hi Sam, let’s collab on 15-sec workout?"
→ Message now for a top collab bump!

Q: What's my untapped audience?
A:
• 25–34 comments most, but 18–24 dominates followers
→ Post career/personal tips to capture 25–34 better

Q: How can I grow faster?
A:
• Reels: 6.2% ER (best)
• Try 9PM posts for 25–34 group
→ Focus on Reels and top commenters for growth

Instructions:
• Use numbers from the user's metrics below
• Answer every question in this bullet format ONLY (no intro/outro)
• Numbers + crisp next step, that's it!

User's key metrics/data:
{profile_summary}

User's question: "{question}"
"""

def make_profile_summary(profile):
    reel_er = get_er(profile["content_performance"]["Reels"], profile["followers"])
    post_er = get_er(profile["content_performance"]["Posts"], profile["followers"])
    gr = get_gr(profile["new_followers_this_week"], profile["followers_last_week"])
    virality_reels = get_virality(profile["content_performance"]["Reels"])
    virality_posts = get_virality(profile["content_performance"]["Posts"])
    top_age = max(
        profile["audience_breakdown"]["age"],
        key=lambda x: profile["audience_breakdown"]["age"][x],
    )
    log_audience_segments(profile["audience_breakdown"])
    competitor = profile["competitor"]
    return (
        f"- Username: {profile['username']} ({profile['full_name']})\n"
        f"- Followers: {profile['followers']}\n"
        f"- Engagement Rate (Reels): {reel_er}%, Posts: {post_er}%\n"
        f"- Weekly Growth Rate: {gr}%\n"
        f"- Reels Virality: {virality_reels}%, Posts Virality: {virality_posts}%\n"
        f"- Top Age Group: {top_age}\n"
        f"- Audience breakdown: {profile['audience_breakdown']}\n"
        f"- Trending Hashtags: {', '.join(profile['trending_hashtags'])}\n"
        f"- Competitor: {competitor['username']} ({competitor['followers']} followers, avg ER: {competitor['avg_er']*100:.1f}%, posts: {competitor['posting_frequency']}/week), top type: {competitor['top_content_type']}\n"
        f"- Top collab: {profile['top_collab']}\n"
    )

def build_prompt(question, profile):
    summary = make_profile_summary(profile)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(profile_summary=summary, question=question)
    print(f"LOG: Prompt for LLM:\n{prompt}")
    return prompt

st.title("Creator Performance Chatbot (Account-Based, Bulleted)")

if "history" not in st.session_state:
    st.session_state.history = []

if "user" not in st.session_state:
    st.session_state.user = profiles[0]["username"]

usernames = [prof["username"] for prof in profiles]
selected_username = st.selectbox("Choose your influencer username (simulate Instagram login):", usernames, index=usernames.index(st.session_state.user))
if selected_username != st.session_state.user:
    st.session_state.user = selected_username
    st.session_state.history = []

user_profile = find_profile(profiles, st.session_state.user)

user_q = st.chat_input("Ask about your growth, best content, top collab, audience, etc.", key="user_message")
if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    prompt = build_prompt(user_q, user_profile)
    try:
        response = message_model(prompt, "deepseek/deepseek-chat-v3.1:free")
        print("LOG: LLM response received.")
    except Exception as e:
        response = f"Sorry, there was an error with the LLM: {e}"
        print(f"LOG: Error from LLM: {e}")
    st.session_state.history.append({"role": "assistant", "content": response})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
