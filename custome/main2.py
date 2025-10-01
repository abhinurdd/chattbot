import streamlit as st
import json
import requests
import os

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

def load_profile(path="profile_metrics.json"):
    with open(path) as f:
        return json.load(f)

profile = load_profile()

def get_er(content, followers):
    val = round(((content["avg_likes"] + content.get("avg_comments",0) + content.get("avg_shares",0) + content.get("avg_saves",0)) / followers) * 100, 2)
    print(f"LOG: Content ER calc: likes={content['avg_likes']}, comments={content.get('avg_comments',0)}, shares={content.get('avg_shares',0)}, saves={content.get('avg_saves',0)}, followers={followers}, ER={val}")
    return val

def get_gr(new, last_week):
    val = round((new / last_week) * 100, 2)
    print(f"LOG: Growth Rate calc: new_followers={new}, last_week={last_week}, GR={val}")
    return val

def get_virality(content):
    val = round((content["avg_shares"] / content["avg_impressions"]) * 100, 2)
    print(f"LOG: Virality calc: shares={content['avg_shares']}, impressions={content['avg_impressions']}, virality={val}")
    return val

def log_audience_segments(aud):
    print(f"LOG: Audience breakdown - Age: {aud['age']}, Gender: {aud['gender']}, Locations: {aud['locations']}")

METHODS_AND_EXAMPLES = """
---
For each user question, use these methods, formulas, and answer styles:
---
Am I doing well? Be honest—am I doing well or not?
- Metrics: ER = ((Likes + Comments + Shares + Saves) / Total Followers) * 100
          GR = (New Followers this week / Followers at previous week) * 100
- Method: 1) Calculate latest ER from post data. 2) Calculate GR. 3) Compare to typical niche values (avg ER = 3–5%).
- Example: "Your ER is 4.8%, slightly above average. Growth is 3.1% weekly—keep it up!"

Give me an idea that will grab attention fast
- Metrics: ER, Virality Score = (Shares per Post / Total Impressions) * 100, Trending Topic
- Method: 1) Find post types with highest ER and Virality; 2) Detect trending hashtags/content. 3) Suggest a fast action idea.
- Example: "Your short Reels with trending audio have highest shares (Virality Score 7.5%). Try a 15-sec Q&A Reel using current trending audio to grab attention fast."

Who’s my top collab? How do I message them?
- Metrics: Engagement Rate, Audience Overlap %.
- Method: 1) Find creators with high overlap & ER. 2) Give a DM template.
- Example: "Hi [Creator Name], I noticed we both create fitness Reels and share a similar audience (45% overlap)..."

What’s really working for me?
- Metrics: Content Type ER, Post Sentiment, Virality Score.
- Method: 1) Show best format by ER/sentiment. 2) Compare to other formats.
- Example: "Your Q&A Reels average 7.2% ER, 80% positive sentiment—this format works best..."

Which age group comments the most?
- Metrics: Comments per Post, Audience Age Segments, Engagement Rate (ER)
- Method: Calculate ER for each segment using (Comments from segment / Followers in segment) * 100; mention top-engaging group.
- Example: "The 25–34 age group is generating the most comments... Creating content tailored to this group, like informative Reels or polls, could further increase interaction."

How can I grow faster here?
- Metrics: ER, GR, Content Type Performance, Audience Insights
- Method: Analyze overall ER + GR, best content type, top segment, suggest posting time and action.
- Example: "Your Reels get highest engagement (6.2% ER), while Stories perform lower. Focus more on short Reels at 9 PM targeting 25–34 age group..."

What are my competitors doing that I'm not?
- Metrics: Competitor ER, Posting Frequency, Engagement Type
- Method: Compare ER and post type. Suggest missing actions.
- Example: "Your competitor’s Reels average 7.1% ER and they post 5 times/week.... Try posting 3–4 short Reels with trending audio."

What’s the next big trend I should catch early?
- Metrics: Trending Hashtags, Virality Score, Content Type ER
- Method: Trend/hashtag mining, suggest high-ER trending content.
- Example: "Short comedic Reels using trending audio are gaining 8–9% ER... Try a 10–15 sec Reel using [current trending audio] to catch this trend early."

What’s my untapped audience? What opportunity am I missing right now? Which direction should I take for more reach? Where can I expand beyond my current niche?
- Metrics: Audience Demographics (Age, Gender, Location), ER by Segment
- Method: Find segment with high ER but low reach, recommend content for them.
- Example: "Currently 80% of your audience is 18–24, but your 25–34 age group shows higher comment rates..."

What’s my hidden strength here?
- Metrics: ER, Content Type Performance, Virality Score
- Method: Find content type with highest engagement and virality.
- Example: "Q&A Reels have consistently high engagement with 7.5% ER and strong share rates..."

Summary of my profile
- Metrics: ER, GR, Content Type Performance, Audience Insights
- Method: Overview ER, GR, content, segments, growth recs.
- Example: "The profile has an overall ER of 6.5% and weekly growth of 3%. Reels are the top-performing content type... Focus on posting more Reels and targeting high-engagement segments for growth."
---
"""

def build_prompt(user_q, profile):
    reel_er = get_er(profile["content_performance"]["Reels"], profile["followers"])
    post_er = get_er(profile["content_performance"]["Posts"], profile["followers"])
    gr = get_gr(profile["new_followers_this_week"], profile["followers_last_week"])
    virality_reels = get_virality(profile["content_performance"]["Reels"])
    virality_posts = get_virality(profile["content_performance"]["Posts"])
    top_age = max(profile["audience_breakdown"]["age"], key=lambda x: profile["audience_breakdown"]["age"][x])
    log_audience_segments(profile["audience_breakdown"])
    competitor = profile["competitor"]

    prompt = f"""
{METHODS_AND_EXAMPLES}

Profile summary:
- Username: {profile['username']} ({profile['full_name']})
- Followers: {profile['followers']}
- Engagement Rate (Reels): {reel_er}%, Posts: {post_er}%
- Weekly Growth Rate: {gr}%
- Reels Virality: {virality_reels}%, Posts Virality: {virality_posts}%
- Top Age Group: {top_age}
- Audience breakdown (Age/Gender/Location): {profile['audience_breakdown']}
- Trending Hashtags: {', '.join(profile['trending_hashtags'])}
- Competitor: {competitor['username']} ({competitor['followers']} followers, avg ER: {competitor['avg_er']*100:.1f}%, posts: {competitor['posting_frequency']}/week), top type: {competitor['top_content_type']}
- Top collab: {profile['top_collab']}

Instructions:
- Pick the right analytic (from the methods above) for the user's question.
- Always use/calculate ER, GR, virality and quote actual numbers using these formulas and metrics above.
- End with one actionable tip, DM, or next step idea.
User's question: "{user_q}"
"""
    print(f"LOG: Prompt for LLM:\n{prompt}")
    return prompt

st.title("Creator Performance Chatbot (with Methods & Samples)")
if "history" not in st.session_state: st.session_state.history = []

user_q = st.chat_input("Ask about your growth, best content, top collab, audience, etc.", key="user_message")

if user_q:
    st.session_state.history.append({"role":"user", "content":user_q})
    prompt = build_prompt(user_q, profile)
    try:
        response = message_model(prompt, "deepseek/deepseek-chat-v3.1:free")
        print("LOG: LLM response received.")
    except Exception as e:
        response = f"Sorry, there was an error with the LLM: {e}"
        print(f"LOG: Error from LLM: {e}")
    st.session_state.history.append({"role":"assistant", "content":response})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
