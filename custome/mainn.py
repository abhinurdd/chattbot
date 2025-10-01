import streamlit as st
import json
import requests
import os

# --- SET YOUR OPENROUTER API KEY HERE ---
OPENROUTER_API_KEY = "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"

# --- OpenRouter helper ---
def message_model(prompt, model="deepseek/deepseek-chat-v3.1:free"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3
    }
    resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=70)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def load_profile(path="profile_metrics.json"):
    with open(path) as f:
        return json.load(f)

profile = load_profile()

# --- Helper: metric formulas (simplified) ---
def get_er(content, followers):
    return round(((content["avg_likes"] + content.get("avg_comments",0) + content.get("avg_shares",0) + content.get("avg_saves",0)) / followers) * 100, 2)
def get_gr(new, last_week): return round((new / last_week) * 100, 2)

def build_prompt(user_q, profile):
    reel_er = get_er(profile["content_performance"]["Reels"], profile["followers"])
    post_er = get_er(profile["content_performance"]["Posts"], profile["followers"])
    gr = get_gr(profile["new_followers_this_week"], profile["followers_last_week"])
    top_age = max(profile["audience_breakdown"]["age"], key=lambda x: profile["audience_breakdown"]["age"][x])
    competitor = profile["competitor"]
    prompt = f"""
You are an expert social media analyst for creators. Use the below metrics and answer the user's question with numbers, reasoning, context and next action.

User's question: "{user_q}"

Profile summary:
- Username: {profile['username']} ({profile['full_name']})
- Followers: {profile['followers']}
- Engagement Rate (Reels): {reel_er}%, Posts: {post_er}%
- Weekly Growth Rate: {gr}%
- Top Age Group: {top_age}
- Trending Hashtags: {', '.join(profile['trending_hashtags'])}
- Competitor: {competitor['username']} ({competitor['followers']} followers, avg ER: {competitor['avg_er']*100:.1f}%, posts: {competitor['posting_frequency']}/week), top type: {competitor['top_content_type']}

Checklist:
- Use math from these fields for ER, GR, Virality, etc.
- Compare to competitor if useful.
- Always recommend a next step or new idea.
- Answer should be concise, data-backed, actionable, and in expert tone.
"""
    return prompt

st.title("Creator Performance Chatbot")
if "history" not in st.session_state: st.session_state.history = []

user_q = st.chat_input("Ask about your growth, best content, top collab, audience, etc.", key="user_message")

if user_q:
    st.session_state.history.append({"role":"user", "content":user_q})
    prompt = build_prompt(user_q, profile)
    try:
        response = message_model(prompt, "openai/gpt-3.5-turbo") # You can change the model to any OpenRouter supported, e.g. "openrouter/gpt-4"
    except Exception as e:
        response = f"Sorry, there was an error with the LLM: {e}"
    st.session_state.history.append({"role":"assistant", "content":response})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
