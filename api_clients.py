import requests
import json
import re
import time
import asyncio
import aiohttp
import ssl
import nest_asyncio
from typing import List, Dict, Optional, Tuple, Union
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    SERPER_API_KEY,
    APIFY_API_TOKEN
)

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Initialize API client
try:
    openrouter_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_BASE,
    )
except Exception as e:
    print(f"Failed to initialize OpenRouter client: {e}")
    openrouter_client = None


# FIXED: Global async session manager without problematic timeout
class AsyncSessionManager:
    def __init__(self):
        self.session = None
        self.connector = None
    
    async def get_session(self):
        if self.session is None or self.session.closed:
            # SSL configuration for development (bypass certificate issues)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.connector = aiohttp.TCPConnector(
                limit=10,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                ssl=ssl_context
            )
            
            # FIXED: Remove global timeout that causes "should be used inside a task" error
            self.session = aiohttp.ClientSession(
                connector=self.connector
                # No global timeout - will use per-request timeouts instead
            )
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector:
            await self.connector.close()

# Global session manager instance
session_manager = AsyncSessionManager()


def _get_message_content(response):
    """Extract message content from API response."""
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content
        return None
    except Exception:
        return None


def message_model(prompt: str, model: str, max_retries: int = 3):
    """Send prompt to AI model with retries (synchronous version for compatibility)."""
    if not openrouter_client or not OPENROUTER_API_KEY:
        print("❌ OpenRouter client not available")
        return None
    
    for attempt in range(max_retries):
        try:
            response = openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            content = _get_message_content(response)
            if content and content.strip():
                return content.strip()
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


async def message_model_async(session: aiohttp.ClientSession, prompt: str, model: str, max_retries: int = 3):
    """FIXED: Async version of AI model call with per-request timeout."""
    if not OPENROUTER_API_KEY:
        print("❌ OpenRouter API key not available")
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nurdd.ai",
        "X-Title": "Nurdd AI Marketing"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    
    for attempt in range(max_retries):
        try:
            # FIXED: Per-request timeout instead of global timeout
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with session.post(OPENROUTER_API_BASE + "/chat/completions", 
                                  headers=headers, 
                                  json=payload,
                                  timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("choices") and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content and content.strip():
                            return content.strip()
                else:
                    print(f"AI API error (attempt {attempt + 1}): HTTP {response.status}")
        except Exception as e:
            print(f"AI API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return None


def _looks_like_name(query: str) -> bool:
    """Simple heuristic to check if a query looks like a person's name"""
    query = query.strip().lower()
    
    # Clearly not names - these are product/topic indicators
    non_name_indicators = [
        'review', 'best', 'top', 'how to', 'tutorial', 'guide', 'tips',
        'laptop', 'phone', 'camera', 'product', 'brand', 'company',
        'vs', 'comparison', 'price', 'buy', 'sell', 'discount', 'deal',
        'unboxing', 'specs', 'features', 'gaming', 'tech', 'mobile',
        'what is', 'which is', 'where to', 'when to', 'why',
        'promote my', 'advertise my', 'marketing for', 'influencers for'
    ]
    
    # Check if query contains clear non-name indicators
    if any(indicator in query for indicator in non_name_indicators):
        return False
    
    # Simple name patterns
    words = query.split()
    
    # Empty or too long (likely not a name)
    if len(words) == 0 or len(words) > 4:
        return False
    
    # Single word that could be a name (like "carryminati", "pewdiepie")
    if len(words) == 1:
        word = words[0]
        if (word.isalpha() or word.replace('_', '').replace('.', '').isalnum()) and 3 <= len(word) <= 20:
            return True
        return False
    
    # Two words that could be first + last name
    if len(words) == 2:
        return all(
            (word.isalpha() or word.replace("'", "").isalpha()) and 2 <= len(word) <= 15 
            for word in words
        )
    
    # Three words (could be full name like "mr beast official")
    if len(words) == 3:
        return all(
            (word.isalpha() or word.replace("'", "").isalpha()) and 2 <= len(word) <= 15 
            for word in words
        )
    
    # Four words - be more strict
    if len(words) == 4:
        return all(
            word.isalpha() and 2 <= len(word) <= 12
            for word in words
        )
    
    return False


def spell_correct_influencer_name(query: str, available_influencers: list = None) -> dict:
    """Use AI to correct spelling mistakes and find the right influencer (synchronous version)"""
    if not available_influencers:
        available_influencers = []
    
    # Create a list of known influencers for context
    influencer_context = ""
    if available_influencers:
        influencer_names = [inf.get('name', '') for inf in available_influencers[:20] if inf.get('name')]
        if influencer_names:
            influencer_context = f"\n\nKnown influencers in database: {', '.join(influencer_names)}"
    
    spelling_correction_prompt = f"""
You are an expert at identifying potential influencer names and correcting spelling mistakes.

User Query: "{query}"
{influencer_context}

Your task:
1. Determine if this could be a person's name (be VERY generous - assume YES unless clearly a product/topic)
2. Correct any obvious spelling mistakes
3. Provide the most likely correct spelling

IMPORTANT GUIDELINES:
- If it looks like ANY kind of name (even unknown ones), treat it as a potential influencer
- People have diverse names from different cultures - don't reject unfamiliar names
- Only say "No" if it's clearly a product, topic, or promotional query
- When in doubt, say "Yes" - it's better to attempt auto-scraping than to miss someone

Examples of what to consider as potential influencer names:
✅ "arpit bala" → YES (could be a name, even if unknown)
✅ "john smith" → YES (common name, could be influencer)  
✅ "tech reviewer guy" → YES (could be referring to a person)
✅ "some random youtuber" → YES (referring to a person)
✅ "dhruv rathi" → YES (misspelling of Dhruv Rathee)
✅ "carry minati" → YES (misspelling of CarryMinati)
✅ "new influencer name" → YES (referring to a person)
❌ "laptop review" → NO (clearly a topic)
❌ "best smartphones 2024" → NO (clearly a product category)
❌ "promote my gaming brand" → NO (promotional request)
❌ "influencers for skincare" → NO (looking for category)

Respond in this EXACT format:

Is Influencer: [Yes/No - be generous with Yes]
Corrected Name: [corrected spelling or original if no correction needed]
Original Query: {query}
Confidence: [0.0 to 1.0]
Reasoning: [brief explanation of decision]
"""
    
    try:
        response = message_model(spelling_correction_prompt, "openai/gpt-oss-20b:free")
        if response:
            result = parse_spelling_correction_response(response, query)
            
            # SAFETY CHECK: If AI says no but it looks like a name, override it
            if not result['is_influencer'] and _looks_like_name(query):
                print(f"⚠️ AI said '{query}' is not influencer, but overriding because it looks like a name")
                result['is_influencer'] = True
                result['confidence'] = 0.75
                result['reasoning'] = "Overridden by heuristic: Appears to be a person's name"
                
            print(f"🤖 AI Analysis: '{query}' → is_influencer={result['is_influencer']} (confidence: {result['confidence']:.2f})")
            return result
            
    except Exception as e:
        print(f"Spelling correction error: {e}")
    
    # Fallback: if it looks like a name, assume it's an influencer
    is_likely_name = _looks_like_name(query)
    print(f"🔧 Fallback heuristic: '{query}' → is_name={is_likely_name}")
    
    return {
        'is_influencer': is_likely_name,
        'corrected_name': query,
        'original_query': query,
        'confidence': 0.7 if is_likely_name else 0.2,
        'reasoning': 'AI spelling correction failed, using heuristic name detection'
    }


async def spell_correct_influencer_name_async(query: str, available_influencers: list = None) -> dict:
    """FIXED: Async version of spelling correction with proper timeout handling"""
    if not available_influencers:
        available_influencers = []
    
    # Create a list of known influencers for context
    influencer_context = ""
    if available_influencers:
        influencer_names = [inf.get('name', '') for inf in available_influencers[:20] if inf.get('name')]
        if influencer_names:
            influencer_context = f"\n\nKnown influencers in database: {', '.join(influencer_names)}"
    
    spelling_correction_prompt = f"""
You are an expert at identifying potential influencer names and correcting spelling mistakes.

User Query: "{query}"
{influencer_context}

Your task:
1. Determine if this could be a person's name (be VERY generous - assume YES unless clearly a product/topic)
2. Correct any obvious spelling mistakes
3. Provide the most likely correct spelling

IMPORTANT GUIDELINES:
- If it looks like ANY kind of name (even unknown ones), treat it as a potential influencer
- People have diverse names from different cultures - don't reject unfamiliar names
- Only say "No" if it's clearly a product, topic, or promotional query
- When in doubt, say "Yes" - it's better to attempt auto-scraping than to miss someone

Respond in this EXACT format:

Is Influencer: [Yes/No - be generous with Yes]
Corrected Name: [corrected spelling or original if no correction needed]
Original Query: {query}
Confidence: [0.0 to 1.0]
Reasoning: [brief explanation of decision]
"""
    
    try:
        session = await session_manager.get_session()
        response = await message_model_async(session, spelling_correction_prompt, "openai/gpt-oss-20b:free")
        if response:
            result = parse_spelling_correction_response(response, query)
            
            # SAFETY CHECK: If AI says no but it looks like a name, override it
            if not result['is_influencer'] and _looks_like_name(query):
                print(f"⚠️ AI said '{query}' is not influencer, but overriding because it looks like a name")
                result['is_influencer'] = True
                result['confidence'] = 0.75
                result['reasoning'] = "Overridden by heuristic: Appears to be a person's name"
                
            print(f"🤖 AI Analysis: '{query}' → is_influencer={result['is_influencer']} (confidence: {result['confidence']:.2f})")
            return result
            
    except Exception as e:
        print(f"Async spelling correction error: {e}")
    
    # Fallback: if it looks like a name, assume it's an influencer
    is_likely_name = _looks_like_name(query)
    print(f"🔧 Fallback heuristic: '{query}' → is_name={is_likely_name}")
    
    return {
        'is_influencer': is_likely_name,
        'corrected_name': query,
        'original_query': query,
        'confidence': 0.7 if is_likely_name else 0.2,
        'reasoning': 'AI spelling correction failed, using heuristic name detection'
    }


def parse_spelling_correction_response(response_text: str, original_query: str) -> dict:
    """Parse AI spelling correction response"""
    result = {
        'is_influencer': True,  # Default to True to be more permissive
        'corrected_name': original_query,
        'original_query': original_query,
        'confidence': 0.5,
        'reasoning': 'Parsing failed, defaulted to influencer'
    }
    
    if not response_text:
        return result
    
    try:
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith('Is Influencer:'):
                is_inf = line.replace('Is Influencer:', '').strip().lower()
                result['is_influencer'] = is_inf in ['yes', 'true', '1', 'y']
            elif line.startswith('Corrected Name:'):
                corrected = line.replace('Corrected Name:', '').strip()
                if corrected and corrected.lower() != 'null':
                    result['corrected_name'] = corrected
            elif line.startswith('Confidence:'):
                conf_str = line.replace('Confidence:', '').strip()
                try:
                    number_match = re.search(r'(\d*\.?\d+)', conf_str)
                    if number_match:
                        conf_value = float(number_match.group(1))
                        if conf_value > 1:
                            conf_value = conf_value / 100
                        result['confidence'] = min(max(conf_value, 0.0), 1.0)
                except:
                    result['confidence'] = 0.5
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line.replace('Reasoning:', '').strip()
    
    except Exception as e:
        print(f"Spelling correction parsing error: {e}")
    
    return result


def enhanced_influencer_normalization(query: str, available_influencers: list = None) -> dict:
    """Improved normalization with spelling correction and database context (synchronous)"""
    if not available_influencers:
        available_influencers = []
    
    # Get names from available influencers for context
    known_names = []
    for inf in available_influencers[:30]:  # Top 30 for context
        name = inf.get('name', '') or inf.get('basic_info', {}).get('name', '')
        username = inf.get('username', '') or inf.get('basic_info', {}).get('username', '')
        if name:
            known_names.append(f"{name} (@{username})")
    
    context = f"\n\nKnown influencers in database:\n" + "\n".join(known_names[:15]) if known_names else ""
    
    enhanced_prompt = f"""
You are an expert Instagram influencer identifier with access to a database of influencers.

Query: "{query}"
{context}

Your task:
1. Correct spelling mistakes if any
2. Identify the most likely official Instagram account
3. Provide variations and handles that might be used

Handle common misspellings and variations:
- "dhruv rathi" → "Dhruv Rathee"
- "carry minati" → "CarryMinati" 
- "harsh beniwall" → "Harsh Beniwal"
- "bhuvan bam" → "Bhuvan Bham"
- "arpit bala" → "Arpit Bala" (keep as is if no known correction)

Generate reasonable username variations:
- Convert spaces to underscores, dots, or remove them
- Add common suffixes like "official", "real", numbers
- Consider cultural naming patterns

Respond in this EXACT format:

Search Name: [corrected official name]

Aliases:
- [alternative spelling variation]
- [another possible variation]

Handles:
- [likely Instagram username]
- [alternative username possibility]

If the query matches someone in the known database exactly, prioritize that information.
For unknown names like "arpit bala", provide reasonable variations.
"""
    
    try:
        response = message_model(enhanced_prompt, "openai/gpt-oss-20b:free")
        if response:
            normalized = parse_normalization_response(response, query)
            # Add confidence based on database match
            if known_names:
                query_lower = query.lower()
                for known in known_names:
                    if query_lower in known.lower():
                        normalized['database_match'] = True
                        break
            return normalized
    except Exception as e:
        print(f"Enhanced normalization error: {e}")
    
    # Enhanced fallback with better variations
    return {
        'search_name': query.title(),  # Capitalize properly
        'aliases': [
            query.lower(),
            query.title(),
            query.replace(' ', ''),
            query.replace(' ', '_')
        ],
        'handles': [
            query.replace(' ', '').lower(),
            query.replace(' ', '_').lower(),
            query.replace(' ', '.').lower(),
            query.replace(' ', '').lower() + 'official',
            query.replace(' ', '').lower() + '22'
        ],
        'database_match': False
    }


def parse_normalization_response(response_text: str, original_query: str) -> dict:
    """Parse AI normalization response with fallbacks."""
    result = {
        'search_name': original_query.title(),
        'aliases': [original_query.lower(), original_query.title()],
        'handles': [
            original_query.replace(' ', '').lower(),
            original_query.replace(' ', '_').lower()
        ],
        'database_match': False
    }
    
    if not response_text:
        return result
    
    try:
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            if 'search name:' in line_lower:
                name = line.split(':', 1)[1].strip()
                if name and len(name) < 100:  # Reasonable name length
                    result['search_name'] = name
                    
            elif line_lower == 'aliases:':
                current_section = 'aliases'
            elif line_lower == 'handles:':
                current_section = 'handles'
            elif line.startswith('- ') and current_section:
                item = line[2:].strip()
                # Only add reasonable items (not long sentences)
                if len(item) < 50 and len(item.split()) <= 4 and item.lower() not in ['none known', 'none', 'n/a']:
                    if current_section in result:
                        result[current_section].append(item)
    
    except Exception as e:
        print(f"Normalization parsing error: {e}")
    
    # Ensure we have some basic variations even if parsing failed
    if not result['aliases'] or len(result['aliases']) < 2:
        result['aliases'] = [original_query.lower(), original_query.title(), original_query.replace(' ', '')]
    
    if not result['handles'] or len(result['handles']) < 2:
        base = original_query.replace(' ', '').lower()
        result['handles'] = [
            base,
            base + 'official',
            base.replace(' ', '_'),
            base.replace(' ', '.'),
            base + '22'
        ]
    
    return result


def parse_profile_selection_response(response_text: str) -> dict:
    """Parse AI profile selection response."""
    result = {}
    
    if not response_text:
        return result
    
    try:
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith('Name:'):
                result['name'] = line.replace('Name:', '').strip()
            elif line.startswith('Username:'):
                result['username'] = line.replace('Username:', '').strip()
            elif line.startswith('Instagram URL:'):
                result['instagram_url'] = line.replace('Instagram URL:', '').strip()
            elif line.startswith('Confidence:'):
                conf_str = line.replace('Confidence:', '').strip()
                try:
                    result['confidence'] = float(re.search(r'(\d*\.?\d+)', conf_str).group(1))
                except:
                    result['confidence'] = 0.5
    except Exception as e:
        print(f"Profile parsing error: {e}")
    
    return result


def parse_scoring_response(response_text: str, username: str, metrics: dict) -> dict:
    """Parse AI scoring response with robust fallbacks."""
    # Default scores that will be returned if parsing fails
    scores = {
        'username': username,
        'postsAnalyzed': metrics.get('postsAnalyzed', 0),
        'avgER_organic': metrics.get('avgEngagement_organic', 0),
        'avgER_sponsored': metrics.get('avgEngagement_sponsored', 0),
        'avgLikes': metrics.get('avgLikes', 0),
        'avg_sponsored_likes': metrics.get('avg_sponsored_likes', 0),
        'avgComments': metrics.get('avgComments', 0),
        'avg_sponsored_comments': metrics.get('avg_sponsored_comments', 0),
        'avgViews': metrics.get('avgViews', 0),
        'Authenticity': 75.0,
        'BrandSafety': 85.0,
        'AudienceMatch': 70.0,
        'ContentQuality': 75.0,
    }
    
    if not response_text or len(response_text.strip()) < 10:
        print("Empty or short AI response, using defaults")
        return scores
    
    try:
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            score_match = re.search(r'(\d+(?:\.\d+)?)', line)
            if not score_match:
                continue
                
            score_value = float(score_match.group(1))
            
            if 'authenticity' in line.lower():
                scores['Authenticity'] = min(100.0, max(0.0, score_value))
            elif 'brand safety' in line.lower():
                scores['BrandSafety'] = min(100.0, max(0.0, score_value))
            elif 'audience match' in line.lower():
                scores['AudienceMatch'] = min(100.0, max(0.0, score_value))
            elif 'content quality' in line.lower():
                scores['ContentQuality'] = min(100.0, max(0.0, score_value))
    
    except Exception as e:
        print(f"Scoring parsing error: {e}")
    
    return scores


def search_google(query: str):
    """Search Google using Serper API (synchronous version)."""
    if not SERPER_API_KEY:
        print("❌ Serper API key missing")
        return None
    
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{query} site:instagram.com",
        "gl": "in",
        "hl": "en",
        "num": 10
    }
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Google search error: {e}")
        return None


async def search_google_async(query: str):
    """FIXED: Async version of Google search with per-request timeout."""
    if not SERPER_API_KEY:
        print("❌ Serper API key missing")
        return None
    
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{query} site:instagram.com",
        "gl": "in",
        "hl": "en",
        "num": 10
    }
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        session = await session_manager.get_session()
        
        # FIXED: Per-request timeout to avoid context manager error
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Google search error: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Async Google search error: {e}")
        # Fallback to synchronous version
        try:
            return search_google(query)
        except:
            return None


def scrape_instagram_posts_primary(username: str):
    """Scrape Instagram posts using Apify (synchronous version)."""
    if not APIFY_API_TOKEN:
        print("❌ Apify token missing")
        return []
    
    url = "https://api.apify.com/v2/acts/apify~instagram-post-scraper/run-sync-get-dataset-items"
    headers = {
        "Authorization": f"Bearer {APIFY_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "resultsLimit": 50,
        "skipPinnedPosts": False,
        "username": [username]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        if response.ok:
            data = response.json()
            items = data if isinstance(data, list) else data.get('items', [])
            print(f"✅ Posts scraper: {len(items)} posts")
            return items
        else:
            print(f"❌ Posts scraper failed: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Posts scraper error: {e}")
        return []


async def scrape_instagram_posts_primary_async(username: str):
    """FIXED: Async version with proper status codes and timeout handling."""
    if not APIFY_API_TOKEN:
        print("❌ Apify token missing")
        return []
    
    url = "https://api.apify.com/v2/acts/apify~instagram-post-scraper/run-sync-get-dataset-items"
    headers = {
        "Authorization": f"Bearer {APIFY_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "resultsLimit": 50,
        "skipPinnedPosts": False,
        "username": [username]
    }
    
    try:
        session = await session_manager.get_session()
        
        # FIXED: Per-request timeout
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
            # FIXED: Accept both 200 and 201 as success
            if response.status in [200, 201]:
                data = await response.json()
                items = data if isinstance(data, list) else data.get('items', [])
                print(f"✅ Async posts scraper: {len(items)} posts (HTTP {response.status})")
                return items
            else:
                print(f"❌ Async posts scraper failed: HTTP {response.status}")
                return []
    except Exception as e:
        print(f"❌ Async posts scraper error: {e}")
        # Fallback to synchronous version
        try:
            return scrape_instagram_posts_primary(username)
        except:
            return []


def scrape_profile_info_only(username: str):
    """Get Instagram profile info using Apify (synchronous version)."""
    if not APIFY_API_TOKEN:
        print("❌ Apify token missing")
        return None
    
    url = "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items"
    headers = {
        "Authorization": f"Bearer {APIFY_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "usernames": [username],
        "resultsLimit": 1,
        "includePostsCount": 0
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        if response.ok:
            data = response.json()
            items = data if isinstance(data, list) else data.get('items', [])
            
            if items:
                profile = items[0]
                profile_data = {
                    "username": profile.get("username", username),
                    "name": profile.get("fullName", "") or profile.get("name", ""),
                    "full_name": profile.get("fullName", "") or profile.get("name", ""),
                    "bio": profile.get("biography", "") or profile.get("bio", ""),
                    "website": profile.get("externalUrl", "") or profile.get("website", ""),
                    "profile_pic_url": profile.get("profilePicUrl", ""),
                    "is_verified": profile.get("verified", False),
                    "is_business_account": profile.get("businessAccount", False),
                    "category": profile.get("businessCategoryName", ""),
                    "followers_count": profile.get("followersCount", 0),
                    "following_count": profile.get("followsCount", 0),
                    "posts_count": profile.get("postsCount", 0),
                    "instagram_url": f"https://instagram.com/{username}/"
                }
                print(f"✅ Profile info: {profile_data['name']} - {profile_data['followers_count']:,} followers")
                return profile_data
        else:
            print(f"❌ Profile scraper failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Profile scraper error: {e}")
        return None


async def scrape_profile_info_only_async(username: str):
    """FIXED: Async version with proper status codes and timeout handling."""
    if not APIFY_API_TOKEN:
        print("❌ Apify token missing")
        return None
    
    url = "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items"
    headers = {
        "Authorization": f"Bearer {APIFY_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "usernames": [username],
        "resultsLimit": 1,
        "includePostsCount": 0
    }
    
    try:
        session = await session_manager.get_session()
        
        # FIXED: Per-request timeout
        timeout = aiohttp.ClientTimeout(total=120)
        
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
            # FIXED: Accept both 200 and 201 as success
            if response.status in [200, 201]:
                data = await response.json()
                items = data if isinstance(data, list) else data.get('items', [])
                
                if items:
                    profile = items[0]
                    profile_data = {
                        "username": profile.get("username", username),
                        "name": profile.get("fullName", "") or profile.get("name", ""),
                        "full_name": profile.get("fullName", "") or profile.get("name", ""),
                        "bio": profile.get("biography", "") or profile.get("bio", ""),
                        "website": profile.get("externalUrl", "") or profile.get("website", ""),
                        "profile_pic_url": profile.get("profilePicUrl", ""),
                        "is_verified": profile.get("verified", False),
                        "is_business_account": profile.get("businessAccount", False),
                        "category": profile.get("businessCategoryName", ""),
                        "followers_count": profile.get("followersCount", 0),
                        "following_count": profile.get("followsCount", 0),
                        "posts_count": profile.get("postsCount", 0),
                        "instagram_url": f"https://instagram.com/{username}/"
                    }
                    print(f"✅ Async profile info: {profile_data['name']} - {profile_data['followers_count']:,} followers (HTTP {response.status})")
                    return profile_data
            else:
                print(f"❌ Async profile scraper failed: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"❌ Async profile scraper error: {e}")
        # Fallback to synchronous version
        try:
            return scrape_profile_info_only(username)
        except:
            return None


def scrape_profile_and_posts_alternative(username: str):
    """Alternative method to get both profile and posts (synchronous version)."""
    if not APIFY_API_TOKEN:
        return None, []
    
    url = "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items"
    headers = {
        "Authorization": f"Bearer {APIFY_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "usernames": [username],
        "resultsLimit": 1,
        "includePostsCount": 30
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        if response.ok:
            data = response.json()
            items = data if isinstance(data, list) else data.get('items', [])
            
            if items:
                profile_info = items[0]
                profile_data = {
                    "username": profile_info.get("username", username),
                    "name": profile_info.get("fullName", "") or profile_info.get("name", ""),
                    "full_name": profile_info.get("fullName", "") or profile_info.get("name", ""),
                    "bio": profile_info.get("biography", ""),
                    "website": profile_info.get("externalUrl", ""),
                    "profile_pic_url": profile_info.get("profilePicUrl", ""),
                    "is_verified": profile_info.get("verified", False),
                    "is_business_account": profile_info.get("businessAccount", False),
                    "category": profile_info.get("businessCategoryName", ""),
                    "followers_count": profile_info.get("followersCount", 0),
                    "following_count": profile_info.get("followsCount", 0),
                    "posts_count": profile_info.get("postsCount", 0),
                    "instagram_url": f"https://instagram.com/{username}/"
                }
                
                posts_data = profile_info.get("posts", []) or profile_info.get("latestPosts", [])
                print(f"✅ Alternative scraper: Profile + {len(posts_data)} posts")
                return profile_data, posts_data
        
        print("❌ Alternative scraper failed")
        return None, []
    except Exception as e:
        print(f"❌ Alternative scraper error: {e}")
        return None, []


async def scrape_profile_and_posts_parallel_async(username: str) -> Tuple[Optional[Dict], List[Dict]]:
    """FIXED: Parallel scraping with proper status codes and timeout handling"""
    if not APIFY_API_TOKEN:
        print("❌ Apify API token missing")
        return None, []
    
    print(f"🚀 Starting parallel scraping for: @{username}")
    
    # Create tasks for parallel execution
    profile_task = asyncio.create_task(
        scrape_profile_info_only_async(username),
        name=f"profile_{username}"
    )
    posts_task = asyncio.create_task(
        scrape_instagram_posts_primary_async(username),
        name=f"posts_{username}"
    )
    
    # Wait for both to complete in parallel
    try:
        profile_data, posts_data = await asyncio.gather(
            profile_task, 
            posts_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(profile_data, Exception):
            print(f"Profile scraping error: {profile_data}")
            profile_data = None
        
        if isinstance(posts_data, Exception):
            print(f"Posts scraping error: {posts_data}")
            posts_data = []
        
        # If posts failed, try alternative method
        if not posts_data and profile_data:
            print("Primary posts scraper failed, trying alternative...")
            try:
                profile_alt, posts_alt = scrape_profile_and_posts_alternative(username)
                if posts_alt:
                    posts_data = posts_alt
                if not profile_data and profile_alt:
                    profile_data = profile_alt
            except Exception as e:
                print(f"Alternative scraper also failed: {e}")
        
        print(f"✅ Parallel scraping completed: Profile={'✅' if profile_data else '❌'}, Posts={len(posts_data) if posts_data else 0}")
        return profile_data, posts_data or []
        
    except Exception as e:
        print(f"❌ Parallel scraping error: {e}")
        # Fallback to synchronous method
        try:
            return scrape_complete_instagram_profile(username)
        except:
            return None, []


def scrape_complete_instagram_profile(username: str):
    """Main function to scrape complete Instagram profile (synchronous version for compatibility)."""
    if not APIFY_API_TOKEN:
        print("❌ Apify API token missing")
        return None, []
    
    print(f"Scraping complete profile for: @{username}")
    
    # Try primary posts scraper
    posts_data = scrape_instagram_posts_primary(username)
    
    # Get profile info
    profile_data = scrape_profile_info_only(username)
    
    # If posts failed, try alternative
    if not posts_data:
        print("Primary posts scraper failed, trying alternative...")
        profile_alt, posts_alt = scrape_profile_and_posts_alternative(username)
        if posts_alt:
            posts_data = posts_alt
        if not profile_data and profile_alt:
            profile_data = profile_alt
    
    return profile_data, posts_data


async def scrape_complete_instagram_profile_async(username: str) -> Tuple[Optional[Dict], List[Dict]]:
    """FIXED: Async version with all fixes applied"""
    try:
        return await scrape_profile_and_posts_parallel_async(username)
    except Exception as e:
        print(f"Async scraping failed: {e}, falling back to sync")
        return scrape_complete_instagram_profile(username)


def test_apify_connection():
    """Test Apify API connection (synchronous version)."""
    if not APIFY_API_TOKEN:
        print("❌ No Apify token found")
        return False
    
    try:
        response = requests.get(
            "https://api.apify.com/v2/acts",
            headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"},
            timeout=10
        )
        if response.ok:
            print("✅ Apify connection successful")
            return True
        else:
            print(f"❌ Apify connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Apify connection error: {e}")
        return False


# FIXED: Utility function for Streamlit integration with better error handling
def run_async_in_sync(async_func):
    """FIXED: Better event loop management for Streamlit with multiple fallback strategies."""
    try:
        # Strategy 1: Use existing event loop with nest_asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # nest_asyncio should handle this
                return loop.run_until_complete(async_func)
            else:
                return loop.run_until_complete(async_func)
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e) or "Event loop is closed" in str(e):
                # Strategy 2: Create new event loop in thread
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(async_func)
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=300)  # 5 minute timeout
            else:
                raise
    
    except Exception as e:
        print(f"All async execution strategies failed: {e}")
        # Strategy 3: Graceful degradation - return None or empty result
        print("Falling back to synchronous operation...")
        return None


# FIXED: Cleanup function with better error handling
async def cleanup_async_resources():
    """Clean up async resources with error handling."""
    try:
        await session_manager.close()
        print("✅ Async resources cleaned up")
    except Exception as e:
        print(f"⚠️ Error during async cleanup: {e}")
