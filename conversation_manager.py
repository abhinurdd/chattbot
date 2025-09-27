import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from api_clients import message_model
from config import MODEL_SCORING
from unified_data_manager import unified_data_manager


class ConversationManager:
    """Enhanced conversation manager with auto-scraping awareness"""
    
    def __init__(self):
        self.context_history = []
        self.user_preferences = {}
    
    def generate_intelligent_response(self, user_message: str, conversation_history: List[Dict]) -> str:
        """Generate intelligent response with awareness of auto-scraped data"""
        
        context_info = self._extract_context_from_message(user_message)
        
        relevant_data = self._get_relevant_context(user_message)
        
        prompt = self._create_conversation_prompt(
            user_message, 
            conversation_history, 
            relevant_data,
            context_info
        )
        
        try:
            response = message_model(prompt, MODEL_SCORING)
            return response if response else "I'd be happy to help, but I'm having trouble generating a response right now."
        except Exception as e:
            print(f"Conversation generation error: {e}")
            return "I encountered an issue while processing your request. Could you please try rephrasing your question?"
    
    def _extract_context_from_message(self, message: str) -> Dict:
        """Extract context about what kind of information the user wants"""
        
        message_lower = message.lower()
        context = {
            'wants_followers': 'followers' in message_lower or 'subscriber' in message_lower,
            'wants_engagement': 'engagement' in message_lower or 'likes' in message_lower,
            'wants_collaborations': 'brand' in message_lower or 'collaboration' in message_lower,
            'wants_bio': 'bio' in message_lower or 'about' in message_lower,
            'wants_content': 'content' in message_lower or 'posts' in message_lower,
            'auto_scraped_indicator': False
        }
        
        return context
    
    def _get_relevant_context(self, user_message: str) -> Dict:
        """Get relevant context from databases with auto-scraping info"""
        
        words = user_message.split()
        potential_names = []
        
        # Look for capitalized words that might be names
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if next word is also capitalized (likely full name)
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    potential_names.append(f"{word} {words[i + 1]}")
                else:
                    potential_names.append(word)
        
        # Search for any of these names in our databases
        found_data = {}
        for name in potential_names:
            try:
                found, data, source = unified_data_manager.find_influencer_anywhere(name, auto_scrape=False)  # Don't auto-scrape in context lookup
                if found:
                    found_data[name] = {
                        'data': data,
                        'source': source,
                        'auto_scraped': source.startswith('auto_scraped')
                    }
            except:
                continue
        
        return found_data
    
    def _create_conversation_prompt(self, user_message: str, history: List[Dict], relevant_data: Dict, context_info: Dict) -> str:
        """Create enhanced conversation prompt with auto-scraping awareness"""
        
        # Get recent conversation context
        recent_messages = history[-6:] if len(history) > 6 else history
        conversation_context = "\n".join([
            f"{msg['role'].title()}: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"{msg['role'].title()}: {msg['content']}"
            for msg in recent_messages[-3:]  # Last 3 messages
        ])
        
        # Prepare influencer data context
        influencer_context = ""
        auto_scraped_info = ""
        
        if relevant_data:
            influencer_context = "\n\nRELEVANT INFLUENCER DATA:\n"
            for name, info in relevant_data.items():
                data = info['data']
                source = info['source']
                
                basic_info = data.get('basic_info', {})
                analysis = data.get('analysis', {})
                
                influencer_context += f"""
• {basic_info.get('name', 'Unknown')} (@{basic_info.get('username', '')})
  Source: {source}
  Followers: {basic_info.get('followers_count', 0):,}
  Verified: {'Yes' if basic_info.get('is_verified') else 'No'}
  Category: {basic_info.get('category', 'Unknown')}
  Engagement Rate: {analysis.get('engagement_rate', 0):.3f}%
  Bio: {basic_info.get('bio', 'No bio available')[:100]}...
  """
                
                if info['auto_scraped']:
                    auto_scraped_info += f"\n🕷️ NOTE: {basic_info.get('name', name)} was just auto-scraped and added to the database with fresh data!"
        
        prompt = f"""
You are Nurdd's AI Marketing Strategist - an expert in influencer marketing with access to comprehensive influencer databases.

RECENT CONVERSATION:
{conversation_context}

CURRENT USER MESSAGE: "{user_message}"

{influencer_context}

{auto_scraped_info}

CONTEXT ANALYSIS:
- User wants follower info: {context_info.get('wants_followers', False)}
- User wants engagement info: {context_info.get('wants_engagement', False)}  
- User wants collaboration info: {context_info.get('wants_collaborations', False)}
- User wants bio/about info: {context_info.get('wants_bio', False)}
- User wants content info: {context_info.get('wants_content', False)}

INSTRUCTIONS:
1. Provide a comprehensive, helpful response using the available data
2. If data was auto-scraped, mention that it's fresh and up-to-date
3. Include specific metrics, follower counts, engagement rates where available
4. Be conversational and engaging
5. If discussing influencers, include their verification status and category
6. Suggest relevant marketing insights based on the data
7. Use emojis appropriately to make the response engaging

RESPONSE GUIDELINES:
- Start with the most relevant information first
- Use bullet points for metrics and key facts
- Include follower counts with proper formatting (e.g., 1,234,567)
- Mention verification status and business account status
- If data is fresh from auto-scraping, highlight this as an advantage
- Provide actionable marketing insights when appropriate

Generate a helpful, detailed response:
"""
        
        return prompt


# Global conversation manager instance
conversation_manager = ConversationManager()
