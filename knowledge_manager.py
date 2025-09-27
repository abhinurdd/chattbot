import json
import re
from datetime import datetime
from typing import List, Dict, Any
from database import get_database


class KnowledgeManager:
    """Manages and processes influencer knowledge base for RAG"""
    
    def __init__(self):
        self.knowledge_base = None
        self.influencer_profiles = {}
        self.last_updated = None
        self._load_knowledge()
    
    def _load_knowledge(self):
        """Load and process database into searchable knowledge"""
        db = get_database()
        profiles = db.get("profiles", {})
        
        self.knowledge_base = []
        self.influencer_profiles = {}
        
        for username, profile in profiles.items():
            try:
                basic_info = profile.get("basic_info", {})
                analysis = profile.get("analysis", {})
                hashtags = profile.get("hashtags", {}).get("most_used", {})
                collaborations = profile.get("brand_collaborations", {})
                
                # FIX: Safe handling of None values
                category = basic_info.get("category", "Unknown")
                if not category or category is None:
                    category = "Unknown"
                
                name = basic_info.get("name", "Unknown")
                if not name or name is None:
                    name = "Unknown"
                
                bio = basic_info.get("bio", "")
                if bio is None:
                    bio = ""
                
                # Create structured knowledge entry
                knowledge_entry = {
                    "username": username,
                    "name": name,
                    "category": category,
                    "followers": basic_info.get("followers_count", 0) or 0,
                    "engagement_rate": analysis.get("engagement_rate", 0) or 0,
                    "bio": bio,
                    "top_hashtags": list(hashtags.keys())[:15] if hashtags else [],
                    "brand_collaborations": collaborations.get("brands_worked_with", [])[:10] if collaborations.get("brands_worked_with") else [],
                    "content_quality": analysis.get("scores", {}).get("ContentQuality", 0) or 0,
                    "authenticity": analysis.get("scores", {}).get("Authenticity", 0) or 0,
                    "brand_safety": analysis.get("scores", {}).get("BrandSafety", 0) or 0,
                    "audience_match": analysis.get("scores", {}).get("AudienceMatch", 0) or 0,
                    "verified": basic_info.get("is_verified", False) or False,
                    "business_account": basic_info.get("is_business_account", False) or False
                }
                
                # Create searchable text representation
                searchable_text = f"""
                Influencer: {knowledge_entry['name']} (@{username})
                Category: {knowledge_entry['category']}
                Followers: {knowledge_entry['followers']:,}
                Engagement Rate: {knowledge_entry['engagement_rate']:.3f}%
                Bio: {knowledge_entry['bio']}
                Content Focus: {', '.join(knowledge_entry['top_hashtags'])}
                Past Brand Collaborations: {', '.join(knowledge_entry['brand_collaborations'])}
                Quality Scores - Content: {knowledge_entry['content_quality']}/100, Authenticity: {knowledge_entry['authenticity']}/100
                Verified: {'Yes' if knowledge_entry['verified'] else 'No'}
                """
                
                knowledge_entry["searchable_text"] = searchable_text
                self.knowledge_base.append(knowledge_entry)
                self.influencer_profiles[username] = knowledge_entry
                
            except Exception as e:
                print(f"Error processing profile {username}: {e}")
                continue
        
        self.last_updated = datetime.now()
        print(f"✅ Knowledge base loaded with {len(self.knowledge_base)} influencer profiles")

    
    def get_relevant_influencers(self, query: str, limit: int = 5) -> List[Dict]:
        """Find most relevant influencers for a query"""
        if not self.knowledge_base:
            return []
        
        query_lower = query.lower()
        scored_influencers = []
        
        for influencer in self.knowledge_base:
            score = 0
            searchable = influencer.get("searchable_text", "").lower()
            
            # Keyword matching with weights
            if any(keyword in searchable for keyword in query_lower.split()):
                score += 10
            
            # Category matching - FIX: Check if category is string before calling lower()
            category = influencer.get("category")
            if isinstance(category, str) and category.lower() in query_lower:
                score += 20
            
            # Hashtag relevance - FIX: Ensure hashtags exist
            hashtags = influencer.get("top_hashtags", [])
            if hashtags:
                matching_hashtags = sum(1 for hashtag in hashtags 
                                    if isinstance(hashtag, str) and 
                                    any(word in hashtag.lower() for word in query_lower.split()))
                score += matching_hashtags * 5
            
            # Quality bonuses - FIX: Safe get with defaults
            content_quality = influencer.get("content_quality", 0)
            engagement_rate = influencer.get("engagement_rate", 0)
            
            if content_quality:
                score += content_quality * 0.1
            if engagement_rate:
                score += engagement_rate * 100
            
            if score > 0:
                influencer_copy = influencer.copy()
                influencer_copy["relevance_score"] = score
                scored_influencers.append(influencer_copy)
        
        # Sort by relevance score
        scored_influencers.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_influencers[:limit]
    
    def get_formatted_knowledge_context(self, relevant_influencers: List[Dict] = None) -> str:
        """Format knowledge for AI context"""
        if relevant_influencers:
            influencers_to_format = relevant_influencers
        else:
            # Get top influencers by followers if no specific relevance
            influencers_to_format = sorted(self.knowledge_base, 
                                         key=lambda x: x["followers"], reverse=True)[:10]
        
        formatted_knowledge = []
        for inf in influencers_to_format:
            formatted = f"""
            • {inf['name']} (@{inf['username']})
              Category: {inf['category']} | Followers: {inf['followers']:,}
              Engagement: {inf['engagement_rate']:.3f}% | Verified: {'✓' if inf['verified'] else '✗'}
              Bio: {inf['bio'][:100]}...
              Content Focus: {', '.join(inf['top_hashtags'][:8])}
              Recent Brands: {', '.join(inf['brand_collaborations'][:5])}
              Quality Score: {inf['content_quality']}/100
            """
            formatted_knowledge.append(formatted.strip())
        
        return "\n\n".join(formatted_knowledge)
    
    def refresh_knowledge(self):
        """Refresh knowledge base from database"""
        self._load_knowledge()


# Global knowledge manager instance
knowledge_manager = KnowledgeManager()
