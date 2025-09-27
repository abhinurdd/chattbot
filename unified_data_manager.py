import json
import os
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from database import get_database, get_profile_from_cache, insert_complete_profile
from fast_semantic_matcher import fast_semantic_matcher
from api_clients import (
    spell_correct_influencer_name, 
    enhanced_influencer_normalization,
    spell_correct_influencer_name_async,
    search_google_async,
    scrape_complete_instagram_profile_async,
    parse_profile_selection_response,
    message_model_async,
    run_async_in_sync,
    session_manager,
    search_google,
    scrape_complete_instagram_profile,
    message_model
)
# FIX: Add missing imports
from workflow_logic import (
    build_search_query_and_filter_candidates,
    format_profile_data,
    analyze_instagram_posts,
    aggregate_post_metrics,
    calculate_scores_manually
)


class UnifiedDataManager:
    """FIXED: Manages access with spelling correction + auto-scraping + robust async"""
    
    def __init__(self):
        self.main_db_loaded = False
        self.embeddings_loaded = False
        self.auto_scraping_enabled = True
        self._progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self._progress_callback = callback
    
    def _update_progress(self, message: str, percentage: int = 0):
        """Update progress if callback is set"""
        if self._progress_callback:
            try:
                self._progress_callback(message, percentage)
            except:
                pass  # Ignore callback errors
        else:
            print(f"📊 {percentage}% - {message}")
    
    def ensure_all_data_loaded(self):
        """Ensure both databases are loaded"""
        if not self.embeddings_loaded:
            fast_semantic_matcher.load_precomputed_embeddings()
            self.embeddings_loaded = fast_semantic_matcher.is_loaded
        
        self.main_db_loaded = True  # Main DB loads automatically via get_database()
    
    def get_available_influencers(self) -> List[Dict]:
        """Get list of available influencers from both databases for context"""
        self.ensure_all_data_loaded()
        
        influencers = []
        
        # From main database
        try:
            db = get_database()
            profiles = db.get("profiles", {})
            for username, profile_data in profiles.items():
                basic_info = profile_data.get("basic_info", {})
                influencers.append({
                    'username': username,
                    'name': basic_info.get('name', ''),
                    'followers': basic_info.get('followers_count', 0),
                    'source': 'main_db'
                })
        except Exception as e:
            print(f"Error loading main DB for context: {e}")
        
        # From embeddings database
        if self.embeddings_loaded:
            for username, data in fast_semantic_matcher.influencer_data.items():
                # Avoid duplicates
                if not any(inf['username'].lower() == username.lower() for inf in influencers):
                    influencers.append({
                        'username': username,
                        'name': data.get('name', ''),
                        'followers': data.get('followers', 0),
                        'source': 'embeddings'
                    })
        
        return influencers
    
    def find_influencer_anywhere(self, influencer_name: str, auto_scrape: bool = True) -> Tuple[bool, Optional[Dict], str]:
        """ROBUST: Search with fallback to sync if async fails"""
        try:
            return run_async_in_sync(self.find_influencer_anywhere_async(influencer_name, auto_scrape))
        except Exception as e:
            print(f"Async search failed: {e}, falling back to sync")
            return self._find_influencer_anywhere_sync(influencer_name, auto_scrape)
    
    def _find_influencer_anywhere_sync(self, influencer_name: str, auto_scrape: bool = True) -> Tuple[bool, Optional[Dict], str]:
        """Synchronous fallback version of influencer search"""
        self.ensure_all_data_loaded()
        
        print(f"🔍 Searching for: '{influencer_name}' (sync mode)")
        self._update_progress(f"Searching for '{influencer_name}' (sync)", 10)
        
        # Step 1: Get available influencers for AI context
        available_influencers = self.get_available_influencers()
        print(f"📊 Using {len(available_influencers)} influencers for AI context")
        
        # Step 2: AI spelling correction (sync version)
        self._update_progress("AI spell checking and name correction (sync)", 20)
        spelling_result = spell_correct_influencer_name(influencer_name, available_influencers)
        
        if not spelling_result['is_influencer']:
            print(f"❌ AI determined '{influencer_name}' is not an influencer name")
            return False, None, "not_influencer"
        
        corrected_name = spelling_result['corrected_name']
        confidence = spelling_result['confidence']
        reasoning = spelling_result['reasoning']
        
        print(f"🤖 AI Correction: '{influencer_name}' → '{corrected_name}' (confidence: {confidence:.2f})")
        print(f"💡 Reasoning: {reasoning}")
        
        # Step 3: Enhanced normalization
        self._update_progress("Normalizing search terms", 30)
        normalized = enhanced_influencer_normalization(corrected_name, available_influencers)
        search_terms = [
            corrected_name,
            normalized.get('search_name', ''),
            *normalized.get('aliases', []),
            *normalized.get('handles', []),
            influencer_name
        ]
        
        # Remove duplicates and empty strings
        search_terms = list(dict.fromkeys([term.strip() for term in search_terms if term and term.strip()]))
        print(f"🔍 Search terms: {search_terms}")
        
        # Step 4: Try each search term in both databases
        self._update_progress("Searching in local databases", 40)
        for term in search_terms:
            # Try main database first
            found, data, username = self._search_main_database(term)
            if found:
                print(f"✅ Found '{influencer_name}' as '@{username}' in main database")
                self._update_progress(f"Found in main database: @{username}", 100)
                return True, data, f"main_db:@{username}"
            
            # Try embeddings database
            found, data, username = self._search_embeddings_database(term)
            if found:
                print(f"✅ Found '{influencer_name}' as '@{username}' in embeddings database")
                self._update_progress(f"Found in embeddings database: @{username}", 100)
                return True, data, f"embeddings:@{username}"
        
        # Step 5: AUTO-SCRAPING (sync version)
        if auto_scrape and self.auto_scraping_enabled:
            print(f"🤖 '{influencer_name}' not found in databases - attempting auto-scraping (sync)...")
            self._update_progress("Not found locally - starting auto-scraping (sync)", 45)
            return self._auto_scrape_and_save_sync(influencer_name, corrected_name, normalized)
        
        print(f"❌ No matches found for '{influencer_name}' or '{corrected_name}'")
        self._update_progress("No matches found", 100)
        return False, None, "not_found"
    
    async def find_influencer_anywhere_async(self, influencer_name: str, auto_scrape: bool = True) -> Tuple[bool, Optional[Dict], str]:
        """ASYNC OPTIMIZED: Search with AI spelling correction + automatic scraping"""
        self.ensure_all_data_loaded()
        
        print(f"🔍 Searching for: '{influencer_name}'")
        self._update_progress(f"Searching for '{influencer_name}'", 10)
        
        # Step 1: Get available influencers for AI context
        available_influencers = self.get_available_influencers()
        print(f"📊 Using {len(available_influencers)} influencers for AI context")
        
        # Step 2: AI spelling correction (async)
        self._update_progress("AI spell checking and name correction", 20)
        try:
            spelling_result = await spell_correct_influencer_name_async(influencer_name, available_influencers)
        except Exception as e:
            print(f"Async spelling correction failed: {e}, using sync")
            spelling_result = spell_correct_influencer_name(influencer_name, available_influencers)
        
        if not spelling_result['is_influencer']:
            print(f"❌ AI determined '{influencer_name}' is not an influencer name")
            return False, None, "not_influencer"
        
        corrected_name = spelling_result['corrected_name']
        confidence = spelling_result['confidence']
        reasoning = spelling_result['reasoning']
        
        print(f"🤖 AI Correction: '{influencer_name}' → '{corrected_name}' (confidence: {confidence:.2f})")
        print(f"💡 Reasoning: {reasoning}")
        
        # Step 3: Enhanced normalization with database context
        self._update_progress("Normalizing search terms", 30)
        normalized = enhanced_influencer_normalization(corrected_name, available_influencers)
        search_terms = [
            corrected_name,
            normalized.get('search_name', ''),
            *normalized.get('aliases', []),
            *normalized.get('handles', []),
            influencer_name  # Also try original
        ]
        
        # Remove duplicates and empty strings
        search_terms = list(dict.fromkeys([term.strip() for term in search_terms if term and term.strip()]))
        print(f"🔍 Search terms: {search_terms}")
        
        # Step 4: Try each search term in both databases
        self._update_progress("Searching in local databases", 40)
        for term in search_terms:
            # Try main database first
            found, data, username = self._search_main_database(term)
            if found:
                print(f"✅ Found '{influencer_name}' as '@{username}' in main database")
                self._update_progress(f"Found in main database: @{username}", 100)
                return True, data, f"main_db:@{username}"
            
            # Try embeddings database
            found, data, username = self._search_embeddings_database(term)
            if found:
                print(f"✅ Found '{influencer_name}' as '@{username}' in embeddings database")
                self._update_progress(f"Found in embeddings database: @{username}", 100)
                return True, data, f"embeddings:@{username}"
        
        # Step 5: AUTO-SCRAPING with async optimizations
        if auto_scrape and self.auto_scraping_enabled:
            print(f"🤖 '{influencer_name}' not found in databases - attempting auto-scraping...")
            self._update_progress("Not found locally - starting auto-scraping", 45)
            try:
                return await self._auto_scrape_and_save_async(influencer_name, corrected_name, normalized)
            except Exception as e:
                print(f"Async auto-scraping failed: {e}, trying sync")
                return self._auto_scrape_and_save_sync(influencer_name, corrected_name, normalized)
        
        print(f"❌ No matches found for '{influencer_name}' or '{corrected_name}'")
        self._update_progress("No matches found", 100)
        return False, None, "not_found"
    
    def _auto_scrape_and_save_sync(self, original_name: str, corrected_name: str, normalized_data: dict) -> Tuple[bool, Optional[Dict], str]:
        """Synchronous version of auto-scraping"""
        from config import MODEL_SELECTOR
        
        try:
            print(f"🕷️ Starting sync auto-scraping workflow for '{original_name}'...")
            self._update_progress("Starting auto-scraping workflow (sync)", 50)
            
            # Step 1: Google search
            self._update_progress("Searching Google for Instagram profiles", 55)
            search_name = normalized_data.get('search_name', corrected_name)
            search_results = search_google(search_name)
            
            if not search_results:
                print("❌ Google search failed during auto-scraping")
                self._update_progress("Google search failed", 100)
                return False, None, "scraping_failed"
            
            # Step 2: Find Instagram candidates
            self._update_progress("Analyzing Instagram profile candidates", 60)
            search_query, candidates = build_search_query_and_filter_candidates(
                normalized_data, original_name, search_results
            )
            
            if not candidates:
                print("❌ No Instagram candidates found during auto-scraping")
                self._update_progress("No Instagram profiles found", 100)
                return False, None, "scraping_failed"
            
            print(f"🎯 Found {len(candidates)} Instagram candidates")
            
            # Step 3: AI selects official profile
            self._update_progress("AI selecting official profile", 65)
            selection_prompt = f"""
Query: "{original_name}" (corrected: "{corrected_name}")

Instagram profile candidates found:
{json.dumps(candidates[:3], indent=2)}

Choose the OFFICIAL Instagram profile that matches "{corrected_name}". 

Respond in this EXACT format:

Name: [best name for this person]
Username: [instagram username without @]
Instagram URL: [full instagram.com URL]
Confidence: [number between 0 and 1]
"""
            
            selection_text = message_model(selection_prompt, MODEL_SELECTOR)
            if not selection_text:
                print("❌ AI profile selection failed during auto-scraping")
                self._update_progress("AI profile selection failed", 100)
                return False, None, "scraping_failed"
            
            selected_profile = parse_profile_selection_response(selection_text)
            
            if not selected_profile or not selected_profile.get("instagram_url"):
                print("❌ Could not determine Instagram profile during auto-scraping")
                self._update_progress("Could not determine profile", 100)
                return False, None, "scraping_failed"
            
            # Step 4: Format and extract username
            profile_data = format_profile_data(selected_profile, original_name)
            username = profile_data.get("username")
            
            if not username:
                print("❌ Could not extract username during auto-scraping")
                self._update_progress("Could not extract username", 100)
                return False, None, "scraping_failed"
            
            print(f"🎯 Selected profile: @{username} (confidence: {selected_profile.get('confidence', 0.5)})")
            self._update_progress(f"Selected profile: @{username}", 70)
            
            # Step 5: Scrape profile and posts (sync)
            print(f"🚀 Starting scraping for @{username}...")
            self._update_progress(f"Scraping profile and posts for @{username}", 75)
            
            scraped_profile, posts_raw = scrape_complete_instagram_profile(username)
            
            has_profile = scraped_profile is not None
            has_posts = posts_raw and len(posts_raw) > 0
            
            print(f"Scraping results:")
            print(f"  - Profile info: {'✅' if has_profile else '❌'}")
            print(f"  - Posts data: {'✅' if has_posts else '❌'} ({len(posts_raw) if posts_raw else 0} posts)")
            
            if not has_profile:
                print(f"❌ Failed to scrape profile data for @{username}")
                self._update_progress("Failed to scrape profile data", 100)
                return False, None, "scraping_failed"
            
            # Step 6: Update profile data
            profile_data.update(scraped_profile)
            print(f"✅ Profile: {profile_data.get('name', 'N/A')} - {profile_data.get('followers_count', 0):,} followers")
            self._update_progress(f"Found: {profile_data.get('name', 'N/A')} - {profile_data.get('followers_count', 0):,} followers", 85)
            
            # Step 7: Analyze posts and calculate scores
            if has_posts:
                print(f"📊 Analyzing {len(posts_raw)} posts...")
                self._update_progress(f"Analyzing {len(posts_raw)} posts", 90)
                
                analyzed_posts = analyze_instagram_posts(posts_raw, username)
                metrics = aggregate_post_metrics(analyzed_posts)
                scores = calculate_scores_manually(metrics, username)
            else:
                print("⚠️ No posts data - creating profile-only entry")
                self._update_progress("No posts found - creating profile-only entry", 90)
                analyzed_posts = []
                metrics = {}
                scores = {
                    "username": username,
                    "postsAnalyzed": 0,
                    "avgER_organic": 0,
                    "avgER_sponsored": 0,
                    "avgLikes": 0,
                    "avg_sponsored_likes": 0,
                    "avgComments": 0,
                    "avg_sponsored_comments": 0,
                    "avgViews": 0,
                    "Authenticity": 70.0,
                    "BrandSafety": 85.0,
                    "AudienceMatch": 60.0,
                    "ContentQuality": 65.0,
                }
            
            # Step 8: Save to database
            print(f"💾 Saving auto-scraped data to database...")
            self._update_progress("Saving to database", 95)
            success = insert_complete_profile(profile_data, analyzed_posts, metrics, scores)
            
            if success:
                print(f"✅ Successfully auto-scraped and saved @{username}")
                self._update_progress(f"Successfully saved @{username}", 100)
                
                # Retrieve saved data and return
                final_data = get_profile_from_cache(username)
                if final_data:
                    return True, final_data, f"auto_scraped:@{username}"
                else:
                    print("❌ Failed to retrieve saved data")
                    return False, None, "scraping_failed"
            else:
                print(f"❌ Failed to save auto-scraped data for @{username}")
                self._update_progress("Failed to save data", 100)
                return False, None, "scraping_failed"
                
        except Exception as e:
            print(f"❌ Sync auto-scraping failed with error: {e}")
            import traceback
            traceback.print_exc()
            self._update_progress(f"Auto-scraping failed: {str(e)}", 100)
            return False, None, "scraping_failed"
    
    async def _auto_scrape_and_save_async(self, original_name: str, corrected_name: str, normalized_data: dict) -> Tuple[bool, Optional[Dict], str]:
        """ASYNC OPTIMIZED: Automatically scrape influencer data and save to database"""  
        from config import MODEL_SELECTOR
        
        try:
            print(f"🕷️ Starting async auto-scraping workflow for '{original_name}'...")
            self._update_progress("Starting auto-scraping workflow", 50)
            
            # Step 1: Async Google search for Instagram profile
            self._update_progress("Searching Google for Instagram profiles", 55)
            search_name = normalized_data.get('search_name', corrected_name)
            search_results = await search_google_async(search_name)
            
            if not search_results:
                print("❌ Google search failed during auto-scraping")
                self._update_progress("Google search failed", 100)
                return False, None, "scraping_failed"
            
            # Step 2: Find Instagram candidates
            self._update_progress("Analyzing Instagram profile candidates", 60)
            search_query, candidates = build_search_query_and_filter_candidates(
                normalized_data, original_name, search_results
            )
            
            if not candidates:
                print("❌ No Instagram candidates found during auto-scraping")
                self._update_progress("No Instagram profiles found", 100)
                return False, None, "scraping_failed"
            
            print(f"🎯 Found {len(candidates)} Instagram candidates")
            
            # Step 3: AI selects official profile (async)
            self._update_progress("AI selecting official profile", 65)
            selection_prompt = f"""
Query: "{original_name}" (corrected: "{corrected_name}")

Instagram profile candidates found:
{json.dumps(candidates[:3], indent=2)}

Choose the OFFICIAL Instagram profile that matches "{corrected_name}". 

Respond in this EXACT format:

Name: [best name for this person]
Username: [instagram username without @]
Instagram URL: [full instagram.com URL]
Confidence: [number between 0 and 1]
"""
            
            try:
                session = await session_manager.get_session()
                selection_text = await message_model_async(session, selection_prompt, MODEL_SELECTOR)
            except Exception as e:
                print(f"Async AI call failed: {e}, trying sync")
                selection_text = message_model(selection_prompt, MODEL_SELECTOR)
            
            if not selection_text:
                print("❌ AI profile selection failed during auto-scraping")
                self._update_progress("AI profile selection failed", 100)
                return False, None, "scraping_failed"
            
            selected_profile = parse_profile_selection_response(selection_text)
            
            if not selected_profile or not selected_profile.get("instagram_url"):
                print("❌ Could not determine Instagram profile during auto-scraping")
                self._update_progress("Could not determine profile", 100)
                return False, None, "scraping_failed"
            
            # Step 4: Format and extract username
            profile_data = format_profile_data(selected_profile, original_name)
            username = profile_data.get("username")
            
            if not username:
                print("❌ Could not extract username during auto-scraping")
                self._update_progress("Could not extract username", 100)
                return False, None, "scraping_failed"
            
            print(f"🎯 Selected profile: @{username} (confidence: {selected_profile.get('confidence', 0.5)})")
            self._update_progress(f"Selected profile: @{username}", 70)
            
            # Step 5: PARALLEL ASYNC SCRAPING (Major optimization!)
            print(f"🚀 Starting parallel scraping for @{username}...")
            self._update_progress(f"Scraping profile and posts for @{username}", 75)
            
            scraped_profile, posts_raw = await scrape_complete_instagram_profile_async(username)
            
            has_profile = scraped_profile is not None
            has_posts = posts_raw and len(posts_raw) > 0
            
            print(f"Scraping results:")
            print(f"  - Profile info: {'✅' if has_profile else '❌'}")
            print(f"  - Posts data: {'✅' if has_posts else '❌'} ({len(posts_raw) if posts_raw else 0} posts)")
            
            if not has_profile:
                print(f"❌ Failed to scrape profile data for @{username}")
                self._update_progress("Failed to scrape profile data", 100)
                return False, None, "scraping_failed"
            
            # Step 6: Update profile data with scraped info
            profile_data.update(scraped_profile)
            print(f"✅ Profile: {profile_data.get('name', 'N/A')} - {profile_data.get('followers_count', 0):,} followers")
            self._update_progress(f"Found: {profile_data.get('name', 'N/A')} - {profile_data.get('followers_count', 0):,} followers", 85)
            
            # Step 7: Analyze posts and calculate scores (background processing)
            if has_posts:
                print(f"📊 Analyzing {len(posts_raw)} posts...")
                self._update_progress(f"Analyzing {len(posts_raw)} posts", 90)
                
                # Run analysis in background task for better UX
                analysis_task = asyncio.create_task(
                    self._analyze_posts_async(posts_raw, username),
                    name=f"analysis_{username}"
                )
                
                analyzed_posts, metrics, scores = await analysis_task
            else:
                print("⚠️ No posts data - creating profile-only entry")
                self._update_progress("No posts found - creating profile-only entry", 90)
                analyzed_posts = []
                metrics = {}
                scores = {
                    "username": username,
                    "postsAnalyzed": 0,
                    "avgER_organic": 0,
                    "avgER_sponsored": 0,
                    "avgLikes": 0,
                    "avg_sponsored_likes": 0,
                    "avgComments": 0,
                    "avg_sponsored_comments": 0,
                    "avgViews": 0,
                    "Authenticity": 70.0,
                    "BrandSafety": 85.0,
                    "AudienceMatch": 60.0,
                    "ContentQuality": 65.0,
                }
            
            # Step 8: Save to database
            print(f"💾 Saving auto-scraped data to database...")
            self._update_progress("Saving to database", 95)
            success = insert_complete_profile(profile_data, analyzed_posts, metrics, scores)
            
            if success:
                print(f"✅ Successfully auto-scraped and saved @{username}")
                self._update_progress(f"Successfully saved @{username}", 100)
                
                # Retrieve saved data and return
                final_data = get_profile_from_cache(username)
                if final_data:
                    return True, final_data, f"auto_scraped:@{username}"
                else:
                    print("❌ Failed to retrieve saved data")
                    return False, None, "scraping_failed"
            else:
                print(f"❌ Failed to save auto-scraped data for @{username}")
                self._update_progress("Failed to save data", 100)
                return False, None, "scraping_failed"
                
        except Exception as e:
            print(f"❌ Async auto-scraping failed with error: {e}")
            import traceback
            traceback.print_exc()
            self._update_progress(f"Auto-scraping failed: {str(e)}", 100)
            return False, None, "scraping_failed"
    
    async def _analyze_posts_async(self, posts_raw: List[Dict], username: str) -> Tuple[List[Dict], Dict, Dict]:
        """Async post analysis for background processing"""
        try:
            # Create async task for analysis
            analysis_task = asyncio.create_task(
                asyncio.to_thread(analyze_instagram_posts, posts_raw, username),
                name=f"analyze_posts_{username}"
            )
            
            analyzed_posts = await analysis_task
            
            # Aggregate metrics
            metrics_task = asyncio.create_task(
                asyncio.to_thread(aggregate_post_metrics, analyzed_posts),
                name=f"aggregate_metrics_{username}"
            )
            
            metrics = await metrics_task
            
            # Calculate scores
            scores_task = asyncio.create_task(
                asyncio.to_thread(calculate_scores_manually, metrics, username),
                name=f"calculate_scores_{username}"
            )
            
            scores = await scores_task
            
            return analyzed_posts, metrics, scores
            
        except Exception as e:
            print(f"Error in async post analysis: {e}")
            # Fallback to sync
            try:
                analyzed_posts = analyze_instagram_posts(posts_raw, username)
                metrics = aggregate_post_metrics(analyzed_posts)
                scores = calculate_scores_manually(metrics, username)
                return analyzed_posts, metrics, scores
            except:
                return [], {}, {}
    
    def _search_main_database(self, search_term: str) -> Tuple[bool, Optional[Dict], str]:
        """Enhanced search in main database with fuzzy matching"""
        
        # Direct username lookup
        cached_data = get_profile_from_cache(search_term.lower())
        if cached_data:
            return True, cached_data, search_term.lower()
        
        # Try username variations
        username_variations = self._generate_username_variations(search_term)
        for variation in username_variations:
            cached_data = get_profile_from_cache(variation)
            if cached_data:
                return True, cached_data, variation
        
        # Search by name in all profiles with fuzzy matching
        try:
            db = get_database()
            profiles = db.get("profiles", {})
            
            search_lower = search_term.lower()
            
            for username, profile_data in profiles.items():
                basic_info = profile_data.get("basic_info", {})
                stored_name = basic_info.get("name", "").lower()
                stored_full_name = basic_info.get("full_name", "").lower()
                
                # Exact match
                if search_lower == stored_name or search_lower == stored_full_name:
                    return True, profile_data, username
                
                # Fuzzy match (contains)
                if stored_name and (search_lower in stored_name or stored_name in search_lower):
                    return True, profile_data, username
                
                if stored_full_name and (search_lower in stored_full_name or stored_full_name in search_lower):
                    return True, profile_data, username
                
                # Check if any word matches
                search_words = search_lower.split()
                name_words = stored_name.split() + stored_full_name.split()
                if len(search_words) > 1 and any(word in name_words for word in search_words if len(word) > 2):
                    return True, profile_data, username
        
        except Exception as e:
            print(f"Main DB search error: {e}")
        
        return False, None, ""
    
    def _search_embeddings_database(self, search_term: str) -> Tuple[bool, Optional[Dict], str]:
        """Enhanced search in embeddings database with fuzzy matching"""
        
        if not self.embeddings_loaded or not fast_semantic_matcher.is_loaded:
            return False, None, ""
        
        search_lower = search_term.lower()
        username_variations = self._generate_username_variations(search_term)
        
        for username, influencer_data in fast_semantic_matcher.influencer_data.items():
            stored_name = influencer_data.get('name', '').lower()
            username_lower = username.lower()
            
            # Direct username match
            if username_lower == search_lower:
                converted_data = self._convert_embeddings_to_main_format(influencer_data, username)
                return True, converted_data, username
            
            # Username variations match
            if username_lower in [v.lower() for v in username_variations]:
                converted_data = self._convert_embeddings_to_main_format(influencer_data, username)
                return True, converted_data, username
            
            # Name fuzzy matching
            if stored_name:
                # Exact match
                if search_lower == stored_name:
                    converted_data = self._convert_embeddings_to_main_format(influencer_data, username)
                    return True, converted_data, username
                
                # Contains match
                if search_lower in stored_name or stored_name in search_lower:
                    converted_data = self._convert_embeddings_to_main_format(influencer_data, username)
                    return True, converted_data, username
                
                # Word-level matching
                search_words = search_lower.split()
                name_words = stored_name.split()
                if len(search_words) > 1 and any(word in name_words for word in search_words if len(word) > 2):
                    converted_data = self._convert_embeddings_to_main_format(influencer_data, username)
                    return True, converted_data, username
        
        return False, None, ""
    
    def _generate_username_variations(self, name: str) -> List[str]:
        """Generate common username variations"""
        name_clean = name.lower().strip()
        variations = [
            name_clean.replace(" ", ""),
            name_clean.replace(" ", "."),
            name_clean.replace(" ", "_"),
            name_clean.replace(" ", "") + "22",
            name_clean.replace(" ", "") + "official",
            name_clean.replace(" ", "") + "real",
            "official" + name_clean.replace(" ", ""),
            "real" + name_clean.replace(" ", ""),
            name_clean.replace(" ", "") + "vlogs",
            name_clean.replace(" ", "") + "youtube"
        ]
        
        # Remove duplicates and return unique variations
        return list(dict.fromkeys(variations))
    
    def _convert_embeddings_to_main_format(self, embeddings_data: Dict, username: str) -> Dict:
        """Convert embeddings format to main database format"""
        
        return {
            "basic_info": {
                "username": username,
                "name": embeddings_data.get('name', 'Unknown'),
                "full_name": embeddings_data.get('name', 'Unknown'),
                "instagram_url": f"https://instagram.com/{username}/",
                "bio": embeddings_data.get('bio', ''),
                "website": "",
                "profile_pic_url": "",
                "is_verified": embeddings_data.get('verified', False),
                "is_business_account": False,
                "category": embeddings_data.get('category', 'Unknown'),
                "followers_count": embeddings_data.get('followers', 0),
                "following_count": 0,
                "posts_count": 0
            },
            "posts": {
                "total_posts": 0,
                "organic_posts": [],
                "sponsored_posts": [],
                "all_posts": []
            },
            "analysis": {
                "metrics": {},
                "scores": {
                    "Authenticity": embeddings_data.get('authenticity', 75),
                    "BrandSafety": embeddings_data.get('brand_safety', 85),
                    "AudienceMatch": embeddings_data.get('audience_match', 70),
                    "ContentQuality": embeddings_data.get('content_quality', 75)
                },
                "engagement_rate": embeddings_data.get('engagement_rate', 0),
                "avg_likes": 0,
                "avg_comments": 0,
                "avg_views": 0
            },
            "brand_collaborations": {
                "total_sponsored_posts": 0,
                "brands_worked_with": embeddings_data.get('brand_collaborations', []),
                "sponsored_posts": []
            },
            "hashtags": {
                "most_used": {tag: 1 for tag in embeddings_data.get('top_hashtags', [])[:10]},
                "total_unique": len(embeddings_data.get('top_hashtags', []))
            },
            "mentions": {
                "most_mentioned": {},
                "total_mentions": 0
            },
            "metadata": {
                "last_scraped": "From embeddings database",
                "scraping_source": "embeddings",
                "analysis_version": "1.0",
                "original_query": "",
                "spelling_corrected": True
            }
        }
    
    def get_database_stats(self) -> Dict:
        """Get combined statistics from both databases"""
        self.ensure_all_data_loaded()
        
        main_db_count = 0
        embeddings_count = 0
        
        try:
            from database import get_database_stats
            main_stats = get_database_stats()
            main_db_count = main_stats.get('total_profiles', 0)
        except:
            pass
        
        if self.embeddings_loaded:
            embeddings_count = len(fast_semantic_matcher.influencer_data)
        
        return {
            'main_database': main_db_count,
            'embeddings_database': embeddings_count,
            'total_unique': max(main_db_count, embeddings_count),  # Rough estimate
            'both_loaded': self.main_db_loaded and self.embeddings_loaded,
            'auto_scraping': self.auto_scraping_enabled
        }
    
    def toggle_auto_scraping(self, enabled: bool):
        """Toggle auto-scraping functionality"""
        self.auto_scraping_enabled = enabled
        print(f"🕷️ Auto-scraping {'enabled' if enabled else 'disabled'}")


# Global instance
unified_data_manager = UnifiedDataManager()
