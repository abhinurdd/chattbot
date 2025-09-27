import json
from datetime import datetime
from api_clients import (
    message_model,
    search_google,
    scrape_complete_instagram_profile,
    test_apify_connection,
    parse_normalization_response,
    parse_profile_selection_response,
    parse_scoring_response
)
from workflow_logic import (
    build_search_query_and_filter_candidates,
    format_profile_data,
    analyze_instagram_posts,
    aggregate_post_metrics,
    calculate_scores_manually
)
from database import (
    init_db,
    check_profile_exists,
    insert_complete_profile,
    get_profile_from_cache,
    get_database_stats
)
from config import (
    MODEL_NORMALIZER,
    MODEL_SELECTOR,
    MODEL_SCORING
)

def run_workflow(chat_message: str):
    """Main workflow with improved error handling."""
    print(f"\n--- Starting workflow for query: '{chat_message}' ---")
    
    # Validate input
    if not chat_message or not chat_message.strip():
        print("❌ Empty query provided")
        return None
    
    try:
        # Step 1: Normalize query with AI (simplified prompt)
        print("\n[Step 1] Normalizing query with AI...")
        prompt1 = f"""
Analyze this name/query: "{chat_message}"

Respond in this EXACT format only:

Search Name: {chat_message}

Aliases:
- None known

Handles:
- None known

Do not add any extra information or biography.
"""
        
        normalized_text = message_model(prompt1, MODEL_NORMALIZER)
        if not normalized_text:
            print("❌ AI normalization failed, using original query")
            normalized_data = {
                'search_name': chat_message,
                'aliases': [],
                'handles': []
            }
        else:
            normalized_data = parse_normalization_response(normalized_text, chat_message)
        
        print(f"✅ Normalized data: {json.dumps(normalized_data, indent=2)}")
        
        # Step 2: Search Google
        print("\n[Step 2] Searching Google...")
        search_name = normalized_data.get('search_name', chat_message)
        search_results = search_google(search_name)
        
        if not search_results:
            print("❌ Google search failed")
            return None
        
        organic_results = search_results.get('organic', [])
        print(f"✅ Found {len(organic_results)} search results")
        
        # Step 3: Process search candidates
        print("\n[Step 3] Processing search candidates...")
        search_query, candidates = build_search_query_and_filter_candidates(
            normalized_data, chat_message, search_results
        )
        
        print(f"Found {len(candidates)} potential candidates")
        
        if not candidates:
            print("❌ No suitable Instagram profiles found")
            return None
        
        # Step 4: Select official profile with AI
        print("\n[Step 4] Selecting official profile with AI...")
        prompt2 = f"""
Query: "{chat_message}"

Instagram profile candidates found:
{json.dumps(candidates[:3], indent=2)}

Choose the OFFICIAL Instagram profile. Respond in this EXACT format:

Name: [best name for this person]
Username: [instagram username without @]
Instagram URL: [full instagram.com URL]
Confidence: [number between 0 and 1]
"""
        
        selection_text = message_model(prompt2, MODEL_SELECTOR)
        if not selection_text:
            # Fallback to first candidate
            print("❌ AI selection failed, using first candidate")
            first_candidate = candidates[0]
            selected_profile = {
                'name': chat_message,
                'username': first_candidate['url'].split('/')[-1],
                'instagram_url': first_candidate['url'],
                'confidence': 0.5
            }
        else:
            selected_profile = parse_profile_selection_response(selection_text)
        
        if not selected_profile or not selected_profile.get("instagram_url"):
            print("❌ Could not determine Instagram profile")
            return None
        
        print(f"✅ Selected profile: {selected_profile.get('username')} (confidence: {selected_profile.get('confidence', 0.5)})")
        
        # Step 5: Format profile data
        print("\n[Step 5] Formatting profile data...")
        profile_data = format_profile_data(selected_profile, chat_message)
        username = profile_data.get("username")
        
        if not username:
            print("❌ Could not extract username")
            return None
        
        # Step 6: Check cache
        print(f"\n[Step 6] Checking cache for @{username}...")
        exists, cached_data = check_profile_exists(username)
        
        if exists and cached_data:
            print(f"🎯 Using cached data for @{username}")
            display_final_report(cached_data)
            return cached_data
        
        # Step 7: Scrape fresh data
        print(f"\n[Step 7] Scraping fresh data for @{username}...")
        scraped_profile, posts_raw = scrape_complete_instagram_profile(username)
        
        has_profile = scraped_profile is not None
        has_posts = posts_raw and len(posts_raw) > 0
        
        print(f"Scraping results:")
        print(f"  - Profile info: {'✅' if has_profile else '❌'}")
        print(f"  - Posts data: {'✅' if has_posts else '❌'} ({len(posts_raw) if posts_raw else 0} posts)")
        
        if not has_profile and not has_posts:
            print(f"❌ No data found for @{username}")
            return None
        
        # Update profile data with scraped info
        if has_profile:
            profile_data.update(scraped_profile)
            print(f"✅ Profile: {profile_data.get('name', 'N/A')} - {profile_data.get('followers_count', 0):,} followers")
        
        # Process posts if available
        if has_posts:
            print(f"\n[Step 8] Analyzing posts...")
            analyzed_posts = analyze_instagram_posts(posts_raw, username)
            
            print(f"\n[Step 9] Aggregating metrics...")
            metrics = aggregate_post_metrics(analyzed_posts)
            
            print(f"\n[Step 10] Calculating scores...")
            
            # Simplified scoring prompt
            prompt3 = f"""
Based on these Instagram metrics, provide scores:

Username: {username}
Posts Analyzed: {metrics.get('postsAnalyzed', 0)}
Average Engagement Rate: {metrics.get('avgEngagement_all', 0):.4f}
Consistency Score: {metrics.get('consistencyScore', 75):.1f}

Respond in this EXACT format:

Authenticity: [score 0-100]
Brand Safety: [score 0-100]
Audience Match: [score 0-100]
Content Quality: [score 0-100]
"""
            
            scoring_text = message_model(prompt3, MODEL_SCORING)
            if scoring_text:
                scores = parse_scoring_response(scoring_text, username, metrics)
            else:
                print("⚠️ AI scoring failed, using manual calculation")
                scores = calculate_scores_manually(metrics, username)
                
        else:
            print("⚠️ No posts data - creating profile-only entry")
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
        
        # Validate scores (ensure they're not all zero)
        if all(scores.get(key, 0) == 0 for key in ['Authenticity', 'BrandSafety', 'AudienceMatch', 'ContentQuality']):
            print("⚠️ All scores are zero, recalculating...")
            scores = calculate_scores_manually(metrics, username)
        
        # Step 11: Save to database
        print(f"\n[Step 11] Saving to database...")
        success = insert_complete_profile(profile_data, analyzed_posts, metrics, scores)
        
        if success:
            final_data = get_profile_from_cache(username)
            display_final_report(final_data)
            return final_data
        else:
            print("❌ Failed to save to database")
            return None
        
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_final_report(profile_data):
    """Display the final analysis report."""
    if not profile_data:
        print("❌ No profile data to display")
        return
    
    basic_info = profile_data.get("basic_info", {})
    analysis = profile_data.get("analysis", {})
    posts = profile_data.get("posts", {})
    collaborations = profile_data.get("brand_collaborations", {})
    hashtags = profile_data.get("hashtags", {})
    
    print("\n" + "="*60)
    print("              INSTAGRAM ANALYSIS REPORT")
    print("="*60)
    print(f"Username: @{basic_info.get('username', 'N/A')}")
    print(f"Name: {basic_info.get('name', 'N/A')}")
    print(f"Verified: {'✅' if basic_info.get('is_verified') else '❌'}")
    print(f"Business Account: {'✅' if basic_info.get('is_business_account') else '❌'}")
    print("-" * 60)
    print("FOLLOWER STATS")
    print(f"Followers: {basic_info.get('followers_count', 0):,}")
    print(f"Following: {basic_info.get('following_count', 0):,}")
    print(f"Total Posts: {basic_info.get('posts_count', 0):,}")
    print("-" * 60)
    print("POST ANALYSIS")
    print(f"Posts Analyzed: {posts.get('total_posts', 0)}")
    print(f"Organic Posts: {len(posts.get('organic_posts', []))}")
    print(f"Sponsored Posts: {len(posts.get('sponsored_posts', []))}")
    print(f"Avg Likes: {analysis.get('avg_likes', 0):,.0f}")
    print(f"Avg Comments: {analysis.get('avg_comments', 0):,.0f}")
    print(f"Avg Views: {analysis.get('avg_views', 0):,.0f}")
    print(f"Engagement Rate: {analysis.get('engagement_rate', 0):.3f}")
    print("-" * 60)
    print("BRAND COLLABORATIONS")
    print(f"Total Sponsored Posts: {collaborations.get('total_sponsored_posts', 0)}")
    print(f"Brands Worked With: {len(collaborations.get('brands_worked_with', []))}")
    if collaborations.get('brands_worked_with'):
        print("Recent Brands: " + ", ".join(collaborations['brands_worked_with'][:5]))
    print("-" * 60)
    print("CONTENT INSIGHTS")
    print(f"Unique Hashtags Used: {hashtags.get('total_unique', 0)}")
    if hashtags.get('most_used'):
        top_hashtags = list(hashtags['most_used'].keys())[:5]
        print(f"Top Hashtags: {', '.join(f'#{tag}' for tag in top_hashtags)}")
    print("-" * 60)
    print("INFLUENCE SCORES")
    scores = analysis.get('scores', {})
    print(f"Authenticity: {scores.get('Authenticity', 0):.1f}/100")
    print(f"Brand Safety: {scores.get('BrandSafety', 0):.1f}/100")
    print(f"Audience Match: {scores.get('AudienceMatch', 0):.1f}/100")
    print(f"Content Quality: {scores.get('ContentQuality', 0):.1f}/100")
    print("="*60)
    
    metadata = profile_data.get("metadata", {})
    print(f"Data cached on: {metadata.get('last_scraped', 'Unknown')}")
    print("✅ Analysis completed successfully!")

def main():
    """Main function."""
    print("Instagram Influencer Analysis Tool")
    print("=" * 40)
    
    # Initialize database
    try:
        init_db()
        stats = get_database_stats()
        print(f"Database loaded: {stats['total_profiles']} profiles cached")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        return
    
    # Test Apify connection
    test_choice = input("\nTest Apify connection first? (y/n): ").strip().lower()
    if test_choice == 'y':
        if not test_apify_connection():
            print("❌ Apify connection failed. Please check your API token.")
            return
        print("\n" + "-" * 40)
    
    # Get user input
    query = input("\nEnter influencer name or query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return
    
    # Run workflow
    result = run_workflow(query)
    
    if result:
        print(f"\n💾 Profile data saved to database")
        print("🤖 Ready for chatbot integration!")
    else:
        print("\n❌ Workflow failed to complete")

if __name__ == '__main__':
    main()
