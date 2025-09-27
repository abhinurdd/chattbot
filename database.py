import json
import os
from datetime import datetime, timedelta
from config import DATABASE_FILE

def get_database():
    """Load the JSON database."""
    if not os.path.exists(DATABASE_FILE):
        return {"profiles": {}, "metadata": {"last_updated": datetime.now().isoformat()}}
    
    try:
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading database: {e}")
        return {"profiles": {}, "metadata": {"last_updated": datetime.now().isoformat()}}

def save_database(data):
    """Save the JSON database."""
    try:
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving database: {e}")
        return False

def init_db():
    """Initialize the JSON database."""
    print("Initializing JSON database...")
    if not os.path.exists(DATABASE_FILE):
        initial_data = {
            "profiles": {},
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_profiles": 0,
                "version": "1.0"
            }
        }
        save_database(initial_data)
    print("JSON database initialized successfully.")

def check_profile_exists(username):
    """Check if a profile already exists in the database."""
    db = get_database()
    profile_data = db["profiles"].get(username.lower())
    
    if not profile_data:
        return False, None
    
    try:
        last_scraped = datetime.fromisoformat(profile_data.get("metadata", {}).get("last_scraped", ""))
        if datetime.now() - last_scraped < timedelta(days=7):
            print(f"✅ Found recent data for @{username} in database (scraped {last_scraped.strftime('%Y-%m-%d')})")
            return True, profile_data
        else:
            print(f"⚠️ Found stale data for @{username} in database (scraped {last_scraped.strftime('%Y-%m-%d')}), will refresh")
            return False, profile_data
    except:
        print(f"⚠️ Found data for @{username} but timestamp invalid, will refresh")
        return False, profile_data

def insert_complete_profile(profile_info, posts_data, metrics_data, scores_data):
    """Insert or update a complete profile with all scraped data."""
    username = profile_info.get("username", "").lower()
    
    if not username:
        print("❌ No username provided for database insertion")
        return False
    
    db = get_database()
    
    profile_data = {
        "basic_info": {
            "username": profile_info.get("username", ""),
            "name": profile_info.get("name", ""),
            "full_name": profile_info.get("full_name", ""),
            "instagram_url": profile_info.get("instagram_url", ""),
            "bio": profile_info.get("bio", ""),
            "website": profile_info.get("website", ""),
            "profile_pic_url": profile_info.get("profile_pic_url", ""),
            "is_verified": profile_info.get("is_verified", False),
            "is_business_account": profile_info.get("is_business_account", False),
            "category": profile_info.get("category", ""),
            "followers_count": profile_info.get("followers_count", 0),
            "following_count": profile_info.get("following_count", 0),
            "posts_count": profile_info.get("posts_count", 0)
        },
        "posts": {
            "total_posts": len(posts_data) if posts_data else 0,
            "organic_posts": [],
            "sponsored_posts": [],
            "all_posts": posts_data or []
        },
        "analysis": {
            "metrics": metrics_data or {},
            "scores": scores_data or {},
            "engagement_rate": metrics_data.get("avgEngagement_all", 0) if metrics_data else 0,
            "avg_likes": metrics_data.get("avgLikes", 0) if metrics_data else 0,
            "avg_comments": metrics_data.get("avgComments", 0) if metrics_data else 0,
            "avg_views": metrics_data.get("avgViews", 0) if metrics_data else 0
        },
        "brand_collaborations": {
            "total_sponsored_posts": 0,
            "brands_worked_with": [],
            "sponsored_posts": []
        },
        "hashtags": {
            "most_used": {},
            "total_unique": 0
        },
        "mentions": {
            "most_mentioned": {},
            "total_mentions": 0
        },
        "metadata": {
            "last_scraped": datetime.now().isoformat(),
            "scraping_source": "apify",
            "analysis_version": "1.0",
            "original_query": profile_info.get("input", "")
        }
    }
    
    if posts_data:
        hashtag_counts = {}
        mention_counts = {}
        brands_worked_with = set()
        
        for post in posts_data:
            if post.get("isAd", False):
                profile_data["posts"]["sponsored_posts"].append(post)
                profile_data["brand_collaborations"]["sponsored_posts"].append(post)
                
                for mention in post.get("mentions", []):
                    if isinstance(mention, str):
                        brands_worked_with.add(mention)
            else:
                profile_data["posts"]["organic_posts"].append(post)
            
            for hashtag in post.get("hashtags", []):
                if isinstance(hashtag, str):
                    hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
            
            for mention in post.get("mentions", []):
                if isinstance(mention, str):
                    mention_counts[mention] = mention_counts.get(mention, 0) + 1
        
        profile_data["brand_collaborations"]["total_sponsored_posts"] = len(profile_data["posts"]["sponsored_posts"])
        profile_data["brand_collaborations"]["brands_worked_with"] = list(brands_worked_with)
        
        profile_data["hashtags"]["most_used"] = dict(sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        profile_data["hashtags"]["total_unique"] = len(hashtag_counts)
        
        profile_data["mentions"]["most_mentioned"] = dict(sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        profile_data["mentions"]["total_mentions"] = sum(mention_counts.values())
    
    db["profiles"][username] = profile_data
    db["metadata"]["total_profiles"] = len(db["profiles"])
    
    if save_database(db):
        print(f"✅ Complete profile data saved for @{username}")
        print(f"   - Basic info: ✅")
        print(f"   - Posts: {len(posts_data)} total" if posts_data else "   - Posts: 0")
        print(f"   - Organic posts: {len(profile_data['posts']['organic_posts'])}")
        print(f"   - Sponsored posts: {len(profile_data['posts']['sponsored_posts'])}")
        print(f"   - Brands worked with: {len(profile_data['brand_collaborations']['brands_worked_with'])}")
        print(f"   - Unique hashtags: {profile_data['hashtags']['total_unique']}")
        return True
    else:
        print(f"❌ Failed to save profile data for @{username}")
        return False

def get_profile_from_cache(username):
    """Get complete profile data from cache."""
    db = get_database()
    return db["profiles"].get(username.lower())

def get_database_stats():
    """Get database statistics."""
    db = get_database()
    return {
        "total_profiles": len(db["profiles"]),
        "last_updated": db["metadata"].get("last_updated"),
        "profiles": list(db["profiles"].keys())
    }
