import re
import json
from urllib.parse import urlparse
from datetime import datetime
import statistics

def safe_get_int(data, key, default=0):
    """Safely extract integer values from data."""
    try:
        value = data.get(key, default)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_get_float(data, key, default=0.0):
    """Safely extract float values from data."""
    try:
        value = data.get(key, default)
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_parse_timestamp(timestamp_str):
    """Safely parse timestamp strings into datetime objects."""
    if not timestamp_str:
        return None
    
    try:
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str.replace('Z', '+00:00')
        elif '+' not in timestamp_str and 'T' in timestamp_str:
            timestamp_str += '+00:00'
            
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError) as e:
        print(f"Error parsing timestamp '{timestamp_str}': {e}")
        return None

def build_search_query_and_filter_candidates(norm_data, fallback_query, search_results):
    """Combines logic from n8n steps 4 and 6."""
    search_name = norm_data.get('search_name', '').strip()
    aliases = norm_data.get('aliases', []) or []
    handles = norm_data.get('handles', []) or []
    
    terms = list(dict.fromkeys([
        term for term in [search_name, *aliases, *handles] if term and term.strip()
    ]))
    
    search_query = ' OR '.join(terms) if terms else fallback_query
    
    organic_results = search_results.get('organic', [])
    seen = set()
    candidates = []
    
    for item in organic_results:
        try:
            url = str(item.get('link', '')).split('?')[0].rstrip('/')
            if not url:
                continue
                
            parsed_url = urlparse(url)
            
            if not parsed_url.hostname or 'instagram.com' not in parsed_url.hostname:
                continue
            
            path_segments = [seg for seg in parsed_url.path.split('/') if seg]
            if len(path_segments) != 1:
                continue
            
            user = path_segments[0]
            if not re.match(r"^[a-z0-9._]+$", user, re.IGNORECASE):
                continue
                
            if user.lower() in seen:
                continue
            
            seen.add(user.lower())
            candidates.append({
                "title": str(item.get('title', '')),
                "url": url,
                "snippet": str(item.get('snippet', ''))
            })
            
        except Exception as e:
            print(f"Error processing search result: {e}")
            continue
    
    if not candidates and organic_results:
        first_ig = next((i for i in organic_results if 'instagram.com/' in str(i.get('link', ''))), None)
        if first_ig:
            candidates = [{
                "title": first_ig.get('title', ''),
                "url": str(first_ig.get('link', '')).split('?')[0].rstrip('/'),
                "snippet": first_ig.get('snippet', ''),
                "fallback": True
            }]
            
    return search_query, candidates

def format_profile_data(selected_profile, original_query):
    """Parses AI response and formats the final profile data."""
    url = selected_profile.get('instagram_url', '')
    username = None
    
    if url:
        if not re.match(r"^https?:\/\/", url, re.IGNORECASE):
            url = 'https://' + url.lstrip('/')
        if not url.endswith('/'):
            url += '/'
        
        match = re.search(r"instagram\.com\/([^\/?#]+)", url, re.IGNORECASE)
        if match:
            username = match.group(1)
    
    return {
        "input": original_query,
        "name": selected_profile.get("name", ""),
        "username": username,
        "instagram_url": url,
        "confidence": safe_get_float(selected_profile, "confidence", 0),
        "reason": selected_profile.get("reason", ""),
        "timestamp": datetime.now().isoformat()
    }

def analyze_instagram_posts(posts_data, username):
    """Analyzes raw post data from Apify."""
    if not posts_data:
        print("No posts data to analyze.")
        return []
    
    results = []
    
    for post in posts_data:
        try:
            top_comment = {"text": "No comments found", "likes": 0}
            latest_comments = post.get('latestComments', [])
            
            if latest_comments and isinstance(latest_comments, list):
                try:
                    best_comment = max(latest_comments, 
                                     key=lambda c: safe_get_int(c, 'likesCount', 0), 
                                     default=None)
                    if best_comment:
                        top_comment = {
                            "text": best_comment.get('text', ''),
                            "likes": safe_get_int(best_comment, 'likesCount', 0)
                        }
                except Exception as e:
                    print(f"Error processing comments: {e}")
            
            hashtags = post.get('hashtags', [])
            if not isinstance(hashtags, list):
                hashtags = []
            
            is_ad = (
                any(h.lower() == 'ad' for h in hashtags if isinstance(h, str)) or 
                post.get('paidPartnership', False) or
                post.get('isSponsored', False)
            )
            
            mentions = post.get('mentions', [])
            if not isinstance(mentions, list):
                mentions = []
            
            caption = post.get('caption', '') or post.get('text', '') or "No caption"
            topic = caption.split('\n')[0][:100]
            
            post_data = {
                "username": username,
                "ownerUsername": post.get('ownerUsername', username),
                "postId": str(post.get('id', '')),
                "postType": post.get('type', 'unknown'),
                "postUrl": post.get('url', ''),
                "timestamp": post.get('timestamp', datetime.now().isoformat()),
                "topic": topic,
                "likesCount": safe_get_int(post, 'likesCount', 0),
                "commentsCount": safe_get_int(post, 'commentsCount', 0),
                "videoViewCount": safe_get_int(post, 'videoViewCount', 0),
                "videoPlayCount": safe_get_int(post, 'videoPlayCount', 0),
                "topComment": top_comment,
                "isAd": is_ad,
                "mentions": mentions,
                "hashtags": hashtags,
            }
            
            results.append(post_data)
            
        except Exception as e:
            print(f"Error analyzing post: {e}")
            continue
    
    print(f"Successfully analyzed {len(results)} posts out of {len(posts_data)} total posts.")
    return results

def aggregate_post_metrics(posts):
    """Aggregates metrics from analyzed posts."""
    if not posts:
        print("No posts to aggregate.")
        return {}
    
    try:
        analyzed_posts = []
        
        for p in posts:
            likes = p.get('likesCount', 0)
            comments = p.get('commentsCount', 0)
            views = max(p.get('videoViewCount', 0), p.get('videoPlayCount', 0))
            
            er = (likes + comments) / views if views > 0 else (likes + comments) / max(likes, 1)
            
            timestamp_obj = safe_parse_timestamp(p.get('timestamp'))
            timestamp_unix = timestamp_obj.timestamp() if timestamp_obj else None
            
            analyzed_posts.append({
                "likes": likes,
                "comments": comments,
                "views": views,
                "er": er,
                "isAd": p.get('isAd', False),
                "hashtags": p.get('hashtags', []),
                "mentions": p.get('mentions', []),
                "ts": timestamp_unix
            })
        
        ers = [p['er'] for p in analyzed_posts if p['er'] > 0]
        ers_org = [p['er'] for p in analyzed_posts if not p['isAd'] and p['er'] > 0]
        ers_spon = [p['er'] for p in analyzed_posts if p['isAd'] and p['er'] > 0]
        
        stdev_er = statistics.stdev(ers) if len(ers) > 1 else 0
        consistency_score = max(0, min(85, 100 - (stdev_er * 10)))
        
        timestamps = sorted([p['ts'] for p in analyzed_posts if p['ts']])
        avg_days_between = None
        if len(timestamps) > 1:
            diffs = [(timestamps[i] - timestamps[i-1]) / (24 * 3600) for i in range(1, len(timestamps))]
            avg_days_between = statistics.mean(diffs)
        
        total_likes = sum(p['likes'] for p in analyzed_posts)
        total_comments = sum(p['comments'] for p in analyzed_posts)
        total_views = sum(p['views'] for p in analyzed_posts)
        
        hashtag_counts = {}
        for p in analyzed_posts:
            for h in p.get('hashtags', []):
                if isinstance(h, str):
                    hashtag_counts[h] = hashtag_counts.get(h, 0) + 1
        
        metrics = {
            "username": posts[0].get('username', ''),
            "postsAnalyzed": len(posts),
            "avgEngagement_all": statistics.mean(ers) if ers else 0,
            "avgEngagement_organic": statistics.mean(ers_org) if ers_org else 0,
            "avgEngagement_sponsored": statistics.mean(ers_spon) if ers_spon else 0,
            "avgLikes": statistics.mean([p['likes'] for p in analyzed_posts]) if analyzed_posts else 0,
            "avgComments": statistics.mean([p['comments'] for p in analyzed_posts]) if analyzed_posts else 0,
            "avgViews": statistics.mean([p['views'] for p in analyzed_posts]) if analyzed_posts else 0,
            "stdevEngagement": stdev_er,
            "consistencyScore": consistency_score,
            "organicPct": (len(ers_org) / len(posts)) * 100 if posts else 0,
            "postingAvgDays": avg_days_between,
            "commentShare": total_comments / (total_likes + total_comments) if (total_likes + total_comments) > 0 else 0,
            "commentsPer10kViews": (total_comments / total_views) * 10000 if total_views > 0 else 0,
            "adDisclosurePct": 100,
            "hashtagCounts": hashtag_counts,
        }
        
        print(f"Aggregated metrics for {len(posts)} posts successfully.")
        return metrics
        
    except Exception as e:
        print(f"Error aggregating post metrics: {e}")
        return {}

def calculate_scores_manually(metrics, username):
    """Calculate scores manually when AI fails."""
    try:
        consistency_score = metrics.get('consistencyScore', 75)
        comment_share = metrics.get('commentShare', 0)
        comments_per_10k_views = metrics.get('commentsPer10kViews', 0)
        hashtag_counts = metrics.get('hashtagCounts', {})
        
        authenticity = max(0, min(100, consistency_score))
        brand_safety = 85
        audience_match = 75 if hashtag_counts else 70
        content_quality = max(0, min(100, 
            60 + (comment_share * 30) + min(20, comments_per_10k_views / 5)
        ))
        
        scores = {
            "username": username,
            "postsAnalyzed": metrics.get('postsAnalyzed', 0),
            "avgER_organic": round(metrics.get('avgEngagement_organic', 0), 6),
            "avgER_sponsored": round(metrics.get('avgEngagement_sponsored', 0), 6),
            "avgLikes": round(metrics.get('avgLikes', 0), 0),
            "avg_sponsored_likes": 0,
            "avgComments": round(metrics.get('avgComments', 0), 0),
            "avg_sponsored_comments": 0,
            "avgViews": round(metrics.get('avgViews', 0), 0),
            "Authenticity": round(authenticity, 1),
            "BrandSafety": round(brand_safety, 1),
            "AudienceMatch": round(audience_match, 1),
            "ContentQuality": round(content_quality, 1),
        }
        
        print(f"Manual score calculation completed:")
        print(f"  - Authenticity: {scores['Authenticity']}")
        print(f"  - Brand Safety: {scores['BrandSafety']}")
        print(f"  - Audience Match: {scores['AudienceMatch']}")
        print(f"  - Content Quality: {scores['ContentQuality']}")
        
        return scores
        
    except Exception as e:
        print(f"Error in manual score calculation: {e}")
        return {
            "username": username,
            "postsAnalyzed": metrics.get('postsAnalyzed', 0),
            "avgER_organic": metrics.get('avgEngagement_organic', 0),
            "avgER_sponsored": metrics.get('avgEngagement_sponsored', 0),
            "avgLikes": metrics.get('avgLikes', 0),
            "avg_sponsored_likes": 0,
            "avgComments": metrics.get('avgComments', 0),
            "avg_sponsored_comments": 0,
            "avgViews": metrics.get('avgViews', 0),
            "Authenticity": 75.0,
            "BrandSafety": 85.0,
            "AudienceMatch": 70.0,
            "ContentQuality": 65.0,
        }

def format_final_report(ai_scores, metrics):
    """Flattens the final AI response for database insertion."""
    try:
        return {
            "username": ai_scores.get("username", metrics.get("username", "")),
            "postsAnalyzed": ai_scores.get("postsAnalyzed", metrics.get("postsAnalyzed", 0)),
            "avgER_organic": ai_scores.get("avgER_organic", metrics.get("avgEngagement_organic", 0)),
            "avgER_sponsored": ai_scores.get("avgER_sponsored", metrics.get("avgEngagement_sponsored", 0)),
            "avgLikes": ai_scores.get("avgLikes", metrics.get("avgLikes", 0)),
            "avg_sponsored_likes": ai_scores.get("avg_sponsored_likes", 0),
            "avgComments": ai_scores.get("avgComments", metrics.get("avgComments", 0)),
            "avg_sponsored_comments": ai_scores.get("avg_sponsored_comments", 0),
            "avgViews": ai_scores.get("avgViews", metrics.get("avgViews", 0)),
            "Authenticity": ai_scores.get("Authenticity", 0),
            "BrandSafety": ai_scores.get("BrandSafety", 0),
            "AudienceMatch": ai_scores.get("AudienceMatch", 0),
            "ContentQuality": ai_scores.get("ContentQuality", 0),
        }
    except Exception as e:
        print(f"Error formatting final report: {e}")
        return {}
