# generate_embeddings.py
import json
import numpy as np
import faiss

from fast_semantic_matcher import fast_semantic_matcher  # fixed import
from knowledge_manager import knowledge_manager

def precompute_all_embeddings():
    """Generate embeddings and FAISS index for all influencers in the knowledge base."""
    print("🔄 Starting batch embedding generation...")

    # Refresh and load influencer knowledge
    knowledge_manager.refresh_knowledge()
    influencers = knowledge_manager.knowledge_base or []
    print(f"📊 Processing {len(influencers)} influencers...")

    if not influencers:
        print("❌ No influencers found. Ensure your JSON database has profiles.")
        return

    # Build text to embed per influencer
    texts = []
    usernames = []
    embeddings_data = {}

    for i, inf in enumerate(influencers):
        username = inf.get('username', f'unknown_{i}')
        name = inf.get('name', '') or ''
        category = inf.get('category', '') or ''
        bio = inf.get('bio', '') or ''
        tags = ' '.join(inf.get('top_hashtags', [])[:10])
        text = f"{name} {category} {bio} {tags}".strip()

        usernames.append(username)
        texts.append(text)

        # Store influencer summary for later lookup
        embeddings_data[username] = {
            'influencer_data': {
                'username': username,
                'name': name,
                'category': category,
                'bio': bio,
                'top_hashtags': inf.get('top_hashtags', [])[:15],
                'followers': inf.get('followers', 0),
                'engagement_rate': inf.get('engagement_rate', 0),
                'verified': inf.get('verified', False),
                'brand_collaborations': inf.get('brand_collaborations', []),
                'content_quality': inf.get('content_quality', 0),
                'authenticity': inf.get('authenticity', 0),
                'brand_safety': inf.get('brand_safety', 0),
                'audience_match': inf.get('audience_match', 0),
            }
        }

        if (i + 1) % 100 == 0:
            print(f"✅ Prepared {i+1}/{len(influencers)}")

    # Encode in batches
    print("🧠 Encoding texts to embeddings...")
    vectors = fast_semantic_matcher.get_batch_embeddings(texts)  # normalized float32 [1][4]
    print(f"✅ Got embeddings: shape={vectors.shape}")

    # Save JSON summaries
    print("💾 Saving influencer_embeddings.json...")
    with open('influencer_embeddings.json', 'w') as f:
        json.dump(embeddings_data, f, indent=2)

    # Build FAISS index (cosine using IP on normalized vectors)
    print("🏗️ Building FAISS index...")
    fast_semantic_matcher.build_faiss_index(vectors, usernames)
    fast_semantic_matcher.save_faiss_index('influencer_index.faiss', 'username_mapping.json')

    print(f"✅ Batch complete! Indexed {len(usernames)} influencers.")

if __name__ == "__main__":
    precompute_all_embeddings()
