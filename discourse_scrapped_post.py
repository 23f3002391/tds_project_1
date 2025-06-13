import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# 🔹 Define date range for filtering
start_date = datetime.fromisoformat('2025-01-01T00:00:00.000Z'.replace('Z', '+00:00'))
end_date = datetime.fromisoformat('2025-04-14T00:00:00.000Z'.replace('Z', '+00:00'))

# 🔹 Load filtered.json
with open("filtered_topics.json", "r", encoding="utf-8") as f:
    filtered_topics = json.load(f)

# 🔹 Base URL and output
base_url = "https://discourse.onlinedegree.iitm.ac.in"
output_json_path = "all_conversations.json"
conversations = []

# 🔹 Optional cookies
cookies = {
    "_forum_session":"YO5B%2FSowO4Q9dAWeVFWGJE9K104ys%2FXhgo4fZidFe7T1sCXoIYDlUf5%2B7aTl6h48AE%2FpyZSqBajFmpdAtwfCupCE%2Fo7KXuKucE0UDpMbF2yA9VdmAwIAQpNTJaPBVtRuHmxclDK4CX13D2QtFBFFA8q0x%2B1Xuin0IhZJOzrRW0LPCGGCr0t86V7W3zm%2B3rcaK2NeKVlN0yS2lw4RQ7kcVDs2NNjwvalrpzPNYWwXrNmfOPV05KsheIzK9fHMNAylo4WQkNlTsp4qv%2FLeUFVEtTkLOURDcg%3D%3D--k%2Boj6QXvAJLOPKM9--DEAbCMZrsjdQ3V5kIomINA%3D%3D",
    '_t':"x48m2ERw7SfOuUHnAYTROYEX4ISMuCn3%2Br3%2BoEgrb2NADh4A9CoWQEI8PZS3K3U1VnK3w6I1yVJUa8MBoQXR743KceW6NUZqxW4j44zyXyufoUg4I9nx5xHC02b%2FE3i7ll1uXUM%2B2EQIFMtwiRrUYf7po2KzIEzptyiAI3sgjzAeo1OpVXjpzuUM1tdmuNfSvqkSPA8zS82nm50KnM47WzxngLnACO5nvT9lp8RRuz20IHFocb2NKbskMqzaTqqr61qWm7dHkSmYDElOWYLYeCdCuZpFLJwCIa7faqm1kTAPEHErYwiniCDqCCgZ0HBq--vP5cIEpIVr7Aq5tE--VwdkA5tWSYltMMv%2BqeCDZA%3D%3D"
}  # Add your session cookies here if needed

# 🔹 Scraping Function
def scrape_conversation(topic_slug, topic_id):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    all_posts = []
    page = 1
    while True:
        url = f"{base_url}/t/{topic_slug}/{topic_id}.json?page={page}"
        response = requests.get(url, headers=headers, cookies=cookies)
        if response.status_code != 200:
            print(f"❌ Failed to fetch page {page} for topic {topic_id}. Status: {response.status_code}")
            break

        data = response.json()
        posts = data.get("post_stream", {}).get("posts", [])
        if not posts:
            break

        all_posts.extend(posts)
        page += 1

    if not all_posts:
        return None

    # Filter by first post's date
    first_post_date = datetime.fromisoformat(all_posts[0]['created_at'].replace('Z', '+00:00'))
    if not (start_date <= first_post_date <= end_date):
        print(f"⏭ Skipped '{topic_slug}' (date out of range)")
        return None

    # Process post contents
    processed_posts = []
    for post in all_posts:
        soup = BeautifulSoup(post.get("cooked", ""), "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        if paragraphs:
            post_text = "\n".join(paragraphs)
            processed_posts.append(post_text)

    return {
        "url": f"{base_url}/t/{topic_slug}/{topic_id}",
        "posts": processed_posts
    }

# 🔹 Loop through topics and collect conversations
for topic_id, topic_slug in filtered_topics.items():
    print(f"🔍 Processing topic: {topic_slug} ({topic_id})")
    convo = scrape_conversation(topic_slug, topic_id)
    if convo:
        conversations.append(convo)

# 🔹 Save to JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(conversations, f, indent=2, ensure_ascii=False)

print(f"\n✅ All conversations saved to: {output_json_path}")