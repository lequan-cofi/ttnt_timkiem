import feedparser
import pandas as pd
import numpy as np
import json
import time
import warnings
import sys
import os # Thêm thư viện os
from dotenv import load_dotenv

# --- SỬA LỖI SUBPROCESS/JOBLIB TRÊN WINDOWS ---
# Đặt một giá trị cố định cho số lõi CPU để tránh lỗi khi joblib/loky
# cố gắng tự động đếm lõi trong một số môi trường phức tạp.
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta, timezone
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from urllib.parse import urlparse

# --- CẤU HÌNH -----------------------------------------------------------------
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# >>> SỬA LỖI UNICODE TRÊN WINDOWS CONSOLE <<<
# Buộc output của chương trình phải là UTF-8 để hiển thị tiếng Việt chính xác
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except TypeError: # Bỏ qua nếu chạy trên môi trường không hỗ trợ
    pass


# Các nguồn RSS để thu thập dữ liệu
RSS_URLS = [
    "https://thanhnien.vn/rss/home.rss",
    "https://tuoitre.vn/rss/tin-moi-nhat.rss",
    "https://vnexpress.net/rss/tin-moi-nhat.rss",
]

# Cấu hình cho các mô hình
NUM_CLUSTERS = 12
SBERT_MODEL = 'vinai/phobert-base-v2'

# --- KIỂM TRA VÀ CẤU HÌNH API KEY (PHẦN GỠ LỖI) ----------------------------
print("--- KIỂM TRA API KEY ---")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ LỖI: Không tìm thấy API Key trong biến môi trường.")
    print("Vui lòng tạo file .env và thêm dòng sau:")
    print("GOOGLE_API_KEY=your_api_key_here")
    exit()  # Dừng chương trình ngay lập tức
else:
    # In ra một phần của key để xác nhận
    print(f"✅ Đã tìm thấy API Key. Bắt đầu bằng: '{api_key[:4]}...'. Kết thúc bằng: '...{api_key[-4:]}'.")
    print("Đang cấu hình với Google...")
    try:
        genai.configure(api_key=api_key)
        print("✅ Cấu hình Google API thành công.")
    except Exception as e:
        print(f"❌ LỖI KHI CẤU HÌNH: {e}")
        # Không thoát, để hàm generate_meaningful_topic_name xử lý lỗi và tiếp tục
        pass


# --- CÁC HÀM CHỨC NĂNG (Giữ nguyên) ----------------------------------------

def get_source_name(link):
    """Trích xuất tên miền chính từ URL để làm tên nguồn."""
    try:
        domain = urlparse(link).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.split('.')[0].capitalize()
    except:
        return "N/A"

def fetch_recent_articles(hours=24):
    """Lấy các bài viết mới từ RSS và trích xuất URL hình ảnh, tên nguồn."""
    print(f"\n1/6: Bắt đầu lấy các bài viết trong vòng {hours} giờ qua...")
    articles = []
    # Sử dụng múi giờ Việt Nam (UTC+7)
    vn_timezone = timezone(timedelta(hours=7))
    time_threshold = datetime.now(vn_timezone) - timedelta(hours=hours)
    for url in RSS_URLS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            published_time = entry.get("published", "")
            image_url = None
            summary_raw = entry.get("summary", "")
            if summary_raw:
                soup = BeautifulSoup(summary_raw, 'html.parser')
                img_tag = soup.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    image_url = img_tag['src']
            
            # Trích xuất tên nguồn từ link bài viết
            source_name = get_source_name(entry.link)

            if published_time:
                try:
                    # Parse thời gian và chuyển sang múi giờ Việt Nam
                    parsed_time = parse_date(published_time).astimezone(vn_timezone)
                    if parsed_time >= time_threshold:
                        articles.append({
                            "title": entry.title, "link": entry.link,
                            "summary_raw": summary_raw, "published_time": parsed_time.isoformat(),
                            "image_url": image_url,
                            "source": source_name
                        })
                except (ValueError, TypeError):
                    continue
    print(f"-> Đã tìm thấy {len(articles)} bài viết mới.")
    return pd.DataFrame(articles)

def clean_text(df):
    """Làm sạch văn bản tóm tắt."""
    print("2/6: Đang làm sạch văn bản...")
    df['summary_cleaned'] = df['summary_raw'].str.lower().str.replace(r'<.*?>', '', regex=True)
    df['summary_cleaned'] = df['summary_cleaned'].str.replace(r'[^\w\s]', '', regex=True)
    df.dropna(subset=['summary_cleaned'], inplace=True)
    df = df[df['summary_cleaned'].str.strip() != ''].reset_index(drop=True)
    print("-> Làm sạch văn bản hoàn tất.")
    return df

def vectorize_text(sentences, model_name):
    """Vector hóa câu bằng S-BERT."""
    print(f"3/6: Đang vector hóa văn bản bằng mô hình {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)
    print("-> Vector hóa hoàn tất.")
    return embeddings

def generate_meaningful_topic_name(keywords, sample_titles):
    """Sử dụng Gemini để tạo tên chủ đề."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""Bạn là một trợ lý biên tập báo chí. Dựa vào các thông tin dưới đây, hãy tạo ra một tên chủ đề ngắn gọn (không quá 6 từ) bằng tiếng Việt để tóm tắt nội dung chính.
        Các từ khóa chính của chủ đề: {keywords}
        Một vài tiêu đề bài viết ví dụ:
        - {"\n- ".join(sample_titles)}
        Tên chủ đề gợi ý:"""
        response = model.generate_content(prompt)
        return response.text.strip().replace("*", "")
    except Exception as e:
        # In ra lỗi cụ thể hơn
        print(f"  - Lỗi khi gọi Gemini API: {type(e).__name__} - {e}. Sử dụng từ khóa làm nhãn thay thế.")
        return keywords

def get_topic_labels(df, num_keywords=5):
    """Gán nhãn chủ đề cho các cụm."""
    print("4/6: Đang gán nhãn chủ đề cho các cụm...")
    topic_labels = {}
    for i in range(NUM_CLUSTERS):
        cluster_df = df[df['topic_cluster'] == i]
        cluster_texts = cluster_df['summary_cleaned'].tolist()
        if len(cluster_texts) < 3:
            topic_labels[str(i)] = "Chủ đề nhỏ (ít bài viết)"
            continue
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_tfidf_scores.argsort()[-num_keywords:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        keywords = ", ".join([feature_names[j] for j in top_indices])
        sample_titles = cluster_df['title'].head(3).tolist()
        meaningful_name = generate_meaningful_topic_name(keywords, sample_titles)
        print(f"  - Cluster {i}: {keywords}  =>  Tên chủ đề: {meaningful_name}")
        topic_labels[str(i)] = meaningful_name
        time.sleep(1)
    print("-> Gán nhãn chủ đề hoàn tất.")
    return topic_labels

def main_pipeline():
    """Hàm chính chạy toàn bộ quy trình."""
    print("\n🚀 BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG HÓA")
    
    # Các bước giữ nguyên
    df = fetch_recent_articles(hours=24)
    if df.empty:
        print("Không có bài viết mới nào. Dừng quy trình.")
        return
    df = clean_text(df)
    embeddings = vectorize_text(df['summary_cleaned'].tolist(), SBERT_MODEL)
    
    print("Đang thực hiện phân cụm...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df['topic_cluster'] = kmeans.fit_predict(embeddings)
    
    topic_labels = get_topic_labels(df)
    
    print("5/6: Đang tính toán ma trận tương đồng...")
    cosine_sim = cosine_similarity(embeddings)
    
    print("6/6: Đang lưu các file kết quả...")
    df.to_csv('final_articles_for_app.csv', index=False, encoding='utf-8-sig')
    np.save('cosine_similarity_matrix.npy', cosine_sim)
    with open('topic_labels.json', 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f, ensure_ascii=False, indent=4)
    
    print("\n✅ QUY TRÌNH HOÀN TẤT! ✅")

if __name__ == "__main__":
    main_pipeline()
