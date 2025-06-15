import feedparser
import pandas as pd
import numpy as np
import json
import time
import warnings
import sys
import os # Thêm thư viện os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
# Xóa API key cũ từ biến môi trường nếu có
if 'GOOGLE_API_KEY' in os.environ:
    del os.environ['GOOGLE_API_KEY']

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
from pyvi import ViTokenizer
from stopwordsiso import stopwords
# --- CẤU HÌNH -----------------------------------------------------------------
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Debug print to verify API key and environment
print(f"Current working directory: {os.getcwd()}")
print(f"Environment file path: {os.path.join(os.getcwd(), '.env')}")
print(f"Environment variables: {dict(os.environ)}")

# >>> SỬA LỖI UNICODE TRÊN WINDOWS CONSOLE <<<
# Buộc output của chương trình phải là UTF-8 để hiển thị tiếng Việt chính xác
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except TypeError: # Bỏ qua nếu chạy trên môi trường không hỗ trợ
    pass


# Các nguồn RSS để thu thập dữ liệu
RSS_URLS = [
    "https://dantri.com.vn/rss/home.rss",
    "https://dantri.com.vn/rss/xa-hoi.rss",
    "https://dantri.com.vn/rss/gia-vang.rss",
    "https://dantri.com.vn/rss/the-thao.rss",
    "https://dantri.com.vn/rss/giao-duc.rss",
    "https://dantri.com.vn/rss/kinh-doanh.rss",
    "https://dantri.com.vn/rss/giai-tri.rss",
    "https://dantri.com.vn/rss/phap-luat.rss",
    "https://dantri.com.vn/rss/cong-nghe.rss",
    "https://dantri.com.vn/rss/tinh-yeu-gioi-tinh.rss",
    "https://dantri.com.vn/rss/noi-vu.rss",
    "https://dantri.com.vn/rss/tam-diem.rss",
    "https://dantri.com.vn/rss/infographic.rss",
    "https://dantri.com.vn/rss/dnews.rss",
    "https://dantri.com.vn/rss/xo-so.rss",
    "https://dantri.com.vn/rss/tet-2025.rss",
    "https://dantri.com.vn/rss/d-buzz.rss",
    "https://dantri.com.vn/rss/su-kien.rss",
    "https://dantri.com.vn/rss/the-gioi.rss",
    "https://dantri.com.vn/rss/doi-song.rss",
    "https://dantri.com.vn/rss/lao-dong-viec-lam.rss",
    "https://dantri.com.vn/rss/tam-long-nhan-ai.rss",
    "https://dantri.com.vn/rss/bat-dong-san.rss",
    "https://dantri.com.vn/rss/du-lich.rss",
    "https://dantri.com.vn/rss/suc-khoe.rss",
    "https://dantri.com.vn/rss/o-to-xe-may.rss",
    "https://dantri.com.vn/rss/khoa-hoc.rss",
    "https://dantri.com.vn/rss/ban-doc.rss",
    "https://dantri.com.vn/rss/dmagazine.rss",
    "https://dantri.com.vn/rss/photo-news.rss",
    "https://dantri.com.vn/rss/toa-dam-truc-tuyen.rss",
    "https://dantri.com.vn/rss/interactive.rss",
    "https://dantri.com.vn/rss/photo-story.rss",
    "https://vnexpress.net/rss/tin-moi-nhat.rss",      # Trang chủ (thường là tin mới nhất)
    "https://vnexpress.net/rss/the-gioi.rss",
    "https://vnexpress.net/rss/thoi-su.rss",
    "https://vnexpress.net/rss/kinh-doanh.rss",
    "https://vnexpress.net/rss/startup.rss",
    "https://vnexpress.net/rss/giai-tri.rss",
    "https://vnexpress.net/rss/the-thao.rss",
    "https://vnexpress.net/rss/phap-luat.rss",
    "https://vnexpress.net/rss/giao-duc.rss",
    "https://vnexpress.net/rss/tin-noi-bat.rss",
    
    # Cột bên phải
    "https://vnexpress.net/rss/suc-khoe.rss",
    "https://vnexpress.net/rss/doi-song.rss",
    "https://vnexpress.net/rss/du-lich.rss",
    "https://vnexpress.net/rss/khoa-hoc.rss",         # Khoa học công nghệ
    "https://vnexpress.net/rss/xe.rss",
    "https://vnexpress.net/rss/y-kien.rss",
    "https://vnexpress.net/rss/tam-su.rss",
    "https://vnexpress.net/rss/cuoi.rss",
    "https://vnexpress.net/rss/tin-xem-nhieu.rss",
    "https://thanhnien.vn/rss/home.rss",                   # Trang chủ
    "https://thanhnien.vn/rss/thoi-su.rss",
    "https://thanhnien.vn/rss/chinh-tri.rss",
    "https://thanhnien.vn/rss/chao-ngay-moi.rss",
    "https://thanhnien.vn/rss/the-gioi.rss",
    "https://thanhnien.vn/rss/kinh-te.rss",
    "https://thanhnien.vn/rss/doi-song.rss",
    "https://thanhnien.vn/rss/suc-khoe.rss",
    "https://thanhnien.vn/rss/gioi-tre.rss",
    "https://thanhnien.vn/rss/tieu-dung-thong-minh.rss",
    "https://thanhnien.vn/rss/giao-duc.rss",
    "https://thanhnien.vn/rss/du-lich.rss",
    "https://thanhnien.vn/rss/van-hoa.rss",
    "https://thanhnien.vn/rss/giai-tri.rss",
    "https://thanhnien.vn/rss/the-thao.rss",
    "https://thanhnien.vn/rss/cong-nghe.rss",
    "https://thanhnien.vn/rss/xe.rss",
    "https://thanhnien.vn/rss/thoi-trang-tre.rss",
    "https://thanhnien.vn/rss/ban-doc.rss",
    "https://thanhnien.vn/rss/rao-vat.rss",
    "https://thanhnien.vn/rss/video.rss",
    "https://thanhnien.vn/rss/dien-dan.rss",
    "https://thanhnien.vn/rss/podcast.rss",
    "https://thanhnien.vn/rss/nhat-ky-tet-viet.rss",
    "https://thanhnien.vn/rss/magazine.rss",
    "https://thanhnien.vn/rss/cung-con-di-tiep-cuoc-doi.rss",
    "https://thanhnien.vn/rss/ban-can-biet.rss",
    "https://thanhnien.vn/rss/cai-chinh.rss",
    "https://thanhnien.vn/rss/blog-phong-vien.rss",
    "https://thanhnien.vn/rss/toi-viet.rss",
    "https://thanhnien.vn/rss/viec-lam.rss",
    "https://thanhnien.vn/rss/tno.rss",
    "https://thanhnien.vn/rss/tin-24h.rss",
    "https://thanhnien.vn/rss/thi-truong.rss",
    "https://thanhnien.vn/rss/tin-nhanh-360.tno",
     # Cột bên trái
    "https://tuoitre.vn/rss/tin-moi-nhat.rss",  # Trang chủ
    "https://tuoitre.vn/rss/the-gioi.rss",
    "https://tuoitre.vn/rss/kinh-doanh.rss",
    "https://tuoitre.vn/rss/xe.rss",
    "https://tuoitre.vn/rss/van-hoa.rss",
    "https://tuoitre.vn/rss/the-thao.rss",
    "https://tuoitre.vn/rss/khoa-hoc.rss",
    "https://tuoitre.vn/rss/gia-that.rss",
    "https://tuoitre.vn/rss/ban-doc.rss",
    "https://tuoitre.vn/rss/video.rss",

    # Cột bên phải
    "https://tuoitre.vn/rss/thoi-su.rss",
    "https://tuoitre.vn/rss/phap-luat.rss",
    "https://tuoitre.vn/rss/cong-nghe.rss",
    "https://tuoitre.vn/rss/nhip-song-tre.rss",
    "https://tuoitre.vn/rss/giai-tri.rss",
    "https://tuoitre.vn/rss/giao-duc.rss",
    "https://tuoitre.vn/rss/suc-khoe.rss",
    "https://tuoitre.vn/rss/thu-gian.rss",
    "https://tuoitre.vn/rss/du-lich.rss"
]

# Cấu hình cho các mô hìnhgit 
NUM_CLUSTERS = 12
SBERT_MODEL = 'Cloyne/vietnamese-sbert-v3'

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
    except Exception:
        # Bắt lỗi cụ thể hơn là một except rỗng
        return "N/A"

def _parse_feed(url):
    """
    Hàm phụ trợ: Tải và phân tích một URL RSS duy nhất.
    Hàm này được thiết kế để chạy trong một luồng riêng.
    """
    try:
        print(f"   - Bắt đầu tải: {url}")
        feed = feedparser.parse(url)
        print(f"   - Hoàn tất tải: {url}")
        return feed
    except Exception as e:
        print(f"   - LỖI khi tải {url}: {e}")
        return None

def fetch_recent_articles_optimized(rss_urls, hours=24):
    """
    Phiên bản tối ưu: Lấy bài viết từ nhiều RSS feed đồng thời,
    lọc bỏ các bài có tiêu đề trùng lặp.
    """
    print(f"\n1. Bắt đầu lấy các bài viết trong vòng {hours} giờ qua (chế độ tối ưu)...")
    articles = []
    seen_titles = set()
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    # --- TỐI ƯU HÓA: Sử dụng ThreadPoolExecutor để tải nhiều feed cùng lúc ---
    parsed_feeds = []
    with ThreadPoolExecutor(max_workers=10) as executor: # max_workers: số luồng chạy song song
        # Tạo các tác vụ tải feed
        future_to_url = {executor.submit(_parse_feed, url): url for url in rss_urls}
        
        # Lấy kết quả khi các tác vụ hoàn thành
        for future in as_completed(future_to_url):
            feed = future.result()
            if feed: # Chỉ xử lý nếu feed được tải thành công
                parsed_feeds.append(feed)

    print("\n2. Đã tải xong các feed, bắt đầu xử lý và lọc bài viết...")
    # Lặp qua các feed đã được tải về thành công
    for feed in parsed_feeds:
        for entry in feed.entries:
            if entry.title in seen_titles:
                continue

            published_time_str = entry.get("published", "")
            if not published_time_str:
                continue

            try:
                parsed_time = parse_date(published_time_str).astimezone(timezone.utc)
                if parsed_time < time_threshold:
                    continue
                    
                # Thêm bài viết vào danh sách và đánh dấu tiêu đề đã gặp
                summary_raw = entry.get("summary", "")
                image_url = None
                if summary_raw:
                    soup = BeautifulSoup(summary_raw, 'html.parser')
                    img_tag = soup.find('img')
                    if img_tag and 'src' in img_tag.attrs:
                        image_url = img_tag['src']
                
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary_raw": summary_raw,
                    "published_time": parsed_time.isoformat(),
                    "image_url": image_url,
                    "source": get_source_name(entry.link)
                })
                seen_titles.add(entry.title)

            except (ValueError, TypeError):
                continue
                
    print(f"\n-> Đã tìm thấy {len(articles)} bài viết mới (sau khi lọc trùng lặp).")
    return pd.DataFrame(articles)


def clean_text(df):
    """
    Làm sạch và xử lý văn bản từ cột 'summary_raw':
    - Chuyển chữ thường, bỏ HTML, dấu câu.
    - Bỏ bài ngắn (<10 từ).
    - Tách từ tiếng Việt bằng pyvi.
    - Bỏ stopword.
    """
    print("\n2. Đang làm sạch văn bản...")
    

    # B1: Làm sạch văn bản cơ bản
    summary = (df['summary_raw']
               .str.lower()
               .str.replace(r'<.*?>', '', regex=True)
               .str.replace(r'[^\w\s]', '', regex=True))
    df['summary_cleaned'] = summary

    # B2: Bỏ NaN, dòng rỗng
    df.dropna(subset=['summary_cleaned'], inplace=True)
    df = df[df['summary_cleaned'].str.strip() != '']

    # B3: Lọc bài có < 10 từ
    original_count = len(df)
    df = df[df['summary_cleaned']
                      .str.split()
                      .str.len() >= 10]
    print(f"-> Đã lọc {original_count - len(df)} bài quá ngắn.")

    # B4: Tokenize + bỏ stopword
    vi_stop = stopwords("vi")
    def remove_stop_pyvi(text):
        tokens = ViTokenizer.tokenize(text).split()
        filtered = [t for t in tokens if t not in vi_stop]
        return " ".join(filtered)

    df['summary_not_stop_word'] = df['summary_cleaned'].apply(remove_stop_pyvi)

    df = df.reset_index(drop=True)
    print("-> Hoàn tất làm sạch.")
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
    df = fetch_recent_articles_optimized(RSS_URLS,hours=24)
    if df.empty:
        print("Không có bài viết mới nào. Dừng quy trình.")
        return
    df = clean_text(df)
    embeddings = vectorize_text(df['summary_not_stop_word'].tolist(), SBERT_MODEL)
    
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
