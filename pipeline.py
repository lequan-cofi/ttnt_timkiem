import feedparser
import pandas as pd
import numpy as np
import json
import time
import warnings
import sys
import os # Thêm thư viện os
from dotenv import load_dotenv
import html
from sklearn.metrics import silhouette_score
import numpy as np
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
NUM_CLUSTERS = None  # Sẽ được cập nhật sau khi tìm K tối ưu
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
    except:
        return "N/A"

def normalize_title(title):
    return html.unescape(title).strip().lower()

def fetch_recent_articles(rss_urls, hours=24):
    """
    Hàm này lấy các bài viết mới từ danh sách các URL RSS trong 24h,
    lọc bỏ các bài có tiêu đề hoặc link trùng lặp.

    Args:
        rss_urls (list): Danh sách các URL của RSS feed.
        hours (int): Số giờ tính từ hiện tại để lấy bài viết.
        
    Returns:
        pd.DataFrame: Một DataFrame chứa thông tin các bài viết duy nhất đã thu thập được.
    """
    print(f"\n1. Bắt đầu lấy các bài viết trong vòng {hours} giờ qua...")
    articles = []
    
    # Khởi tạo set để lưu các tiêu đề và link đã gặp
    seen_titles = set()
    seen_links = set()
    
    # Đặt mốc thời gian giới hạn (chỉ lấy bài mới hơn mốc này)
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    # Lặp qua từng URL trong danh sách
    failed_rss = []  # Danh sách RSS lỗi
    successful_rss = []  # Danh sách RSS thành công
    
    for url in rss_urls:
        try:
            print(f"  - Đang xử lý: {url}")
            feed = feedparser.parse(url)
            
            # Kiểm tra feed có hợp lệ không
            if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                failed_rss.append(f"{url} - Không có bài viết hoặc RSS không hợp lệ")
                print(f"    ❌ Lỗi: Không có bài viết")
                continue
                
            entry_count = 0
            for entry in feed.entries:
                norm_title = normalize_title(entry.title)
                link = entry.link.strip()
                # Kiểm tra trùng lặp cả tiêu đề và link
                if norm_title in seen_titles or link in seen_links:
                    continue
                published_time = entry.get("published", "")
                summary_raw = entry.get("summary", "")
                image_url = None
                # Trích xuất URL hình ảnh từ trong thẻ <img> của tóm tắt
                if summary_raw:
                    soup = BeautifulSoup(summary_raw, 'html.parser')
                    img_tag = soup.find('img')
                    if img_tag and 'src' in img_tag.attrs:
                        image_url = img_tag['src']
                # Lấy tên nguồn từ link bài viết
                source_name = get_source_name(entry.link)
                # Xử lý và kiểm tra thời gian đăng bài
                if published_time:
                    try:
                        # Giữ nguyên thời gian từ RSS (đã ở múi giờ Việt Nam)
                        parsed_time = parse_date(published_time)
                        
                        # Chuyển về UTC chỉ để so sánh với time_threshold
                        time_for_comparison = parsed_time.astimezone(timezone.utc)
                        
                        if time_for_comparison >= time_threshold:
                            articles.append({
                                "title": entry.title,
                                "link": entry.link,
                                "summary_raw": summary_raw,
                                "published_time": parsed_time.isoformat(),  # Lưu thời gian gốc
                                "image_url": image_url,
                                "source": source_name
                            })
                            seen_titles.add(norm_title)
                            seen_links.add(link)
                            entry_count += 1
                    except (ValueError, TypeError):
                        continue
            
            # Thêm vào danh sách thành công
            successful_rss.append(f"{url} - {entry_count} bài viết")
            print(f"    ✅ Thành công: {entry_count} bài viết")
            
        except Exception as e:
            failed_rss.append(f"{url} - Lỗi: {str(e)}")
            print(f"    ❌ Lỗi: {str(e)}")
    
    # Hiển thị tổng kết
    print(f"\n📊 TỔNG KẾT RSS:")
    print(f"✅ Thành công: {len(successful_rss)} RSS")
    print(f"❌ Thất bại: {len(failed_rss)} RSS")
    
    if failed_rss:
        print(f"\n❌ DANH SÁCH RSS THẤT BẠI:")
        for failed in failed_rss:
            print(f"  - {failed}")
    
    if successful_rss:
        print(f"\n✅ DANH SÁCH RSS THÀNH CÔNG:")
        for success in successful_rss:
            print(f"  - {success}")
    
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
        prompt = f"""Bạn là một trợ lý biên tập báo chí. Dựa vào các thông tin dưới đây, hãy tạo ra chỉ một tên chủ đề ngắn gọn duy nhất (không quá 6 từ) bằng tiếng Việt để tóm tắt nội dung chính.
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
    # Sử dụng số cụm thực tế từ dữ liệu
    actual_clusters = df['topic_cluster'].nunique()
    for i in range(actual_clusters):
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
        time.sleep(4.1)
    print("-> Gán nhãn chủ đề hoàn tất.")
    return topic_labels

def find_optimal_clusters(embeddings, max_clusters=20):
    """Tìm số cụm (K) tối ưu bằng hệ số Silhouette.
    
    Args:
        embeddings (numpy.ndarray): Ma trận embeddings của các bài viết
        max_clusters (int): Số cụm tối đa cần xem xét (mặc định là 20)
        
    Returns:
        int: Số cụm tối ưu (K) có hệ số Silhouette cao nhất
    """
    print("\n3. Đang tìm số cụm (K) tối ưu bằng hệ số Silhouette...")
    
    silhouette_scores = []
    possible_k_values = range(2, max_clusters + 1)

    for k in possible_k_values:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)

    # Tự động tìm K có silhouette score cao nhất
    best_k_index = np.argmax(silhouette_scores)
    optimal_k = possible_k_values[best_k_index]

    print(f"-> Đã tìm thấy số cụm tối ưu là K={optimal_k}")
    return optimal_k

def main_pipeline():
    """Hàm chính chạy toàn bộ quy trình."""
    print("\n🚀 BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG HÓA")
    
    # Các bước giữ nguyên
    df = fetch_recent_articles(RSS_URLS,hours=24)
    if df.empty:
        print("Không có bài viết mới nào. Dừng quy trình.")
        return
    df = clean_text(df)
    embeddings = vectorize_text(df['summary_not_stop_word'].tolist(), SBERT_MODEL)
    
    # Thay thế đoạn code cũ bằng lời gọi hàm mới
    NUM_CLUSTERS = find_optimal_clusters(embeddings)
    # -----------------------------------------------------------------------------


    # BƯỚC 4: Chạy K-Means cuối cùng với số cụm đã tìm được
    # -----------------------------------------------------------------------------
    print(f"\n4. Đang thực hiện phân cụm K-Means với {NUM_CLUSTERS} cụm...")
    # Khởi tạo mô hình K-Means với NUM_CLUSTERS đã được tự động xác định
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)

    # Huấn luyện mô hình và dự đoán cụm
    df['topic_cluster'] = kmeans.fit_predict(embeddings)
    print("-> Phân cụm hoàn tất.")



    print("\n📊 Phân bổ số lượng bài viết vào các cụm (chủ đề):")
    cluster_counts = df['topic_cluster'].value_counts().sort_index()
    print(cluster_counts)
        
    topic_labels = get_topic_labels(df)
        
    print("5/6: Đang tính toán ma trận tương đồng...")
    cosine_sim = cosine_similarity(embeddings)
    
    print("6/6: Đang lưu các file kết quả...")
    df.to_csv('final_articles_for_app.csv', index=False, encoding='utf-8-sig')
    np.save('cosine_similarity_matrix.npy', cosine_sim)
    with open('topic_labels.json', 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f, ensure_ascii=False, indent=4)
    # Lưu embeddings.npy cho app
    np.save('embeddings.npy', embeddings)
    print("✅ Đã lưu embeddings.npy.")
    print("\n✅ QUY TRÌNH HOÀN TẤT! ✅")

if __name__ == "__main__":
    main_pipeline()
