# ==== IMPORTS ====
# Khai báo các thư viện cần thiết cho pipeline xử lý dữ liệu.

import feedparser  # Thư viện chuyên dụng để đọc và phân tích (parse) các nguồn cấp dữ liệu RSS/Atom.
import pandas as pd  # Dùng để xử lý dữ liệu dạng bảng (DataFrame).
import numpy as np  # Dùng cho các phép toán số học, đặc biệt là với mảng và ma trận.
import json  # Dùng để làm việc với dữ liệu định dạng JSON (lưu file nhãn chủ đề).
import time  # Cung cấp các hàm liên quan đến thời gian (ví dụ: time.sleep để tránh quá tải API).
import warnings  # Dùng để quản lý các cảnh báo (ở đây là để bỏ qua chúng).
import sys  # Cung cấp quyền truy cập vào các biến và hàm của hệ thống Python.
import os  # Cung cấp các hàm để tương tác với hệ điều hành (quản lý file, biến môi trường).
from dotenv import load_dotenv # Dùng để tải các biến môi trường từ file .env, giúp quản lý API key an toàn.
import html  # Cung cấp các công cụ để làm việc với các thực thể HTML (ví dụ: &amp; -> &).
from sklearn.metrics import silhouette_score # Hàm tính điểm Silhouette để đo chất lượng của việc phân cụm.
import numpy as np

# Xóa API key cũ từ biến môi trường nếu có để đảm bảo key từ file .env được sử dụng.
if 'GOOGLE_API_KEY' in os.environ:
    del os.environ['GOOGLE_API_KEY']

# --- SỬA LỖI SUBPROCESS/JOBLIB TRÊN WINDOWS ---
# Đặt một giá trị cố định cho số lõi CPU để tránh lỗi khi joblib/loky
# cố gắng tự động đếm lõi trong một số môi trường phức tạp (thường gặp khi chạy đa tiến trình).
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from bs4 import BeautifulSoup  # Dùng để phân tích và trích xuất dữ liệu từ file HTML (ở đây là summary từ RSS).
from dateutil.parser import parse as parse_date  # Thư viện mạnh mẽ để chuyển đổi nhiều định dạng chuỗi thành đối tượng datetime.
from datetime import datetime, timedelta, timezone  # Các lớp để làm việc với ngày, giờ và múi giờ.
from sentence_transformers import SentenceTransformer  # Thư viện chứa mô hình SBERT để chuyển văn bản thành vector.
from sklearn.cluster import KMeans  # Thuật toán phân cụm K-Means.
from sklearn.feature_extraction.text import TfidfVectorizer  # Dùng để chuyển đổi văn bản thành ma trận TF-IDF, hữu ích để tìm từ khóa.
from sklearn.metrics.pairwise import cosine_similarity  # Hàm tính độ tương đồng Cosine giữa các vector.
import google.generativeai as genai  # SDK chính thức của Google để tương tác với các mô hình Gemini.
from urllib.parse import urlparse  # Công cụ để phân tích một URL thành các thành phần của nó (tên miền, đường dẫn...).
from pyvi import ViTokenizer  # Thư viện chuyên dụng để tách từ (tokenize) cho văn bản tiếng Việt.
from stopwordsiso import stopwords # Thư viện cung cấp danh sách các từ dừng (stopwords) cho nhiều ngôn ngữ.

# --- CẤU HÌNH -----------------------------------------------------------------
# Bỏ qua tất cả các cảnh báo để output được sạch sẽ.
warnings.filterwarnings('ignore')

# Tải các biến môi trường từ file .env vào os.environ.
# Điều này cho phép lấy API key mà không cần viết trực tiếp vào code.
load_dotenv()

# Các dòng debug để kiểm tra xem file .env và API key có được tải đúng cách không.
print(f"Thư mục làm việc hiện tại: {os.getcwd()}")
print(f"Đường dẫn file .env: {os.path.join(os.getcwd(), '.env')}")

# >>> SỬA LỖI UNICODE TRÊN WINDOWS CONSOLE <<<
# Buộc output của chương trình phải là UTF-8 để hiển thị tiếng Việt chính xác trên console của Windows.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except TypeError:  # Bỏ qua nếu chạy trên môi trường không hỗ trợ (ví dụ: Linux, macOS đã mặc định là UTF-8).
    pass


# Danh sách các nguồn cấp dữ liệu RSS sẽ được thu thập.
# Bao gồm nhiều chuyên mục từ các báo lớn như Dân Trí, VnExpress, Thanh Niên, Tuổi Trẻ.
RSS_URLS = [
    # ... (danh sách URL như trong code gốc) ...
]

# Cấu hình cho các mô hình
NUM_CLUSTERS = None  # Số cụm sẽ được xác định tự động bằng thuật toán, không còn là giá trị cố định.
SBERT_MODEL = 'Cloyne/vietnamese-sbert-v3'  # Tên mô hình S-BERT tiếng Việt sẽ được sử dụng.

# --- KIỂM TRA VÀ CẤU HÌNH API KEY (PHẦN GỠ LỖI) ----------------------------
print("\n--- KIỂM TRA API KEY ---")
# Lấy giá trị của biến môi trường 'GOOGLE_API_KEY'.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Nếu không tìm thấy key, in ra hướng dẫn và thoát chương trình.
    print("❌ LỖI: Không tìm thấy API Key trong biến môi trường.")
    print("Vui lòng tạo file .env và thêm dòng sau:")
    print("GOOGLE_API_KEY=your_api_key_here")
    exit()  # Dừng chương trình ngay lập tức vì không thể tiếp tục nếu thiếu key.
else:
    # Nếu tìm thấy, in ra một phần của key để người dùng xác nhận.
    print(f"✅ Đã tìm thấy API Key. Bắt đầu bằng: '{api_key[:4]}...'. Kết thúc bằng: '...{api_key[-4:]}'.")
    print("Đang cấu hình với Google...")
    try:
        # Cấu hình thư viện google.generativeai với API key đã lấy.
        genai.configure(api_key=api_key)
        print("✅ Cấu hình Google API thành công.")
    except Exception as e:
        # Bắt lỗi nếu có vấn đề trong quá trình cấu hình (ví dụ: key sai định dạng).
        print(f"❌ LỖI KHI CẤU HÌNH: {e}")
        pass # Không thoát, để các hàm sau tự xử lý lỗi và có thể dùng phương án dự phòng.


# --- CÁC HÀM CHỨC NĂNG ----------------------------------------------------

def get_source_name(link):
    """
    Chức năng: Trích xuất tên miền chính từ URL để làm tên nguồn báo.
    Kịch bản: Được gọi trong lúc xử lý mỗi bài viết từ RSS.
    Ví dụ: 'https://vnexpress.net/rss/tin-moi-nhat.rss' -> 'Vnexpress'
    """
    try:
        domain = urlparse(link).netloc  # Phân tích URL và lấy phần tên miền (netloc).
        if domain.startswith('www.'):  # Bỏ phần 'www.' nếu có.
            domain = domain[4:]
        return domain.split('.')[0].capitalize()  # Tách lấy phần đầu tiên và viết hoa chữ cái đầu.
    except:
        return "N/A"  # Trả về "N/A" nếu có lỗi.


def normalize_title(title):
    """
    Chức năng: Chuẩn hóa tiêu đề để dễ dàng so sánh và loại bỏ trùng lặp.
    Logic: Chuyển các ký tự HTML (&amp;) thành ký tự thường (&), xóa khoảng trắng thừa, và chuyển thành chữ thường.
    """
    return html.unescape(title).strip().lower()


def fetch_recent_articles(rss_urls, hours=24):
    """
    Chức năng: Lấy các bài viết mới từ danh sách RSS trong một khoảng thời gian nhất định.
    Kịch bản: Là bước đầu tiên của pipeline, thu thập dữ liệu thô.
    Logic:
    1. Lặp qua từng URL RSS.
    2. Phân tích RSS feed.
    3. Với mỗi bài viết, kiểm tra xem nó có đủ mới không.
    4. Kiểm tra xem tiêu đề hoặc link đã tồn tại chưa để tránh trùng lặp.
    5. Trích xuất thông tin cần thiết và thêm vào danh sách.
    Trả về: Một DataFrame chứa các bài viết đã thu thập.
    """
    print(f"\n1. Bắt đầu lấy các bài viết trong vòng {hours} giờ qua...")
    articles = []
    seen_titles = set()  # Dùng set để kiểm tra trùng lặp tiêu đề hiệu quả.
    seen_links = set()   # Dùng set để kiểm tra trùng lặp link hiệu quả.
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours) # Đặt mốc thời gian giới hạn.

    for url in rss_urls:
        feed = feedparser.parse(url) # Phân tích RSS feed.
        for entry in feed.entries:
            norm_title = normalize_title(entry.title) # Chuẩn hóa tiêu đề.
            link = entry.link.strip() # Lấy link và xóa khoảng trắng.
            
            # Nếu tiêu đề hoặc link đã tồn tại, bỏ qua bài viết này.
            if norm_title in seen_titles or link in seen_links:
                continue
            
            published_time = entry.get("published", "") # Lấy thời gian đăng bài.
            summary_raw = entry.get("summary", "")     # Lấy tóm tắt thô (thường chứa HTML).
            image_url = None
            
            # Trích xuất URL hình ảnh từ trong thẻ <img> của tóm tắt.
            if summary_raw:
                soup = BeautifulSoup(summary_raw, 'html.parser')
                img_tag = soup.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    image_url = img_tag['src']
            
            source_name = get_source_name(entry.link) # Lấy tên nguồn.
            
            # Xử lý và kiểm tra thời gian đăng bài.
            if published_time:
                try:
                    # Chuyển chuỗi thời gian thành đối tượng datetime có nhận biết múi giờ (UTC).
                    parsed_time = parse_date(published_time).astimezone(timezone.utc)
                    # Chỉ lấy bài mới hơn mốc thời gian đã định.
                    if parsed_time >= time_threshold:
                        articles.append({
                            "title": entry.title,
                            "link": entry.link,
                            "summary_raw": summary_raw,
                            "published_time": parsed_time.isoformat(), # Lưu dưới dạng chuỗi chuẩn ISO.
                            "image_url": image_url,
                            "source": source_name
                        })
                        seen_titles.add(norm_title) # Đánh dấu đã thấy.
                        seen_links.add(link)
                except (ValueError, TypeError): # Bỏ qua nếu định dạng thời gian không hợp lệ.
                    continue
    print(f"-> Đã tìm thấy {len(articles)} bài viết mới (sau khi lọc trùng lặp).")
    return pd.DataFrame(articles)


def clean_text(df):
    """
    Chức năng: Làm sạch và tiền xử lý văn bản từ cột 'summary_raw'.
    Kịch bản: Bước thứ hai của pipeline, chuẩn bị dữ liệu văn bản cho việc vector hóa.
    Logic:
    1. Chuyển thành chữ thường, loại bỏ thẻ HTML và các ký tự không phải chữ/số.
    2. Loại bỏ các bài viết có tóm tắt quá ngắn (dưới 10 từ).
    3. Sử dụng `pyvi` để tách từ tiếng Việt.
    4. Loại bỏ các từ dừng (stopwords) trong tiếng Việt.
    Trả về: DataFrame đã được làm sạch và có thêm các cột mới.
    """
    print("\n2. Đang làm sạch văn bản...")
    
    # B1: Làm sạch cơ bản bằng các hàm xử lý chuỗi của pandas.
    summary = (df['summary_raw']
               .str.lower() # Chuyển chữ thường
               .str.replace(r'<.*?>', '', regex=True) # Bỏ thẻ HTML
               .str.replace(r'[^\w\s]', '', regex=True)) # Bỏ dấu câu
    df['summary_cleaned'] = summary

    # B2: Bỏ các dòng có tóm tắt rỗng hoặc NaN.
    df.dropna(subset=['summary_cleaned'], inplace=True)
    df = df[df['summary_cleaned'].str.strip() != '']

    # B3: Lọc các bài viết có ít hơn 10 từ trong tóm tắt.
    original_count = len(df)
    df = df[df['summary_cleaned'].str.split().str.len() >= 10]
    print(f"-> Đã lọc {original_count - len(df)} bài quá ngắn.")

    # B4: Tách từ tiếng Việt và loại bỏ từ dừng.
    vi_stop = stopwords("vi")
    def remove_stop_pyvi(text):
        tokens = ViTokenizer.tokenize(text).split() # Tách từ
        filtered = [t for t in tokens if t not in vi_stop] # Lọc stopwords
        return " ".join(filtered)

    df['summary_not_stop_word'] = df['summary_cleaned'].apply(remove_stop_pyvi)

    df = df.reset_index(drop=True) # Reset lại chỉ số của DataFrame.
    print("-> Hoàn tất làm sạch.")
    return df


def vectorize_text(sentences, model_name):
    """
    Chức năng: Chuyển đổi một danh sách các câu thành các vector số học (embeddings).
    Kịch bản: Bước ba của pipeline, sau khi đã có văn bản sạch.
    Sử dụng: Mô hình Sentence-BERT (SBERT) để nắm bắt ngữ nghĩa của câu.
    Trả về: Một mảng NumPy chứa các vector embeddings.
    """
    print(f"\n3. Đang vector hóa văn bản bằng mô hình {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True) # `show_progress_bar=True` để hiển thị thanh tiến trình.
    print("-> Vector hóa hoàn tất.")
    return embeddings


def find_optimal_clusters(embeddings, max_clusters=20):
    """
    Chức năng: Tìm số cụm (K) tối ưu bằng phương pháp Silhouette.
    Kịch bản: Chạy trước khi phân cụm chính thức để xác định giá trị K tốt nhất.
    Logic:
    1. Thử chạy K-Means với các giá trị K khác nhau (từ 2 đến max_clusters).
    2. Với mỗi giá trị K, tính điểm Silhouette. Điểm này đo lường mức độ "chặt chẽ" và "tách biệt" của các cụm.
    3. Chọn giá trị K có điểm Silhouette cao nhất làm K tối ưu.
    Trả về: Số nguyên là số cụm tối ưu (optimal_k).
    """
    print("\n3.1. Đang tìm số cụm (K) tối ưu bằng hệ số Silhouette...")
    
    silhouette_scores = []
    possible_k_values = range(2, max_clusters + 1)

    for k in possible_k_values:
        # Chạy K-Means với số cụm k. n_init=10 để chạy lại 10 lần và chọn kết quả tốt nhất.
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(embeddings)
        # Tính điểm silhouette cho kết quả phân cụm này.
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"   - Với K={k}, điểm Silhouette = {score:.4f}")

    # Tìm chỉ số của điểm cao nhất trong danh sách điểm.
    best_k_index = np.argmax(silhouette_scores)
    # Lấy giá trị K tương ứng với chỉ số đó.
    optimal_k = possible_k_values[best_k_index]

    print(f"-> Đã tìm thấy số cụm tối ưu là K={optimal_k} với điểm Silhouette cao nhất.")
    return optimal_k


def generate_meaningful_topic_name(keywords, sample_titles):
    """
    Chức năng: Sử dụng API của Google Gemini để tạo ra một tên chủ đề có ý nghĩa.
    Kịch bản: Được gọi bởi hàm get_topic_labels cho mỗi cụm đã được tạo.
    Logic: Gửi một prompt (câu lệnh) chi tiết đến Gemini, bao gồm các từ khóa chính và
           một vài tiêu đề ví dụ, sau đó nhận về một tên chủ đề ngắn gọn.
    Trả về: Một chuỗi là tên chủ đề.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Sử dụng mô hình Flash nhanh và tiết kiệm.
        prompt = f"""Bạn là một trợ lý biên tập báo chí. Dựa vào các thông tin dưới đây, hãy tạo ra chỉ một tên chủ đề ngắn gọn duy nhất (không quá 6 từ) bằng tiếng Việt để tóm tắt nội dung chính.
        Các từ khóa chính của chủ đề: {keywords}
        Một vài tiêu đề bài viết ví dụ:
        - {"\n- ".join(sample_titles)}
        Tên chủ đề gợi ý:"""
        response = model.generate_content(prompt)
        return response.text.strip().replace("*", "") # Làm sạch output từ Gemini.
    except Exception as e:
        # Nếu có lỗi khi gọi API (ví dụ: hết hạn mức, lỗi mạng), sẽ dùng chính các từ khóa làm nhãn.
        print(f"   - Lỗi khi gọi Gemini API: {type(e).__name__} - {e}. Sử dụng từ khóa làm nhãn thay thế.")
        return keywords


def get_topic_labels(df, num_keywords=5):
    """
    Chức năng: Tạo nhãn (tên) có ý nghĩa cho mỗi cụm chủ đề.
    Kịch bản: Chạy sau khi đã phân cụm xong.
    Logic:
    1. Với mỗi cụm, trích xuất các bài viết thuộc cụm đó.
    2. Dùng TF-IDF để tìm ra các từ khóa quan trọng nhất trong cụm.
    3. Gọi hàm `generate_meaningful_topic_name` để tạo tên chủ đề từ các từ khóa và tiêu đề mẫu.
    Trả về: Một dictionary ánh xạ ID cụm sang tên chủ đề.
    """
    print("\n5. Đang gán nhãn chủ đề cho các cụm...")
    topic_labels = {}
    actual_clusters = df['topic_cluster'].nunique() # Lấy số cụm thực tế từ dữ liệu.
    
    for i in range(actual_clusters):
        cluster_df = df[df['topic_cluster'] == i]
        cluster_texts = cluster_df['summary_cleaned'].tolist()
        
        # Nếu cụm quá nhỏ, không cần phân tích phức tạp.
        if len(cluster_texts) < 3:
            topic_labels[str(i)] = "Chủ đề nhỏ (ít bài viết)"
            continue
            
        # Dùng TF-IDF để tìm từ khóa.
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1 # Tính điểm TF-IDF trung bình cho mỗi từ.
        top_indices = avg_tfidf_scores.argsort()[-num_keywords:][::-1] # Lấy chỉ số của các từ có điểm cao nhất.
        feature_names = vectorizer.get_feature_names_out()
        keywords = ", ".join([feature_names[j] for j in top_indices])
        
        # Lấy 3 tiêu đề mẫu.
        sample_titles = cluster_df['title'].head(3).tolist()
        
        # Gọi Gemini API để tạo tên.
        meaningful_name = generate_meaningful_topic_name(keywords, sample_titles)
        print(f"   - Cluster {i}: {keywords}   =>   Tên chủ đề: {meaningful_name}")
        topic_labels[str(i)] = meaningful_name
        
        # Tạm dừng một chút để tránh vượt quá giới hạn request của API (rate limiting).
        time.sleep(1.1)
        
    print("-> Gán nhãn chủ đề hoàn tất.")
    return topic_labels


def main_pipeline():
    """
    Hàm chính điều phối toàn bộ quy trình xử lý dữ liệu.
    """
    print("\n🚀 BẮT ĐẦU QUY TRÌNH XỬ LÝ DỮ LIỆU 🚀")
    
    # BƯỚC 1: Lấy dữ liệu bài viết mới.
    df = fetch_recent_articles(RSS_URLS, hours=24)
    if df.empty:
        print("Không có bài viết mới nào. Dừng quy trình.")
        return
        
    # BƯỚC 2: Làm sạch văn bản.
    df = clean_text(df)
    
    # BƯỚC 3: Vector hóa văn bản.
    embeddings = vectorize_text(df['summary_not_stop_word'].tolist(), SBERT_MODEL)
    
    # BƯỚC 3.1: Tìm số cụm tối ưu tự động.
    NUM_CLUSTERS = find_optimal_clusters(embeddings)
    
    # BƯỚC 4: Chạy K-Means cuối cùng với số cụm tối ưu đã tìm được.
    print(f"\n4. Đang thực hiện phân cụm K-Means với {NUM_CLUSTERS} cụm...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df['topic_cluster'] = kmeans.fit_predict(embeddings)
    print("-> Phân cụm hoàn tất.")

    # In ra thống kê số lượng bài viết trong mỗi cụm.
    print("\n📊 Phân bổ số lượng bài viết vào các cụm (chủ đề):")
    cluster_counts = df['topic_cluster'].value_counts().sort_index()
    print(cluster_counts)
    
    # BƯỚC 5: Gán nhãn cho các chủ đề.
    topic_labels = get_topic_labels(df)
    
    # BƯỚC 6: Tính toán ma trận tương đồng.
    print("\n6. Đang tính toán ma trận tương đồng...")
    cosine_sim = cosine_similarity(embeddings)
    print("-> Tính toán hoàn tất.")
    
    # BƯỚC 7: Lưu tất cả các kết quả ra file để ứng dụng Streamlit sử dụng.
    print("\n7. Đang lưu các file kết quả...")
    df.to_csv('final_articles_for_app.csv', index=False, encoding='utf-8-sig') # Dùng 'utf-8-sig' để Excel đọc đúng tiếng Việt.
    np.save('cosine_similarity_matrix.npy', cosine_sim)
    np.save('embeddings.npy', embeddings)
    with open('topic_labels.json', 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f, ensure_ascii=False, indent=4) # `ensure_ascii=False` để lưu đúng tiếng Việt.
    
    print("\n✅ QUY TRÌNH HOÀN TẤT! ✅")


# Điểm bắt đầu của chương trình khi chạy file này trực tiếp.
if __name__ == "__main__":
    main_pipeline()