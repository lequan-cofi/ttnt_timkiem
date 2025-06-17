# ==== IMPORTS ====
import streamlit as st  # Framework để xây dựng giao diện web
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
import numpy as np  # Thư viện tính toán số học
import json  # Xử lý dữ liệu JSON
import re  # Xử lý chuỗi với regular expression
import sys  # Truy cập các biến và hàm của hệ thống
import subprocess  # Chạy các lệnh hệ thống
import requests  # Gửi HTTP requests
from bs4 import BeautifulSoup  # Parse và xử lý HTML
from urllib.parse import urlparse, quote, unquote  # Xử lý URL
from sklearn.metrics.pairwise import cosine_similarity  # Tính độ tương đồng cosine
from sentence_transformers import SentenceTransformer  # Mô hình ngôn ngữ để tạo vector
# from streamlit_extras.app_logo import app_logo
# --- CẤU HÌNH TRANG VÀ CSS ---

st.set_page_config(page_title="Tạp chí của bạn", page_icon="📖", layout="wide")

# ==== KHỞI TẠO TRẠNG THÁI ====
# Khởi tạo các biến session state để lưu trữ trạng thái
if 'read_articles' not in st.session_state:
    st.session_state.read_articles = set()  # Tập hợp các bài viết đã đọc
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []  # Lịch sử đọc bài viết
if 'current_view' not in st.session_state:
    st.session_state.current_view = "main"  # View hiện tại (main/detail)
if 'current_article_id' not in st.session_state:
    st.session_state.current_article_id = None  # ID bài viết đang xem
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "Dành cho bạn (Tất cả)"  # Chủ đề đã chọn
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []  # Nguồn tin đã chọn

@st.cache_resource
def get_sbert_model():
    """Lấy mô hình SBERT đã được cache."""
    return SentenceTransformer('Cloyne/vietnamese-sbert-v3')

def local_css(file_name):
    """Đọc và áp dụng CSS từ file."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{file_name}'.")

# ==== HÀM TẢI DỮ LIỆU ---
@st.cache_data
def load_data_and_embeddings():
    """Tải dữ liệu và embeddings từ các file."""
    # Đọc dữ liệu từ CSV
    df = pd.read_csv('final_articles_for_app.csv')
    # Đọc ma trận độ tương đồng
    cosine_sim = np.load('cosine_similarity_matrix.npy')
    # Đọc nhãn chủ đề
    with open('topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
    # Chuyển đổi thời gian với xử lý lỗi
    try:
        df['published_time'] = pd.to_datetime(df['published_time'], errors='coerce')
        # Loại bỏ các bài viết có thời gian không hợp lệ
        df = df.dropna(subset=['published_time'])
    except Exception as e:
        st.error(f"Lỗi khi chuyển đổi thời gian: {e}")
        # Nếu lỗi, giữ nguyên dạng string
        pass
    # Sử dụng cột source trực tiếp từ CSV
    df['source_name'] = df['source']
    # Đọc embeddings
    try:
        embeddings = np.load('embeddings.npy')
    except:
        st.error("Không tìm thấy file embeddings.npy. Vui lòng chạy lại pipeline để tạo embeddings.")
        st.stop()
    return df, cosine_sim, topic_labels, embeddings

def calculate_average_vector(article_ids, cosine_sim):
    """Tính vector trung bình từ 5 bài viết gần nhất."""
    if not article_ids:
        return None
    
    vectors = []
    for article_id in article_ids:
        if article_id < len(cosine_sim):
            vectors.append(cosine_sim[article_id])
    
    if not vectors:
        return None
    
    return np.mean(vectors, axis=0)

def get_similar_articles_by_history(df, cosine_sim, history_articles, exclude_articles=None):
    """Lấy bài viết tương tự dựa trên lịch sử đọc."""
    if not history_articles:
        return pd.DataFrame()
    
    # Lấy 5 bài viết mới đọc gần nhất
    recent_articles = history_articles[:5]
    
    # Tính vector trung bình
    avg_vector = calculate_average_vector(recent_articles, cosine_sim)
    if avg_vector is None:
        return pd.DataFrame()
    
    # Tính độ tương đồng
    similarity_scores = cosine_similarity([avg_vector], cosine_sim)[0]
    
    # Loại bỏ bài đã đọc
    if exclude_articles:
        mask = ~df.index.isin(exclude_articles)
        similarity_scores[~mask] = -1
    
    # Lấy top bài viết tương tự
    top_indices = np.argsort(similarity_scores)[::-1][:10]
    similar_articles = df.iloc[top_indices].copy()
    similar_articles['similarity_score'] = similarity_scores[top_indices]
    
    return similar_articles

def crawl_article_content(url):
    """Crawl nội dung bài viết từ URL và làm sạch HTML.
    
    Hàm này thực hiện các nhiệm vụ:
    1. Tải nội dung trang web từ URL được cung cấp
    2. Phân tích HTML để tìm nội dung chính của bài viết
    3. Làm sạch HTML bằng cách loại bỏ các thẻ và thuộc tính không cần thiết
    4. Xử lý các thẻ đặc biệt như ảnh và liên kết
    5. Trả về nội dung đã được làm sạch dưới dạng chuỗi HTML
    """

    try:
        # Thiết lập headers để giả lập trình duyệt thật
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive'
        }
        
        # Tải nội dung trang web với timeout 10 giây
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'  # Đảm bảo encoding tiếng Việt
        soup = BeautifulSoup(response.text, 'html.parser')  # Parse HTML

        # ==== XỬ LÝ ẢNH ====
        # Chuyển đổi ảnh lazy-load sang ảnh thường
        for img in soup.find_all('img'):
            # Nếu ảnh có thuộc tính data-src (lazy-load), chuyển thành src
            if img.has_attr('data-src'):
                img['src'] = img['data-src']
            # Xóa ảnh không có src hoặc src rỗng
            if not img.has_attr('src') or not img['src'].strip():
                img.decompose()
            # Thêm thuộc tính lazy-load cho tất cả ảnh
            img['loading'] = 'lazy'

        # Xóa các thẻ figure không chứa nội dung
        for fig in soup.find_all('figure'):
            if not fig.get_text(strip=True) and not fig.find('img'):
                fig.decompose()

        article = None
        
        # ==== XỬ LÝ THEO TỪNG TRANG BÁO ====
        # VnExpress
        if 'vnexpress.net' in url:
            # Tìm nội dung chính theo thứ tự ưu tiên các class
            article = soup.find('article', class_='fck_detail')
            if not article:
                # Nếu không tìm thấy, lấy article đầu tiên
                article = soup.find('article')
            if article:
                # Xóa các thẻ không cần thiết
                for tag in article.find_all(['script', 'style', 'iframe']):
                    tag.decompose()
                return str(article)

        # Tuổi Trẻ và Thanh Niên
        elif 'tuoitre.vn' in url or 'thanhnien.vn' in url:
            # Tìm nội dung theo class detail-content
            article = soup.find('div', class_='detail-content')
            if not article:
                # Nếu không tìm thấy, lấy div có nhiều text nhất
                divs = soup.find_all('div')
                article = max(divs, key=lambda d: len(d.get_text(strip=True))) if divs else None

        # Dân Trí
        elif 'dantri.com.vn' in url:
            # Tìm nội dung theo class dt-news__content
            article = soup.find('div', class_='dt-news__content')
            if not article:
                article = soup.find('article')
            if not article:
                # Nếu không tìm thấy, lấy div có nhiều text nhất
                divs = soup.find_all('div')
                article = max(divs, key=lambda d: len(d.get_text(strip=True))) if divs else None

        # ==== LÀM SẠCH NỘI DUNG ====
        if article:
            # 1. Xóa các thẻ không cần thiết
            for tag in article.find_all(['script', 'style', 'iframe', 'button', 'ins', 'noscript', 'form']):
                tag.decompose()

            # 2. Xử lý các thẻ <a> (liên kết)
            for a in article.find_all('a'):
                # Mở link trong tab mới
                a['target'] = '_blank'
                # Thêm thuộc tính bảo mật
                a['rel'] = 'noopener noreferrer'
                # Xóa các thuộc tính JavaScript
                if a.has_attr('onclick'):
                    del a['onclick']
                # Xóa các thuộc tính không cần thiết
                for attr in ['data-', 'on', 'class', 'id']:
                    for a_attr in list(a.attrs):
                        if a_attr.startswith(attr):
                            del a[a_attr]

            # 3. Xử lý các thẻ <img> (ảnh)
            for img in article.find_all('img'):
                # Thêm lazy-load
                img['loading'] = 'lazy'
                # Chuẩn hóa đường dẫn ảnh
                if img.has_attr('src') and not img['src'].startswith(('http://', 'https://')):
                    img['src'] = 'https://vnexpress.net' + img['src']
                # Xóa các thuộc tính không cần thiết
                for attr in ['data-', 'on', 'class', 'id']:
                    for img_attr in list(img.attrs):
                        if img_attr.startswith(attr):
                            del img[img_attr]

            # 4. Xóa các div rỗng
            for div in article.find_all('div'):
                if not div.get_text(strip=True) and not div.find('img'):
                    div.decompose()

            # 5. Xóa mọi thuộc tính không cần thiết
            for tag in article.find_all(True):
                # Chỉ giữ lại các thuộc tính cần thiết
                allowed_attrs = ['src', 'alt', 'href', 'target', 'rel', 'loading']
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in allowed_attrs}

            return str(article)

        return None

    except Exception as e:
        # In lỗi để debug
        print(f"Lỗi khi crawl bài viết từ {url}: {str(e)}")
        return None

def render_detail_view(article_id, df, cosine_sim, topic_labels):
    """Hiển thị chi tiết bài viết và các bài viết liên quan."""
    try:
        # Lấy thông tin bài viết
        article = df.loc[article_id]
    except KeyError:
        st.error("Không tìm thấy bài viết.")
        if st.button("⬅️ Quay lại danh sách"):
            st.session_state.current_view = "main"
            st.session_state.current_article_id = None
            st.rerun()
        return
    
    # Cập nhật lịch sử đọc
    st.session_state.read_articles.add(article_id)
    if article_id not in st.session_state.reading_history:
        st.session_state.reading_history.insert(0, article_id)
    
    # CSS cho giao diện
    st.markdown("""
        <style>
            /* Ẩn sidebar khi mở rộng */
            [data-testid="stSidebar"][aria-expanded="true"]{
                display: none;
            }
          
            /* Style cho nội dung bài viết */
            .article-content, .fck_detail, .detail-content, .dt-news__content {
                font-size: 15px !important;
                line-height: 1.6 !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            /* Style cho ảnh trong bài viết */
            .article-content img,
            .fck_detail img,
            .detail-content img,
            .dt-news__content img {
                max-width: 100% !important;
                height: auto !important;
                display: block !important;
                margin: 10px auto !important;
                border-radius: 8px !important;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
            }
            
            /* Style cho liên kết */
            .article-content a,
            .fck_detail a,
            .detail-content a,
            .dt-news__content a {
                color: #0066cc !important;
                text-decoration: none !important;
                border-bottom: 1px solid transparent !important;
                transition: border-color 0.2s ease !important;
            }
            .article-content a:hover,
            .fck_detail a:hover,
            .detail-content a:hover,
            .dt-news__content a:hover {
                border-bottom-color: #0066cc !important;
            }
            
            /* Style cho blockquote */
            .article-content blockquote,
            .fck_detail blockquote,
            .detail-content blockquote,
            .dt-news__content blockquote {
                border-left: 4px solid #0066cc !important;
                margin: 1em 0 !important;
                padding: 0.5em 1em !important;
                background: #f8f9fa !important;
                font-style: italic !important;
            }
            
            /* Style cho đoạn văn */
            .article-content p,
            .fck_detail p,
            .detail-content p,
            .dt-news__content p {
                margin: 1em 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Nút quay lại
    if st.button("⬅️ Quay lại danh sách"):
        st.session_state.current_view = "main"
        st.session_state.current_article_id = None
        st.rerun()
    
    # Hiển thị tiêu đề và thông tin bài viết
    st.title(article['title'])
    
    # Giữ nguyên thời gian từ RSS (đã đúng múi giờ Việt Nam)
    vn_time = article['published_time']
    
    st.caption(f"Nguồn: {article['source_name']} | Xuất bản: {vn_time.strftime('%d-%m-%Y %H:%M')}")
    st.markdown("---")
    
    # Chia layout thành 2 cột
    col1, col2 = st.columns([0.6, 0.4])
    
    # Cột trái: Nội dung bài viết
    with col1:
        # Hiển thị ảnh đại diện
        if pd.notna(article['image_url']):
            st.markdown(f'<img src="{article["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';" style="width:100%;">', unsafe_allow_html=True)
        else:
            st.markdown('<img src="no-image-png-2.webp" style="width:100%;">', unsafe_allow_html=True)
        
        # Tải và hiển thị nội dung bài viết
        with st.spinner("Đang tải nội dung bài viết..."):
            article_content = crawl_article_content(article['link'])
            if article_content:
                # Bọc nội dung trong container
                article_content = f'<div class="article-content">{article_content}</div>'
                st.markdown(article_content, unsafe_allow_html=True)
            else:
                # Hiển thị tóm tắt nếu không lấy được nội dung đầy đủ
                st.warning("Không thể tải nội dung đầy đủ của bài viết. Hiển thị tóm tắt thay thế.")
                summary_raw = article.get('summary_raw', '')
                summary_without_img = re.sub(r'<img[^>]*>', '', summary_raw, flags=re.IGNORECASE)
                st.markdown(f'<div class="article-content">{summary_without_img}</div>', unsafe_allow_html=True)
        
        # Link đến bài viết gốc
        st.link_button("Đọc toàn bộ bài viết trên trang gốc", article['link'])
    
    # Cột phải: Bài viết liên quan
    with col2:
        st.subheader("Khám phá thêm")
        # Cho phép chọn loại bài viết liên quan
        rec_type = st.radio("Hiển thị các bài viết:", ("Có nội dung tương tự", "Trong cùng chủ đề"), key=f"rec_type_{article_id}")
        
        if rec_type == "Có nội dung tương tự":
            # Hiển thị bài viết tương tự dựa trên độ tương đồng ngữ nghĩa
            st.markdown("##### Dựa trên phân tích ngữ nghĩa:")
            sim_scores = sorted(list(enumerate(cosine_sim[article_id])), key=lambda x: x[1], reverse=True)[1:6]
            for i, (article_index, score) in enumerate(sim_scores):
                rec_article = df.iloc[article_index]
                with st.container(border=True):
                    rec_col1, rec_col2 = st.columns([0.25, 0.75])
                    with rec_col1:
                        if pd.notna(rec_article['image_url']):
                            st.markdown(f'<img src="{rec_article["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';" style="width:100%;">', unsafe_allow_html=True)
                        else:
                            st.markdown('<img src="no-image-png-2.webp" style="width:100%;">', unsafe_allow_html=True)
                    with rec_col2:
                        if st.button(rec_article['title'], key=f"rec_{article_index}"):
                            # Cập nhật lịch sử đọc cho bài viết được chọn
                            st.session_state.read_articles.add(article_index)
                            if article_index not in st.session_state.reading_history:
                                st.session_state.reading_history.insert(0, article_index)
                            st.session_state.current_article_id = article_index
                            st.rerun()
                        st.caption(f"Nguồn: {rec_article['source_name']} | Độ tương đồng: {score:.2f}")
        else:
            # Hiển thị bài viết cùng chủ đề
            cluster_id = article['topic_cluster']
            topic_name = topic_labels.get(str(cluster_id), "N/A")
            st.markdown(f"##### Thuộc chủ đề: **{topic_name}**")
            same_cluster_df = df[(df['topic_cluster'] == cluster_id) & (df.index != article_id)].head(5)
            for i, row in same_cluster_df.iterrows():
                with st.container(border=True):
                    rec_col1, rec_col2 = st.columns([0.25, 0.75])
                    with rec_col1:
                        if pd.notna(row['image_url']):
                            st.markdown(f'<img src="{row["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';" style="width:100%;">', unsafe_allow_html=True)
                        else:
                            st.markdown('<img src="no-image-png-2.webp" style="width:100%;">', unsafe_allow_html=True)
                    with rec_col2:
                        if st.button(row['title'], key=f"rec_{i}"):
                            # Cập nhật lịch sử đọc cho bài viết được chọn
                            st.session_state.read_articles.add(i)
                            if i not in st.session_state.reading_history:
                                st.session_state.reading_history.insert(0, i)
                            st.session_state.current_article_id = i
                            st.rerun()
                        similarity_score = cosine_sim[article_id][i]
                        st.caption(f"Nguồn: {row['source_name']} | Độ tương đồng: {similarity_score:.2f}")

def render_search_results(query, df, embeddings, sbert_model):
    """Hiển thị kết quả tìm kiếm."""
    st.header(f"Kết quả tìm kiếm cho: \"{query}\"")
    # Vector hóa câu truy vấn
    with st.spinner("Đang phân tích và tìm kiếm..."):
        query_vector = sbert_model.encode([query])
        # Tính độ tương đồng
        similarities = cosine_similarity(query_vector, embeddings)[0]
        # Sắp xếp và lấy kết quả
        sim_scores = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)
        result_indices = [i[0] for i in sim_scores]
        result_df = df.iloc[result_indices].copy()
    render_main_grid(result_df, f"Kết quả cho: \"{query}\"")

# ==== CÁC HÀM HIỂN THỊ (RENDER) ---
def render_main_grid(df, selected_topic_name):
    """Hiển thị lưới bài viết chính."""
    st.header(f"Bảng tin: {selected_topic_name}")
    st.markdown(f"Tìm thấy **{len(df)}** bài viết liên quan.")
    st.markdown("---")
    
    # Chia layout thành 3 cột
    num_columns = 3
    cols = st.columns(num_columns)
    
    if df.empty:
        st.warning("Không có bài viết nào phù hợp với lựa chọn của bạn.")
    else:
        # Hiển thị từng bài viết
        for i, (index, row) in enumerate(df.iterrows()):
            with cols[i % num_columns]:
                # Xử lý hình ảnh
                image_html = ''
                if pd.notna(row["image_url"]):
                    image_html = f'<div class="card-image-container"><img src="{row["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';"></div>'
                else:
                    image_html = '<div class="card-image-container"><img src="no-image-png-2.webp"></div>'
                
                # Tạo card bài viết
                card_html = f"""<div class="article-card">
                                {image_html}
                                <div class="article-content">
                                    <div class="article-title">{row['title']}</div>
                                    <div class="article-source">{row['source_name']}</div>
                                </div>
                           </div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Nút đọc bài viết
                if st.button("Đọc bài viết", key=f"read_{index}"):
                    st.session_state.current_article_id = index
                    st.session_state.current_view = "detail"
                    st.rerun()

# --- LUỒNG CHÍNH CỦA ỨNG DỤNG ---
local_css("style.css")

# --- PHẦN LOGIC MỚI: QUẢN LÝ TRẠNG THÁI ---
if 'update_log' not in st.session_state:
    st.session_state.update_log = ""
if 'update_error' not in st.session_state:
    st.session_state.update_error = ""
if 'update_success' not in st.session_state:
    st.session_state.update_success = False

# Initialize session state for reading history and interest tracking
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []
if 'interest_vector' not in st.session_state:
    st.session_state.interest_vector = None
if 'interest_articles' not in st.session_state:
    st.session_state.interest_articles = None

df, cosine_sim, topic_labels, embeddings = load_data_and_embeddings()
sbert_model = get_sbert_model()

# --- Ô TÌM KIẾM SEMANTIC Ở HEADER ---
search_col1, search_col2 = st.columns([0.85, 0.15])
with search_col1:
    search_input = st.text_input(
        "Tìm kiếm bài viết (theo ngữ nghĩa, nhập từ khóa hoặc câu hỏi):",
        value=st.session_state.get('search_query', ''),
        key="search_input",
        placeholder="Nhập nội dung bạn muốn tìm...",
        label_visibility="collapsed"
    )
with search_col2:
    search_button = st.button("🔍 Tìm kiếm", use_container_width=True)

if search_input and (search_button or search_input != st.session_state.get('search_query', '')):
    st.session_state['search_query'] = search_input
    st.session_state['current_view'] = "search"
    st.rerun()

if st.session_state.get('current_view', 'main') == "search" and st.session_state.get('search_query', ''):
    with st.spinner("Đang phân tích và tìm kiếm..."):
        query_vector = sbert_model.encode([st.session_state['search_query']])
        similarities = cosine_similarity(query_vector, embeddings)[0]
        sim_scores = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)
        result_indices = [i[0] for i in sim_scores]
        result_df = df.iloc[result_indices].copy()
    st.sidebar.info("Bạn đang ở trang kết quả tìm kiếm. Chọn danh mục khác hoặc bấm 'Quay lại' để trở về.")
    if st.sidebar.button("⬅️ Quay lại trang chủ", use_container_width=True):
        st.session_state['search_query'] = ''
        st.session_state['current_view'] = "main"
        st.rerun()
    render_main_grid(result_df, f"Kết quả cho: \"{st.session_state['search_query']}\"")
    st.stop()
elif st.session_state.current_view == "detail" and st.session_state.current_article_id is not None:
    render_detail_view(st.session_state.current_article_id, df, cosine_sim, topic_labels)
else:
    # --- GIAO DIỆN THANH BÊN ---
    st.sidebar.title("Tạp chí của bạn")
    st.sidebar.markdown("---")

    # Nút cập nhật dữ liệu
    if st.sidebar.button("🔄 Cập nhật dữ liệu", use_container_width=True):
        with st.spinner("⏳ Đang tải lại dữ liệu..."):
            try:
                # Xóa cache để buộc tải lại dữ liệu từ CSV
                st.cache_data.clear()
                st.session_state.update_log = "Đã xóa cache và tải lại dữ liệu từ final_articles_for_app.csv"
                st.session_state.update_error = ""
                st.session_state.update_success = True
            except Exception as e:
                st.session_state.update_error = f"Lỗi khi tải lại dữ liệu: {e}"
                st.session_state.update_success = False

    # Hiển thị kết quả cập nhật và nút tải lại
    if st.session_state.update_success:
        st.sidebar.success("✅ Đã tải lại dữ liệu!")
        with st.sidebar.expander("Xem chi tiết"):
            st.info(st.session_state.update_log)
            if st.session_state.update_error:
                st.error("Lỗi:")
                st.code(st.session_state.update_error)
        if st.sidebar.button("Xem dữ liệu mới", use_container_width=True):
            st.session_state.update_success = False # Reset cờ
            st.rerun()

    st.sidebar.markdown("---")

    if df is None:
        st.error("Lỗi: Không tìm thấy file dữ liệu. Vui lòng bấm nút 'Cập nhật dữ liệu' ở thanh bên.")
    else:
        # --- PHẦN LỌC THEO CHỦ ĐỀ ---
        st.sidebar.subheader("Khám phá các chủ đề")
        topic_display_list = ["Dành cho bạn (Tất cả)", "Bài viết đã đọc", "Dựa trên lịch sử đọc"] + [v for k, v in topic_labels.items()]
        
        st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        for topic in topic_display_list:
            is_active = (topic == st.session_state.selected_topic)
            active_class = "active" if is_active else ""
            icon = "📖" if topic != "Bài viết đã đọc" and topic != "Dựa trên lịch sử đọc" else "👀"if topic == "Bài viết đã đọc" else "🎯"
            if st.sidebar.button(f"{icon} {topic}", key=f"topic_{topic}", use_container_width=True):
                st.session_state.selected_topic = topic
                st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        st.sidebar.markdown("---")

        # --- BỔ SUNG: PHẦN LỌC THEO NGUỒN ---
        st.sidebar.subheader("Lọc theo nguồn")
        all_sources = sorted(df['source_name'].unique().tolist())
        selected_sources = st.sidebar.multiselect(
            "Chọn một hoặc nhiều nguồn:",
            options=all_sources,
            default=st.session_state.selected_sources
        )
        
        if selected_sources != st.session_state.selected_sources:
            st.session_state.selected_sources = selected_sources
            st.rerun()

        # --- HIỂN THỊ VIEW TƯƠNG ỨNG ---
        if st.session_state.selected_topic == "Bài viết đã đọc":
            if st.session_state.read_articles:
                # Lấy danh sách bài viết đã đọc theo thứ tự trong reading_history (mới nhất lên đầu)
                ordered_articles = [article_id for article_id in st.session_state.reading_history if article_id in st.session_state.read_articles]
                # Tạo DataFrame với thứ tự đã sắp xếp
                display_df = df[df.index.isin(ordered_articles)].copy()
                # Sắp xếp lại theo thứ tự trong ordered_articles
                display_df = display_df.reindex(ordered_articles)
            else:
                display_df = pd.DataFrame()
                st.info("Bạn chưa đọc bài viết nào.")
        elif st.session_state.selected_topic == "Dựa trên lịch sử đọc":
            if len(st.session_state.reading_history) > 0:
                display_df = get_similar_articles_by_history(
                    df, cosine_sim,
                    st.session_state.reading_history,
                    exclude_articles=st.session_state.read_articles
                )
                if display_df.empty:
                    st.info("Không tìm thấy bài viết tương tự dựa trên lịch sử đọc.")
            else:
                display_df = pd.DataFrame()
                st.info("Bạn chưa có lịch sử đọc bài viết nào.")
        elif st.session_state.selected_topic != "Dành cho bạn (Tất cả)":
            selected_key_list = [k for k, v in topic_labels.items() if v == st.session_state.selected_topic]
            if selected_key_list:
                display_df = df[df['topic_cluster'] == int(selected_key_list[0])].copy()
            else:
                display_df = pd.DataFrame()
        else:
            display_df = df.copy()

        # Áp dụng bộ lọc nguồn
        if st.session_state.selected_sources:
            display_df = display_df[display_df['source_name'].isin(st.session_state.selected_sources)]

        # Sắp xếp và hiển thị
        if not display_df.empty:
            # Chỉ sắp xếp theo thời gian đăng nếu không phải là bài viết đã đọc
            if st.session_state.selected_topic != "Bài viết đã đọc":
                # Sắp xếp đơn giản như Excel - so sánh trực tiếp datetime
                display_df = display_df.sort_values(by='published_time', ascending=False, kind='mergesort')
        render_main_grid(display_df, st.session_state.selected_topic)

