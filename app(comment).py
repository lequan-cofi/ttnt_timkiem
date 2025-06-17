# ==== IMPORTS ====
# Khai báo các thư viện cần thiết cho ứng dụng.
import streamlit as st  # Framework chính để xây dựng giao diện web.
import pandas as pd  # Dùng để xử lý dữ liệu dạng bảng (DataFrame).
import numpy as np  # Dùng cho các phép toán số học, đặc biệt là với mảng và ma trận.
import json  # Dùng để làm việc với dữ liệu định dạng JSON (ở đây là file nhãn chủ đề).
import re  # Dùng cho Regular Expressions, để xử lý và làm sạch chuỗi văn bản.
import sys  # Cung cấp quyền truy cập vào các biến và hàm của hệ thống Python (dùng để chạy pipeline).
import subprocess  # Cho phép chạy các tiến trình mới, lệnh hệ thống (dùng để chạy pipeline.py).
import requests  # Dùng để gửi các yêu cầu HTTP (tải nội dung web).
from bs4 import BeautifulSoup  # Dùng để phân tích (parse) và trích xuất dữ liệu từ file HTML.
from urllib.parse import urlparse, quote, unquote  # Các công cụ để xử lý URL.
from sklearn.metrics.pairwise import cosine_similarity  # Hàm tính độ tương đồng Cosine giữa các vector.
from sentence_transformers import SentenceTransformer  # Thư viện chứa mô hình SBERT để chuyển văn bản thành vector.

# --- CẤU HÌNH TRANG VÀ CSS ---

# Thiết lập cấu hình ban đầu cho trang web Streamlit.
# page_title: Tiêu đề hiển thị trên tab của trình duyệt.
# page_icon: Icon hiển thị trên tab của trình duyệt.
# layout="wide": Cho phép nội dung ứng dụng chiếm toàn bộ chiều rộng của màn hình.
st.set_page_config(page_title="Tạp chí của bạn", page_icon="📖", layout="wide")


# ==== KHỞI TẠO TRẠNG THÁI ====
# `st.session_state` là một đối tượng giống từ điển (dictionary) để lưu trữ trạng thái của phiên làm việc.
# Nó giúp giữ lại giá trị của các biến qua mỗi lần Streamlit chạy lại kịch bản (ví dụ: khi người dùng bấm nút).

# Khởi tạo một tập hợp (set) để lưu ID của các bài viết đã đọc. Dùng set để tránh trùng lặp và truy vấn nhanh.
if 'read_articles' not in st.session_state:
    st.session_state.read_articles = set()

# Khởi tạo một danh sách (list) để lưu lịch sử đọc theo thứ tự thời gian (bài mới nhất ở đầu).
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []

# Lưu trạng thái hiển thị hiện tại của ứng dụng: "main" (trang chính), "detail" (chi tiết), "search" (tìm kiếm).
if 'current_view' not in st.session_state:
    st.session_state.current_view = "main"

# Lưu ID của bài viết đang được xem chi tiết.
if 'current_article_id' not in st.session_state:
    st.session_state.current_article_id = None

# Lưu chủ đề đang được chọn ở thanh bên. Mặc định là xem tất cả.
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "Dành cho bạn (Tất cả)"

# Lưu các nguồn tin đang được chọn ở thanh bên.
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []


# Decorator của Streamlit để cache tài nguyên (resource).
# Mô hình ngôn ngữ là một tài nguyên lớn, tải lâu. Decorator này đảm bảo mô hình chỉ được tải MỘT LẦN
# và được tái sử dụng trong suốt phiên làm việc, giúp tăng tốc đáng kể.
@st.cache_resource
def get_sbert_model():
    """
    Chức năng: Tải và cache mô hình SBERT.
    Kịch bản: Được gọi khi ứng dụng cần vector hóa văn bản (lúc tìm kiếm).
    Trả về: Một đối tượng mô hình SentenceTransformer.
    """
    # Trả về mô hình 'Cloyne/vietnamese-sbert-v3', một mô hình đã được huấn luyện cho tiếng Việt.
    return SentenceTransformer('Cloyne/vietnamese-sbert-v3')


def local_css(file_name):
    """
    Chức năng: Đọc nội dung từ một file CSS cục bộ và áp dụng vào ứng dụng Streamlit.
    Kịch bản: Được gọi một lần ở đầu luồng chính để tùy chỉnh giao diện.
    Tham số:
        - file_name (str): Tên của file CSS (ví dụ: 'style.css').
    """
    try:
        # Mở và đọc file với encoding 'utf-8' để hỗ trợ tiếng Việt.
        with open(file_name, "r", encoding="utf-8") as f:
            # Dùng st.markdown để chèn thẻ <style> vào HTML của trang.
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Nếu không tìm thấy file, hiển thị thông báo lỗi.
        st.error(f"Lỗi: Không tìm thấy file '{file_name}'.")


# ==== HÀM TẢI DỮ LIỆU ---
# Decorator của Streamlit để cache dữ liệu.
# Tương tự @st.cache_resource, nhưng dành cho dữ liệu (như DataFrame). Nếu các file đầu vào không thay đổi,
# kết quả của hàm (dữ liệu đã đọc) sẽ được lấy từ cache mà không cần đọc lại file, giúp ứng dụng khởi động nhanh.
@st.cache_data
def load_data_and_embeddings():
    """
    Chức năng: Tải tất cả các file dữ liệu cần thiết cho ứng dụng.
    Kịch bản: Được gọi một lần khi ứng dụng khởi động để nạp dữ liệu vào bộ nhớ.
    Logic:
    1. Đọc file CSV chứa thông tin các bài báo.
    2. Tải ma trận tương đồng Cosine đã được tính toán trước.
    3. Tải file JSON chứa tên của các chủ đề.
    4. Tải file chứa các vector embedding của bài báo.
    5. Thực hiện một số tiền xử lý cơ bản (chuyển đổi cột thời gian).
    Trả về: Một tuple chứa DataFrame, ma trận tương đồng, nhãn chủ đề, và embeddings.
    """
    # Đọc dữ liệu chính từ file CSV vào một DataFrame của pandas.
    df = pd.read_csv('final_articles_for_app.csv')
    # Tải ma trận tương đồng đã được tính toán sẵn từ file .npy (định dạng nhị phân của NumPy).
    cosine_sim = np.load('cosine_similarity_matrix.npy')
    # Mở và đọc file JSON chứa ánh xạ từ ID chủ đề (dạng chuỗi) sang tên chủ đề.
    with open('topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
    # Chuyển đổi cột 'published_time' từ dạng chuỗi sang dạng datetime để có thể sắp xếp và định dạng.
    df['published_time'] = pd.to_datetime(df['published_time'])
    # Tạo một cột mới 'source_name' từ cột 'source' để thống nhất tên.
    df['source_name'] = df['source']
    # Tải các vector embedding từ file .npy.
    try:
        embeddings = np.load('embeddings.npy')
    except:
        # Nếu không tìm thấy file, hiển thị lỗi và dừng ứng dụng.
        st.error("Không tìm thấy file embeddings.npy. Vui lòng chạy lại pipeline để tạo embeddings.")
        st.stop()
    # Trả về tất cả các dữ liệu đã tải dưới dạng một tuple.
    return df, cosine_sim, topic_labels, embeddings


def calculate_average_vector(article_ids, cosine_sim):
    """
    Chức năng: Tính toán một "vector sở thích" trung bình dựa trên các bài báo người dùng đã đọc.
    Kịch bản: Được gọi bởi hàm get_similar_articles_by_history.
    Logic: Lấy các vector tương ứng với các article_id từ ma trận cosine_sim và tính trung bình cộng.
           Vector kết quả này đại diện cho "khẩu vị" đọc báo của người dùng.
    Tham số:
        - article_ids (list): Danh sách ID các bài báo.
        - cosine_sim (np.array): Ma trận tương đồng.
    Trả về: Một vector numpy trung bình hoặc None nếu không có ID nào hợp lệ.
    """
    # Nếu danh sách ID rỗng, không có gì để tính, trả về None.
    if not article_ids:
        return None
    
    vectors = []
    # Lặp qua các ID bài báo được cung cấp.
    for article_id in article_ids:
        # Đảm bảo ID nằm trong phạm vi của ma trận.
        if article_id < len(cosine_sim):
            # Mỗi hàng trong cosine_sim có thể được coi là một vector đại diện cho quan hệ của bài báo đó với các bài khác.
            vectors.append(cosine_sim[article_id])
    
    # Nếu không tìm thấy vector nào hợp lệ, trả về None.
    if not vectors:
        return None
    
    # Tính vector trung bình theo cột (axis=0).
    return np.mean(vectors, axis=0)


def get_similar_articles_by_history(df, cosine_sim, history_articles, exclude_articles=None):
    """
    Chức năng: Đề xuất các bài viết mới dựa trên lịch sử đọc của người dùng.
    Kịch bản: Được gọi khi người dùng chọn mục "Dựa trên lịch sử đọc".
    Logic:
    1. Lấy 5 bài báo mới nhất từ lịch sử.
    2. Tính "vector sở thích" trung bình từ chúng.
    3. So sánh "vector sở thích" này với tất cả các bài báo khác.
    4. Loại bỏ các bài đã đọc và trả về 10 bài phù hợp nhất.
    Trả về: Một DataFrame chứa các bài báo được đề xuất.
    """
    # Nếu lịch sử đọc rỗng, trả về một DataFrame rỗng.
    if not history_articles:
        return pd.DataFrame()
    
    # Chỉ xem xét 5 bài viết được đọc gần đây nhất để xác định sở thích hiện tại.
    recent_articles = history_articles[:5]
    
    # Tính vector sở thích trung bình.
    avg_vector = calculate_average_vector(recent_articles, cosine_sim)
    if avg_vector is None:
        return pd.DataFrame()
    
    # Tính độ tương đồng cosine giữa vector sở thích và TẤT CẢ các bài báo trong hệ thống.
    # Kết quả là một mảng chứa điểm tương đồng cho mỗi bài báo.
    similarity_scores = cosine_similarity([avg_vector], cosine_sim)[0]
    
    # Loại bỏ các bài viết người dùng đã đọc khỏi danh sách đề xuất.
    if exclude_articles:
        # Tạo một mặt nạ (mask) boolean: True cho các bài chưa đọc, False cho các bài đã đọc.
        mask = ~df.index.isin(exclude_articles)
        # Đặt điểm tương đồng của các bài đã đọc thành -1 để chúng không bao giờ được chọn.
        similarity_scores[~mask] = -1
    
    # Sắp xếp các chỉ số của bài báo theo điểm tương đồng giảm dần và lấy 10 chỉ số đầu tiên.
    top_indices = np.argsort(similarity_scores)[::-1][:10]
    # Lấy ra các bài báo tương ứng từ DataFrame gốc.
    similar_articles = df.iloc[top_indices].copy()
    # Thêm cột điểm tương đồng để có thể hiển thị.
    similar_articles['similarity_score'] = similarity_scores[top_indices]
    
    return similar_articles


def crawl_article_content(url):
    """
    Chức năng: Tải và làm sạch nội dung chi tiết của một bài báo từ URL gốc.
    Kịch bản: Được gọi khi người dùng xem chi tiết một bài viết để có trải nghiệm đọc "sạch".
    Logic:
    1. Gửi yêu cầu HTTP để lấy HTML.
    2. Dùng BeautifulSoup để phân tích HTML.
    3. Dựa vào URL, sử dụng các quy tắc riêng để tìm khối nội dung chính.
    4. Thực hiện một loạt các bước làm sạch: xóa thẻ script/style, xử lý link/ảnh, xóa thuộc tính thừa.
    Trả về: Một chuỗi HTML đã được làm sạch hoặc None nếu thất bại.
    """
    try:
        # Thiết lập headers để giả lập một trình duyệt thật, tránh bị chặn.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive'
        }
        
        # Gửi yêu cầu GET đến URL, với timeout 10 giây để tránh treo ứng dụng.
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'  # Đảm bảo đọc đúng ký tự tiếng Việt.
        soup = BeautifulSoup(response.text, 'html.parser')  # Phân tích HTML.

        # ==== XỬ LÝ ẢNH TRƯỚC KHI TÌM NỘI DUNG ====
        # Lặp qua tất cả các thẻ <img> trong toàn bộ trang.
        for img in soup.find_all('img'):
            # Nhiều trang web dùng 'data-src' cho kỹ thuật lazy-loading. Ta cần chuyển nó thành 'src'.
            if img.has_attr('data-src'):
                img['src'] = img['data-src']
            # Nếu ảnh không có 'src' hoặc 'src' rỗng, nó là thẻ rác, xóa đi.
            if not img.has_attr('src') or not img['src'].strip():
                img.decompose()
            # Thêm thuộc tính loading='lazy' để trình duyệt chỉ tải ảnh khi người dùng cuộn tới.
            img['loading'] = 'lazy'

        # Xóa các thẻ <figure> (thường bao ảnh) nếu chúng rỗng.
        for fig in soup.find_all('figure'):
            if not fig.get_text(strip=True) and not fig.find('img'):
                fig.decompose()

        article = None
        
        # ==== XỬ LÝ THEO TỪNG TRANG BÁO (LOGIC RIÊNG) ====
        # Mỗi trang có cấu trúc HTML khác nhau, ta cần quy tắc riêng cho từng trang.
        if 'vnexpress.net' in url:
            # VnExpress thường chứa nội dung trong <article class="fck_detail">.
            article = soup.find('article', class_='fck_detail')
            if not article:
                # Nếu không có class đó (ví dụ ở trang video, long-form), thử tìm thẻ <article> bất kỳ.
                article = soup.find('article')
            if article:
                # Với VnExpress, đôi khi vẫn còn script/style bên trong, cần xóa trước khi trả về.
                for tag in article.find_all(['script', 'style', 'iframe']):
                    tag.decompose()
                return str(article)

        elif 'tuoitre.vn' in url or 'thanhnien.vn' in url:
            # Tuổi Trẻ và Thanh Niên thường có cấu trúc tương tự.
            article = soup.find('div', class_='detail-content')
            if not article:
                # Nếu không tìm thấy, dùng phương án dự phòng: tìm div có nhiều chữ nhất.
                divs = soup.find_all('div')
                article = max(divs, key=lambda d: len(d.get_text(strip=True))) if divs else None

        elif 'dantri.com.vn' in url:
            # Dân Trí dùng class dt-news__content.
            article = soup.find('div', class_='dt-news__content')
            if not article:
                article = soup.find('article')
            if not article:
                # Phương án dự phòng.
                divs = soup.find_all('div')
                article = max(divs, key=lambda d: len(d.get_text(strip=True))) if divs else None

        # ==== LÀM SẠCH NỘI DUNG (LOGIC CHUNG) ====
        # Nếu đã tìm được khối nội dung `article`.
        if article:
            # 1. Xóa các thẻ không phải là nội dung và có thể gây lỗi.
            for tag in article.find_all(['script', 'style', 'iframe', 'button', 'ins', 'noscript', 'form']):
                tag.decompose()

            # 2. Xử lý các thẻ <a> (liên kết).
            for a in article.find_all('a'):
                a['target'] = '_blank'  # Mở link trong tab mới.
                a['rel'] = 'noopener noreferrer'  # Tăng tính bảo mật.
                if a.has_attr('onclick'):  # Xóa các thuộc tính JavaScript.
                    del a['onclick']
                # Xóa các thuộc tính không cần thiết khác để HTML gọn gàng.
                for attr in ['data-', 'on', 'class', 'id']:
                    for a_attr in list(a.attrs):
                        if a_attr.startswith(attr):
                            del a[a_attr]

            # 3. Xử lý các thẻ <img> (ảnh).
            for img in article.find_all('img'):
                img['loading'] = 'lazy' # Đảm bảo ảnh có lazy-loading.
                # Nếu đường dẫn ảnh là tương đối (ví dụ: /vne/2024/img.jpg), ta cần nối tên miền vào.
                if img.has_attr('src') and not img['src'].startswith(('http://', 'https://')):
                    img['src'] = 'https://vnexpress.net' + img['src'] # Giả định là VnExpress, cần cải tiến thêm.
                # Xóa các thuộc tính không cần thiết.
                for attr in ['data-', 'on', 'class', 'id']:
                    for img_attr in list(img.attrs):
                        if img_attr.startswith(attr):
                            del img[img_attr]

            # 4. Xóa các div rỗng không chứa text hoặc ảnh.
            for div in article.find_all('div'):
                if not div.get_text(strip=True) and not div.find('img'):
                    div.decompose()

            # 5. Bước làm sạch cuối cùng: chỉ giữ lại các thuộc tính an toàn và cần thiết.
            for tag in article.find_all(True):
                allowed_attrs = ['src', 'alt', 'href', 'target', 'rel', 'loading']
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in allowed_attrs}

            # Chuyển đối tượng soup đã làm sạch về dạng chuỗi HTML và trả về.
            return str(article)

        # Nếu không tìm được khối nội dung nào, trả về None.
        return None

    except Exception as e:
        # Nếu có bất kỳ lỗi nào xảy ra, in ra để debug và trả về None.
        print(f"Lỗi khi crawl bài viết từ {url}: {str(e)}")
        return None


def render_detail_view(article_id, df, cosine_sim, topic_labels):
    """
    Chức năng: Hiển thị giao diện chi tiết của một bài viết, bao gồm nội dung và các bài liên quan.
    Kịch bản: Được gọi khi người dùng bấm vào một bài viết từ trang chính.
    """
    try:
        # Lấy thông tin bài viết từ DataFrame bằng ID (index).
        article = df.loc[article_id]
    except KeyError:
        # Nếu không tìm thấy ID (ví dụ sau khi cập nhật tin tức), hiển thị lỗi.
        st.error("Không tìm thấy bài viết.")
        if st.button("⬅️ Quay lại danh sách"):
            st.session_state.current_view = "main"
            st.session_state.current_article_id = None
            st.rerun()
        return
    
    # Cập nhật lịch sử đọc: thêm ID vào set và list trong session_state.
    st.session_state.read_articles.add(article_id)
    if article_id not in st.session_state.reading_history:
        st.session_state.reading_history.insert(0, article_id) # Thêm vào đầu list.
    
    # Chèn một khối CSS tùy chỉnh để định dạng lại trang chi tiết cho dễ đọc.
    st.markdown("""
        <style>
            /* Các quy tắc CSS để làm đẹp nội dung bài viết */
            /* ... (như trong code gốc) ... */
        </style>
    """, unsafe_allow_html=True)

    # Nút quay lại trang chính.
    if st.button("⬅️ Quay lại danh sách"):
        st.session_state.current_view = "main"
        st.session_state.current_article_id = None
        st.rerun() # Chạy lại script để cập nhật view.
    
    # Hiển thị thông tin chính của bài viết.
    st.title(article['title'])
    vn_time = article['published_time'].tz_convert('Asia/Ho_Chi_Minh') # Chuyển sang múi giờ Việt Nam.
    st.caption(f"Nguồn: {article['source_name']} | Xuất bản: {vn_time.strftime('%d-%m-%Y %H:%M')}")
    st.markdown("---")
    
    # Chia layout thành 2 cột: 60% cho nội dung, 40% cho phần đề xuất.
    col1, col2 = st.columns([0.6, 0.4])
    
    # Cột trái: Nội dung bài viết.
    with col1:
        # Hiển thị ảnh đại diện, với ảnh dự phòng nếu URL lỗi.
        if pd.notna(article['image_url']):
            st.markdown(f'<img src="{article["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';" style="width:100%;">', unsafe_allow_html=True)
        else:
            st.markdown('<img src="no-image-png-2.webp" style="width:100%;">', unsafe_allow_html=True)
        
        # Hiển thị spinner trong khi chờ tải nội dung.
        with st.spinner("Đang tải nội dung bài viết..."):
            article_content = crawl_article_content(article['link'])
            if article_content:
                # Nếu crawl thành công, hiển thị nội dung đã được làm sạch.
                article_content = f'<div class="article-content">{article_content}</div>'
                st.markdown(article_content, unsafe_allow_html=True)
            else:
                # Nếu crawl thất bại, hiển thị cảnh báo và nội dung tóm tắt.
                st.warning("Không thể tải nội dung đầy đủ của bài viết. Hiển thị tóm tắt thay thế.")
                summary_raw = article.get('summary_raw', '')
                summary_without_img = re.sub(r'<img[^>]*>', '', summary_raw, flags=re.IGNORECASE)
                st.markdown(f'<div class="article-content">{summary_without_img}</div>', unsafe_allow_html=True)
        
        # Nút liên kết đến bài viết gốc.
        st.link_button("Đọc toàn bộ bài viết trên trang gốc", article['link'])
    
    # Cột phải: Bài viết liên quan.
    with col2:
        st.subheader("Khám phá thêm")
        # Cho phép người dùng chọn kiểu đề xuất. `key` phải là duy nhất cho mỗi bài viết.
        rec_type = st.radio("Hiển thị các bài viết:", ("Có nội dung tương tự", "Trong cùng chủ đề"), key=f"rec_type_{article_id}")
        
        if rec_type == "Có nội dung tương tự":
            # Đề xuất dựa trên độ tương đồng ngữ nghĩa.
            st.markdown("##### Dựa trên phân tích ngữ nghĩa:")
            # Sắp xếp điểm tương đồng của bài hiện tại với các bài khác và lấy 5 bài cao nhất.
            # `[1:6]` để bỏ qua bài đầu tiên (chính nó, điểm là 1.0).
            sim_scores = sorted(list(enumerate(cosine_sim[article_id])), key=lambda x: x[1], reverse=True)[1:6]
            for i, (article_index, score) in enumerate(sim_scores):
                rec_article = df.iloc[article_index]
                with st.container(border=True): # Vẽ khung cho mỗi bài đề xuất.
                    rec_col1, rec_col2 = st.columns([0.25, 0.75])
                    with rec_col1: # Cột ảnh
                        if pd.notna(rec_article['image_url']):
                            st.markdown(f'<img src="{rec_article["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';" style="width:100%;">', unsafe_allow_html=True)
                        else:
                            st.markdown('<img src="no-image-png-2.webp" style="width:100%;">', unsafe_allow_html=True)
                    with rec_col2: # Cột tiêu đề
                        # Tiêu đề là một nút bấm.
                        if st.button(rec_article['title'], key=f"rec_{article_index}"):
                            # Khi bấm, cập nhật lịch sử và ID bài viết hiện tại, rồi chạy lại để chuyển trang.
                            st.session_state.read_articles.add(article_index)
                            if article_index not in st.session_state.reading_history:
                                st.session_state.reading_history.insert(0, article_index)
                            st.session_state.current_article_id = article_index
                            st.rerun()
                        st.caption(f"Nguồn: {rec_article['source_name']} | Độ tương đồng: {score:.2f}")
        else:
            # Đề xuất các bài viết trong cùng chủ đề.
            cluster_id = article['topic_cluster']
            topic_name = topic_labels.get(str(cluster_id), "N/A")
            st.markdown(f"##### Thuộc chủ đề: **{topic_name}**")
            # Lọc DataFrame để lấy các bài cùng chủ đề, khác bài hiện tại.
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
                            # Logic tương tự như trên.
                            st.session_state.read_articles.add(i)
                            if i not in st.session_state.reading_history:
                                st.session_state.reading_history.insert(0, i)
                            st.session_state.current_article_id = i
                            st.rerun()
                        # Tính và hiển thị độ tương đồng để người dùng tham khảo.
                        similarity_score = cosine_sim[article_id][i]
                        st.caption(f"Nguồn: {row['source_name']} | Độ tương đồng: {similarity_score:.2f}")


def render_search_results(query, df, embeddings, sbert_model):
    """
    Chức năng: Xử lý logic tìm kiếm ngữ nghĩa và hiển thị kết quả.
    Kịch bản: Được gọi khi người dùng nhập và gửi một truy vấn tìm kiếm.
    """
    st.header(f"Kết quả tìm kiếm cho: \"{query}\"")
    with st.spinner("Đang phân tích và tìm kiếm..."):
        # Vector hóa câu truy vấn của người dùng.
        query_vector = sbert_model.encode([query])
        # Tính độ tương đồng giữa vector truy vấn và tất cả các vector bài báo.
        similarities = cosine_similarity(query_vector, embeddings)[0]
        # Sắp xếp kết quả.
        sim_scores = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)
        result_indices = [i[0] for i in sim_scores]
        result_df = df.iloc[result_indices].copy()
    # Gọi hàm render_main_grid để hiển thị kết quả tìm kiếm.
    render_main_grid(result_df, f"Kết quả cho: \"{query}\"")


# ==== HÀM HIỂN THỊ CHÍNH (RENDER) ---
def render_main_grid(df, selected_topic_name):
    """
    Chức năng: Hiển thị một lưới các bài viết.
    Kịch bản: Được dùng để hiển thị trang chính, các trang chủ đề, và kết quả tìm kiếm.
    Tham số:
        - df (DataFrame): DataFrame chứa các bài viết cần hiển thị.
        - selected_topic_name (str): Tên chủ đề để hiển thị làm tiêu đề.
    """
    st.header(f"Bảng tin: {selected_topic_name}")
    st.markdown(f"Tìm thấy **{len(df)}** bài viết liên quan.")
    st.markdown("---")
    
    # Thiết lập layout dạng lưới với 3 cột.
    num_columns = 3
    cols = st.columns(num_columns)
    
    if df.empty:
        st.warning("Không có bài viết nào phù hợp với lựa chọn của bạn.")
    else:
        # Lặp qua từng bài viết trong DataFrame. `enumerate` để lấy cả chỉ số `i`.
        for i, (index, row) in enumerate(df.iterrows()):
            # `with cols[i % num_columns]:` là một mẹo để phân phối các bài viết lần lượt vào các cột.
            # i=0 -> col 0, i=1 -> col 1, i=2 -> col 2, i=3 -> col 0, ...
            with cols[i % num_columns]:
                # Tạo HTML cho ảnh, có ảnh dự phòng.
                image_html = ''
                if pd.notna(row["image_url"]):
                    image_html = f'<div class="card-image-container"><img src="{row["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';"></div>'
                else:
                    image_html = '<div class="card-image-container"><img src="no-image-png-2.webp"></div>'
                
                # Tạo HTML cho "card" bài viết bằng f-string. Các class CSS này được định nghĩa trong style.css.
                card_html = f"""<div class="article-card">
                                        {image_html}
                                        <div class="article-content">
                                            <div class="article-title">{row['title']}</div>
                                            <div class="article-source">{row['source_name']}</div>
                                        </div>
                                   </div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Nút "Đọc bài viết" bên dưới mỗi card.
                # `key` phải là duy nhất cho mỗi nút, ta dùng index của bài viết.
                if st.button("Đọc bài viết", key=f"read_{index}"):
                    # Khi bấm nút, lưu ID bài viết và đổi view sang "detail".
                    st.session_state.current_article_id = index
                    st.session_state.current_view = "detail"
                    # Chạy lại script để hiển thị view mới.
                    st.rerun()


# --- LUỒNG CHÍNH CỦA ỨNG DỤNG ---
# Bắt đầu thực thi từ đây.

# Áp dụng file CSS tùy chỉnh.
local_css("style.css")

# Khởi tạo các biến session_state liên quan đến việc cập nhật pipeline.
if 'update_log' not in st.session_state:
    st.session_state.update_log = ""
if 'update_error' not in st.session_state:
    st.session_state.update_error = ""
if 'update_success' not in st.session_state:
    st.session_state.update_success = False

# Các biến này có vẻ bị lặp lại, có thể xóa.
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []
if 'interest_vector' not in st.session_state:
    st.session_state.interest_vector = None
if 'interest_articles' not in st.session_state:
    st.session_state.interest_articles = None

# Tải dữ liệu và mô hình ngay từ đầu. Nhờ có cache, việc này chỉ thực sự chạy lần đầu tiên.
df, cosine_sim, topic_labels, embeddings = load_data_and_embeddings()
sbert_model = get_sbert_model()

# --- Ô TÌM KIẾM Ở ĐẦU TRANG ---
search_col1, search_col2 = st.columns([0.85, 0.15])
with search_col1:
    search_input = st.text_input(
        "Tìm kiếm bài viết (theo ngữ nghĩa, nhập từ khóa hoặc câu hỏi):",
        value=st.session_state.get('search_query', ''), # Giữ lại nội dung tìm kiếm cũ.
        key="search_input",
        placeholder="Nhập nội dung bạn muốn tìm...",
        label_visibility="collapsed" # Ẩn nhãn để tiết kiệm không gian.
    )
with search_col2:
    search_button = st.button("🔍 Tìm kiếm", use_container_width=True)

# Logic xử lý tìm kiếm: kích hoạt khi có nội dung và người dùng bấm nút HOẶC nội dung thay đổi (và họ nhấn Enter).
if search_input and (search_button or search_input != st.session_state.get('search_query', '')):
    st.session_state['search_query'] = search_input
    st.session_state['current_view'] = "search"
    st.rerun() # Chuyển sang view tìm kiếm.

# --- BỘ ĐỊNH TUYẾN VIEW (VIEW ROUTER) ---
# Đây là logic chính điều khiển những gì được hiển thị trên màn hình.

# Kịch bản 1: Người dùng đang ở view TÌM KIẾM.
if st.session_state.get('current_view', 'main') == "search" and st.session_state.get('search_query', ''):
    with st.spinner("Đang phân tích và tìm kiếm..."):
        query_vector = sbert_model.encode([st.session_state['search_query']])
        similarities = cosine_similarity(query_vector, embeddings)[0]
        sim_scores = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)
        result_indices = [i[0] for i in sim_scores]
        result_df = df.iloc[result_indices].copy()
    
    # Hiển thị thông báo và nút quay lại ở sidebar.
    st.sidebar.info("Bạn đang ở trang kết quả tìm kiếm. Chọn danh mục khác hoặc bấm 'Quay lại' để trở về.")
    if st.sidebar.button("⬅️ Quay lại trang chủ", use_container_width=True):
        st.session_state['search_query'] = ''
        st.session_state['current_view'] = "main"
        st.rerun()
    render_main_grid(result_df, f"Kết quả cho: \"{st.session_state['search_query']}\"")
    st.stop() # Dừng script tại đây để không chạy phần `else` bên dưới.

# Kịch bản 2: Người dùng đang ở view CHI TIẾT BÀI VIẾT.
elif st.session_state.current_view == "detail" and st.session_state.current_article_id is not None:
    render_detail_view(st.session_state.current_article_id, df, cosine_sim, topic_labels)

# Kịch bản 3: Người dùng đang ở view TRANG CHÍNH (mặc định).
else:
    # --- GIAO DIỆN THANH BÊN (SIDEBAR) ---
    st.sidebar.title("Tạp chí của bạn")
    st.sidebar.markdown("---")

    # Nút cập nhật tin tức.
    if st.sidebar.button("🔄 Cập nhật tin tức mới", use_container_width=True):
        with st.spinner("⏳ Đang chạy pipeline... Việc này có thể mất vài phút."):
            try:
                # Chạy file pipeline.py như một tiến trình con.
                process = subprocess.run(
                    [sys.executable, 'pipeline.py'], capture_output=True, text=True,
                    encoding='utf-8', errors='ignore' # Ghi lại output và error.
                )
                st.session_state.update_log = process.stdout # Lưu log thành công.
                st.session_state.update_error = process.stderr # Lưu log lỗi.
                st.session_state.update_success = True
                st.cache_data.clear() # QUAN TRỌNG: Xóa cache dữ liệu để lần rerun sau sẽ tải lại file mới.
            except Exception as e:
                st.session_state.update_error = f"Lỗi nghiêm trọng khi chạy pipeline: {e}"
                st.session_state.update_success = False

    # Hiển thị kết quả sau khi cập nhật.
    if st.session_state.update_success:
        st.sidebar.success("✅ Cập nhật hoàn tất!")
        with st.sidebar.expander("Xem chi tiết quá trình"):
            st.code(st.session_state.update_log)
            if st.session_state.update_error:
                st.error("Lỗi từ pipeline:")
                st.code(st.session_state.update_error)
        # Nút này chỉ để người dùng bấm để tải lại trang và thấy dữ liệu mới.
        if st.sidebar.button("Xem tin tức mới", use_container_width=True):
            st.session_state.update_success = False # Reset cờ để thông báo biến mất.
            st.rerun()

    st.sidebar.markdown("---")

    if df is None:
        st.error("Lỗi: Không tìm thấy file dữ liệu. Vui lòng bấm nút 'Cập nhật tin tức mới' ở thanh bên.")
    else:
        # --- PHẦN LỌC THEO CHỦ ĐỀ ---
        st.sidebar.subheader("Khám phá các chủ đề")
        # Tạo danh sách các mục để hiển thị: 3 mục đặc biệt + các chủ đề từ file.
        topic_display_list = ["Dành cho bạn (Tất cả)", "Bài viết đã đọc", "Dựa trên lịch sử đọc"] + [v for k, v in topic_labels.items()]
        
        # Dùng vòng lặp để tạo các nút điều hướng cho từng chủ đề.
        st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        for topic in topic_display_list:
            is_active = (topic == st.session_state.selected_topic)
            active_class = "active" if is_active else ""
            icon = "📖" if topic not in ["Bài viết đã đọc", "Dựa trên lịch sử đọc"] else "👁️" if topic == "Bài viết đã đọc" else "🎯"
            if st.sidebar.button(f"{icon} {topic}", key=f"topic_{topic}", use_container_width=True):
                st.session_state.selected_topic = topic
                st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        st.sidebar.markdown("---")

        # --- PHẦN LỌC THEO NGUỒN TIN ---
        st.sidebar.subheader("Lọc theo nguồn")
        all_sources = sorted(df['source_name'].unique().tolist())
        selected_sources = st.sidebar.multiselect(
            "Chọn một hoặc nhiều nguồn:",
            options=all_sources,
            default=st.session_state.selected_sources
        )
        
        # Nếu lựa chọn nguồn thay đổi, cập nhật session_state và chạy lại.
        if selected_sources != st.session_state.selected_sources:
            st.session_state.selected_sources = selected_sources
            st.rerun()

        # --- LOGIC LỌC VÀ HIỂN THỊ BÀI VIẾT ---
        # Dựa vào chủ đề được chọn, ta tạo ra DataFrame `display_df` tương ứng.
        if st.session_state.selected_topic == "Bài viết đã đọc":
            if st.session_state.read_articles:
                # Sắp xếp các bài đã đọc theo thứ tự mới nhất lên đầu.
                ordered_articles = [article_id for article_id in st.session_state.reading_history if article_id in st.session_state.read_articles]
                display_df = df[df.index.isin(ordered_articles)].copy()
                display_df = display_df.reindex(ordered_articles) # Sắp xếp lại df theo thứ tự của list.
            else:
                display_df = pd.DataFrame()
                st.info("Bạn chưa đọc bài viết nào.")
        elif st.session_state.selected_topic == "Dựa trên lịch sử đọc":
            if len(st.session_state.reading_history) > 0:
                # Gọi hàm đề xuất dựa trên lịch sử.
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
            # Tìm ID của chủ đề đã chọn.
            selected_key_list = [k for k, v in topic_labels.items() if v == st.session_state.selected_topic]
            if selected_key_list:
                # Lọc DataFrame theo `topic_cluster`.
                display_df = df[df['topic_cluster'] == int(selected_key_list[0])].copy()
            else:
                display_df = pd.DataFrame()
        else:
            # Nếu là "Dành cho bạn (Tất cả)", không lọc gì cả.
            display_df = df.copy()

        # Áp dụng bộ lọc nguồn (nếu có) LÊN TRÊN kết quả đã lọc theo chủ đề.
        if st.session_state.selected_sources:
            display_df = display_df[display_df['source_name'].isin(st.session_state.selected_sources)]

        # Sắp xếp và hiển thị kết quả cuối cùng.
        if not display_df.empty:
            # Chỉ sắp xếp theo thời gian nếu không phải đang xem lịch sử đọc (vì lịch sử đã được sắp xếp).
            if st.session_state.selected_topic != "Bài viết đã đọc":
                display_df = display_df.sort_values(by='published_time', ascending=False)
            render_main_grid(display_df, st.session_state.selected_topic)