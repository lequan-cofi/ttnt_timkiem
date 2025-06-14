import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import sys
import subprocess
from urllib.parse import urlparse, quote, unquote
from sklearn.metrics.pairwise import cosine_similarity

# --- CẤU HÌNH TRANG VÀ CSS ---
st.set_page_config(page_title="Tạp chí của bạn", page_icon="📖", layout="wide")

# Initialize session state
if 'read_articles' not in st.session_state:
    st.session_state.read_articles = set()
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []  # List to store all read articles
if 'current_view' not in st.session_state:
    st.session_state.current_view = "main"  # "main" or "detail"
if 'current_article_id' not in st.session_state:
    st.session_state.current_article_id = None
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "Dành cho bạn (Tất cả)"
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []

def local_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{file_name}'.")

# --- HÀM TẢI DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_articles_for_app.csv')
        cosine_sim = np.load('cosine_similarity_matrix.npy')
        with open('topic_labels.json', 'r', encoding='utf-8') as f:
            topic_labels = json.load(f)
        df['published_time'] = pd.to_datetime(df['published_time'])
        # Tạo cột 'source_name' để lọc dễ dàng hơn
        df['source_name'] = df['link'].apply(get_source_name)
        return df, cosine_sim, topic_labels
    except FileNotFoundError:
        return None, None, None

def get_source_name(link):
    try:
        domain = urlparse(link).netloc
        if domain.startswith('www.'): domain = domain[4:]
        return domain.split('.')[0].capitalize()
    except:
        return "N/A"

# --- CÁC HÀM HIỂN THỊ (RENDER) ---
def render_main_grid(df, selected_topic_name):
    st.header(f"Bảng tin: {selected_topic_name}")
    st.markdown(f"Tìm thấy **{len(df)}** bài viết liên quan.")
    st.markdown("---")
    num_columns = 3
    cols = st.columns(num_columns)
    if df.empty:
        st.warning("Không có bài viết nào phù hợp với lựa chọn của bạn.")
    else:
        for i, (index, row) in enumerate(df.iterrows()):
            with cols[i % num_columns]:
                # Xử lý hình ảnh với placeholder
                image_html = ''
                if pd.notna(row["image_url"]):
                    image_html = f'<div class="card-image-container"><img src="{row["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';"></div>'
                else:
                    image_html = '<div class="card-image-container"><img src="no-image-png-2.webp"></div>'
                
                # Sử dụng cột 'source_name' đã tạo
                source_name = row['source_name']
                card_html = f"""<div class="article-card">
                                {image_html}
                                <div class="article-content">
                                    <div class="article-title">{row['title']}</div>
                                    <div class="article-source">{source_name}</div>
                                </div>
                           </div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("Đọc bài viết", key=f"read_{index}"):
                    st.session_state.current_article_id = index
                    st.session_state.current_view = "detail"
                    st.rerun()

def calculate_interest_vector(df, cosine_sim, article_ids):
    """Calculate interest vector from reading history."""
    if not article_ids:
        return None
    
    # Get vectors for articles in history
    vectors = []
    for article_id in article_ids:
        if article_id < len(cosine_sim):
            vectors.append(cosine_sim[article_id])
    
    if not vectors:
        return None
    
    # Calculate average vector
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector

def update_interest_vector(df, cosine_sim, article_id):
    """Update interest vector when new article is read."""
    if article_id not in st.session_state.reading_history:
        # Add to reading history (keep last 5)
        st.session_state.reading_history.insert(0, article_id)
        st.session_state.reading_history = st.session_state.reading_history[:5]
        
        # Calculate new interest vector
        st.session_state.interest_vector = calculate_interest_vector(
            df, cosine_sim, st.session_state.reading_history
        )
        
        # Update interest articles if vector exists
        if st.session_state.interest_vector is not None:
            similarity_scores = cosine_similarity([st.session_state.interest_vector], cosine_sim)[0]
            # Create mask to exclude read articles
            mask = ~df.index.isin(st.session_state.reading_history)
            similarity_scores[~mask] = -1  # Set similarity to -1 for read articles
            # Get top similar articles
            top_indices = np.argsort(similarity_scores)[::-1][:10]
            st.session_state.interest_articles = df.iloc[top_indices].copy()
            st.session_state.interest_articles['similarity_score'] = similarity_scores[top_indices]

def get_interest_articles():
    """Get articles based on user's interests."""
    if st.session_state.interest_articles is not None:
        return st.session_state.interest_articles
    return pd.DataFrame()

def calculate_average_vector(article_ids, cosine_sim):
    """Calculate average vector from last 5 articles."""
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
    """Get similar articles based on reading history."""
    if not history_articles:
        return pd.DataFrame()
    
    # Chỉ lấy 5 bài viết mới đọc gần nhất
    recent_articles = history_articles[:5]
    
    avg_vector = calculate_average_vector(recent_articles, cosine_sim)
    if avg_vector is None:
        return pd.DataFrame()
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity([avg_vector], cosine_sim)[0]
    
    # Create mask to exclude read articles
    if exclude_articles:
        mask = ~df.index.isin(exclude_articles)
        similarity_scores[~mask] = -1
    
    # Get top similar articles
    top_indices = np.argsort(similarity_scores)[::-1][:10]
    similar_articles = df.iloc[top_indices].copy()
    similar_articles['similarity_score'] = similarity_scores[top_indices]
    
    return similar_articles

def render_detail_view(article_id, df, cosine_sim, topic_labels):
    try:
        article = df.loc[article_id]
    except KeyError:
        st.error("Không tìm thấy bài viết.")
        if st.button("⬅️ Quay lại danh sách"):
            st.session_state.current_view = "main"
            st.session_state.current_article_id = None
            st.rerun()
        return
    
    # Add article to read articles set and update reading history
    st.session_state.read_articles.add(article_id)
    if article_id not in st.session_state.reading_history:
        st.session_state.reading_history.insert(0, article_id)  # Add to history without limit
    
    if st.button("⬅️ Quay lại danh sách"):
        st.session_state.current_view = "main"
        st.session_state.current_article_id = None
        st.rerun()
    
    st.title(article['title'])
    vn_time = article['published_time'].tz_convert('Asia/Ho_Chi_Minh')
    st.caption(f"Nguồn: {article['source_name']} | Xuất bản: {vn_time.strftime('%d-%m-%Y %H:%M')}")
    st.markdown("---")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if pd.notna(article['image_url']):
            st.markdown(f'<img src="{article["image_url"]}" onerror="this.onerror=null; this.src=\'no-image-png-2.webp\';" style="width:100%;">', unsafe_allow_html=True)
        else:
            st.markdown('<img src="no-image-png-2.webp" style="width:100%;">', unsafe_allow_html=True)
        st.subheader("Tóm tắt")
        summary_raw = article.get('summary_raw', '')
        summary_without_img = re.sub(r'<img[^>]*>', '', summary_raw, flags=re.IGNORECASE)
        st.markdown(summary_without_img, unsafe_allow_html=True)
        st.link_button("Đọc toàn bộ bài viết trên trang gốc", article['link'])
    with col2:
        st.subheader("Khám phá thêm")
        rec_type = st.radio("Hiển thị các bài viết:", ("Có nội dung tương tự", "Trong cùng chủ đề"), key=f"rec_type_{article_id}")
        
        if rec_type == "Có nội dung tương tự":
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
                            st.session_state.current_article_id = article_index
                            st.rerun()
                        st.caption(f"Độ tương đồng: {score:.2f}")
        else: # Cùng chủ đề
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
                            st.session_state.current_article_id = i
                            st.rerun()
                        st.caption(f"Nguồn: {row['source_name']}")

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

df, cosine_sim, topic_labels = load_data()

# --- GIAO DIỆN THANH BÊN ---
st.sidebar.title("Tạp chí của bạn")
st.sidebar.markdown("---")

# Nút cập nhật
if st.sidebar.button("🔄 Cập nhật tin tức mới", use_container_width=True):
    with st.spinner("⏳ Đang chạy pipeline... Việc này có thể mất vài phút."):
        try:
            process = subprocess.run(
                [sys.executable, 'pipeline.py'], capture_output=True, text=True,
                encoding='utf-8', errors='ignore'
            )
            st.session_state.update_log = process.stdout
            st.session_state.update_error = process.stderr
            st.session_state.update_success = True
            st.cache_data.clear() # Xóa cache để chuẩn bị tải lại
        except Exception as e:
            st.session_state.update_error = f"Lỗi nghiêm trọng khi chạy pipeline: {e}"
            st.session_state.update_success = False

# Hiển thị kết quả cập nhật và nút tải lại
if st.session_state.update_success:
    st.sidebar.success("✅ Cập nhật hoàn tất!")
    with st.sidebar.expander("Xem chi tiết quá trình"):
        st.code(st.session_state.update_log)
        if st.session_state.update_error:
            st.error("Lỗi từ pipeline:")
            st.code(st.session_state.update_error)
    if st.sidebar.button("Xem tin tức mới", use_container_width=True):
        st.session_state.update_success = False # Reset cờ
        st.rerun()

st.sidebar.markdown("---")

if df is None:
    st.error("Lỗi: Không tìm thấy file dữ liệu. Vui lòng bấm nút 'Cập nhật tin tức mới' ở thanh bên.")
else:
    # --- PHẦN LỌC THEO CHỦ ĐỀ ---
    st.sidebar.subheader("Khám phá các chủ đề")
    topic_display_list = ["Dành cho bạn (Tất cả)", "Bài viết đã đọc", "Dựa trên lịch sử đọc"] + [v for k, v in topic_labels.items()]
    
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    for topic in topic_display_list:
        is_active = (topic == st.session_state.selected_topic)
        active_class = "active" if is_active else ""
        icon = "📖" if topic != "Bài viết đã đọc" and topic != "Dựa trên lịch sử đọc" else "👁️" if topic == "Bài viết đã đọc" else "🎯"
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
    if st.session_state.current_view == "detail" and st.session_state.current_article_id is not None:
        render_detail_view(st.session_state.current_article_id, df, cosine_sim, topic_labels)
    else:
        # Lọc theo chủ đề
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
                display_df = display_df.sort_values(by='published_time', ascending=False)
        render_main_grid(display_df, st.session_state.selected_topic)