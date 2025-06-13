import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import sys
import subprocess
from urllib.parse import urlparse, quote, unquote

# --- CẤU HÌNH TRANG VÀ CSS ---
st.set_page_config(page_title="Tạp chí của bạn", page_icon="📖", layout="wide")

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
                    try:
                        # Thêm proxy cho hình ảnh để tránh CORS
                        proxy_url = f"https://images.weserv.nl/?url={row['image_url']}"
                        image_html = f'<div class="card-image-container"><img src="{proxy_url}" onerror="this.onerror=null; this.src=\'https://via.placeholder.com/400x225?text=Không+có+hình+ảnh\';"></div>'
                    except:
                        image_html = '<div class="card-image-container" style="background-color:#f0f2f6;"><img src="https://via.placeholder.com/400x225?text=Không+có+hình+ảnh"></div>'
                else:
                    image_html = '<div class="card-image-container" style="background-color:#f0f2f6;"><img src="https://via.placeholder.com/400x225?text=Không+có+hình+ảnh"></div>'
                
                # Sử dụng cột 'source_name' đã tạo
                source_name = row['source_name']
                card_html = f"""<a href="?article_id={index}" target="_self" class="article-card">
                                    {image_html}
                                    <div class="article-content">
                                        <div class="article-title">{row['title']}</div>
                                        <div class="article-source">{source_name}</div>
                                    </div>
                               </a>"""
                st.markdown(card_html, unsafe_allow_html=True)

def render_detail_view(article_id, df, cosine_sim, topic_labels):
    try:
        article = df.loc[article_id]
    except KeyError:
        st.error("Không tìm thấy bài viết.")
        st.markdown('<a href="javascript:history.back()" target="_self">⬅️ Quay lại trang chính</a>', unsafe_allow_html=True)
        return
    
    # Sử dụng JavaScript để quay lại trang trước
    st.markdown('<a href="javascript:history.back()" target="_self">⬅️ Quay lại danh sách</a>', unsafe_allow_html=True)
    st.title(article['title'])
    # Hiển thị thời gian theo múi giờ Việt Nam
    vn_time = article['published_time'].tz_convert('Asia/Ho_Chi_Minh')
    st.caption(f"Nguồn: {article['source_name']} | Xuất bản: {vn_time.strftime('%d-%m-%Y %H:%M')}")
    st.markdown("---")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if pd.notna(article['image_url']):
            try:
                # Thêm proxy cho hình ảnh để tránh CORS
                proxy_url = f"https://images.weserv.nl/?url={article['image_url']}"
                st.image(proxy_url, use_column_width=True, on_click=lambda: None)
            except:
                st.image("https://via.placeholder.com/800x450?text=Không+có+hình+ảnh", use_column_width=True)
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
                        if pd.notna(rec_article['image_url']): st.image(rec_article['image_url'])
                    with rec_col2:
                        st.markdown(f"<a href='?article_id={article_index}' target='_self'>{rec_article['title']}</a>", unsafe_allow_html=True)
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
                        if pd.notna(row['image_url']): st.image(row['image_url'])
                    with rec_col2:
                        st.markdown(f"<a href='?article_id={i}' target='_self'>{row['title']}</a>", unsafe_allow_html=True)
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

df, cosine_sim, topic_labels = load_data()

# --- GIAO DIỆN THANH BÊN ---
# st.sidebar.image("https://static.vecteezy.com/system/resources/previews/023/388/587/original/paper-icon-vector.jpg", width=100)
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
    topic_display_list = ["Dành cho bạn (Tất cả)"] + [v for k, v in topic_labels.items()]
    query_params = st.query_params
    selected_topic_display = unquote(query_params.get("topic", topic_display_list[0]))
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    for topic in topic_display_list:
        is_active = (topic == selected_topic_display)
        active_class = "active" if is_active else ""
        topic_url = quote(topic)
        link = f'/?topic={topic_url}'
        icon = "📖"
        st.sidebar.markdown(f'<a href="{link}" target="_self" class="sidebar-item {active_class}">{icon} &nbsp; {topic}</a>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # --- BỔ SUNG: PHẦN LỌC THEO NGUỒN ---
    st.sidebar.subheader("Lọc theo nguồn")
    all_sources = sorted(df['source_name'].unique().tolist())
    selected_sources = st.sidebar.multiselect(
        "Chọn một hoặc nhiều nguồn:",
        options=all_sources,
        default=[] # Mặc định không chọn nguồn nào
    )
    # --- KẾT THÚC PHẦN BỔ SUNG ---

    # --- HIỂN THỊ VIEW TƯƠNG ỨNG ---
    if "article_id" in query_params:
        try:
            article_id = int(query_params.get("article_id"))
            render_detail_view(article_id, df, cosine_sim, topic_labels)
        except (ValueError, IndexError):
            st.error("ID bài viết không hợp lệ.")
            st.markdown('<a href="/" target="_self">⬅️ Quay lại trang chính</a>', unsafe_allow_html=True)
    else:
        # Lọc theo chủ đề
        if selected_topic_display != "Dành cho bạn (Tất cả)":
            selected_key_list = [k for k, v in topic_labels.items() if v == selected_topic_display]
            if selected_key_list:
                display_df = df[df['topic_cluster'] == int(selected_key_list[0])].copy()
            else:
                display_df = pd.DataFrame()
        else:
            display_df = df.copy()

        # BỔ SUNG: Áp dụng bộ lọc nguồn
        if selected_sources: # Nếu người dùng đã chọn ít nhất một nguồn
            display_df = display_df[display_df['source_name'].isin(selected_sources)]

        # Sắp xếp và hiển thị
        display_df = display_df.sort_values(by='published_time', ascending=False)
        render_main_grid(display_df, selected_topic_display)