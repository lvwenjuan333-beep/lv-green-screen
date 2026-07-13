import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageSequenceClip
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="绿幕自动化合成系统 Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

st.markdown("""<style>
    footer {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}
    button[data-testid="stSidebarCollapseButton"] {
        position: fixed !important; top: 12px !important; left: 12px !important;
        z-index: 99999 !important; background-color: #059669 !important; color: white !important;
        border-radius: 50% !important; width: 34px !important; height: 34px !important;
        opacity: 1 !important; visibility: visible !important;
        box-shadow: 0 2px 12px rgba(5,150,105,0.5) !important; border: none !important;
        transition: transform 0.2s ease;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        transform: scale(1.15); box-shadow: 0 4px 16px rgba(5,150,105,0.7) !important;
    }
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #fcfcfd;
    }
    .main-title { font-size:24px !important; font-weight:600 !important; color:#0f172a !important; margin-bottom:4px !important; letter-spacing:-0.5px; }
    .sub-title { font-size:13px !important; font-weight:400 !important; color:#64748b !important; margin-bottom:24px !important; }
    .section-header { font-size:14px !important; font-weight:600 !important; color:#059669; border-left:3px solid #059669; padding-left:8px !important; margin-bottom:16px !important; }
    [data-testid="stSidebar"] { background-color:#f8fafc !important; border-right:1px solid #e2e8f0 !important; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-title">绿幕自动化合成系统 Pro (高抗噪完全体)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">大角度抗扭曲体系 ── 针对手部绿光反光穿帮深度优化，大倾角运动自适应追踪</div>', unsafe_allow_html=True)

# 基础四角对齐状态初始化
param_list = ["tl_x","tl_y","tr_x","tr_y","br_x","br_y","bl_x","bl_y"]
for p in param_list:
    if p not in st.session_state: st.session_state[p]=0

def update_via_slider(k): st.session_state[k] = st.session_state[k+"_slide"]
def update_via_num(k): st.session_state[k+"_slide"] = st.session_state[k]

st.sidebar.markdown('<p style="font-size:13px; font-weight:600; color:#0f172a; margin-bottom:10px;">🎛️ 1. 影视级抗噪参数调整</p>', unsafe_allow_html=True)
skin_protect = st.sidebar.slider("肤色保护强度（防止手指残缺）", 0.3, 0.9, 0.52, 0.02)
green_thresh = st.sidebar.number_input("核心绿幕剔除强度", 10, 100, 36, 1)
soft_falloff = st.sidebar.slider("边缘半透明交融软度", 1.0, 15.0, 7.5, 0.5)
spill_suppress = st.sidebar.slider("手指泛绿洗白强度", 0.0, 1.0, 0.90, 0.05)
track_stability = st.sidebar.number_input("大角度运动时序抗抖窗口", 1, 30, 6, 1)

st.sidebar.markdown("<hr style='margin:16px 0; border-color:#edf2f7;'/>", unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;">📐 2. 边界透视形变微调</p>', unsafe_allow_html=True)

def dual_control_widget(label, key_name):
    st.sidebar.markdown(f'<p style="font-size:11px; color:#475569; margin-bottom:1px; margin-top:6px;">{label}</p>', unsafe_allow_html=True)
    if key_name + "_slide" not in st.session_state: st.session_state[key_name + "_slide"] = 0
    st.sidebar.number_input(f"num_{key_name}", -200, 200, key=key_name, on_change=update_via_num, args=(key_name,), label_visibility="collapsed")
    st.sidebar.slider(f"slide_{key_name}", -200, 200, key=key_name + "_slide", on_change=update_via_slider, args=(key_name,), label_visibility="collapsed")

for label, key in [("左上角 X", "tl_x"), ("左上角 Y", "tl_y"), ("右上角 X", "tr_x"), ("右上角 Y", "tr_y"),
                  ("右下角 X", "br_x"), ("右下角 Y", "br_y"), ("左下角 X", "bl_x"), ("左下角 Y", "bl_y")]:
    dual_control_widget(label, key)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ---------- 核心合成函数（不变） ----------
def process_core_frame_master(frame_green, frame_game, last_valid_pts, history_pts):
    width, height = frame_green.shape[1], frame_green.shape[0]
    bg_float = frame_green.astype(np.float32)
    
    hsv = cv2.cvtColor(frame_green, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green).astype(np.float32) / 255.0
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin).astype(np.float32) / 255.0
    
    mask_green_safe = np.clip(mask_green - mask_skin * skin_protect, 0, 1)
    alpha = 1.0 - mask_green_safe
    blur_size = int(soft_falloff) * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    
    mask_binary = (alpha < 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_pts = None
    if contours:
        valid = [c for c in contours if cv2.contourArea(c) > 3000]
        if valid:
            c = max(valid, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            detected_pts = order_points(np.array(box, dtype=np.float32))

    if detected_pts is not None:
        if last_valid_pts is not None:
            move_dist = np.mean(np.linalg.norm(detected_pts - last_valid_pts, axis=1))
            if move_dist > 220: history_pts = []
        history_pts.append(detected_pts)
        if len(history_pts) > track_stability: history_pts.pop(0)
        smoothed_pts = np.mean(history_pts, axis=0).astype(np.float32)
        last_valid_pts = smoothed_pts
    else:
        smoothed_pts = last_valid_pts

    if smoothed_pts is not None:
        target_pts = smoothed_pts.copy()
        target_pts[0][0] += st.session_state.tl_x; target_pts[0][1] += st.session_state.tl_y
        target_pts[1][0] += st.session_state.tr_x; target_pts[1][1] += st.session_state.tr_y
        target_pts[2][0] += st.session_state.br_x; target_pts[2][1] += st.session_state.br_y
        target_pts[3][0] += st.session_state.bl_x; target_pts[3][1] += st.session_state.bl_y
        
        gh, gw, _ = frame_game.shape
        pts_game = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]])
        M, _ = cv2.findHomography(pts_game, target_pts, cv2.RANSAC, 5.0)
        if M is None:
            M = cv2.getPerspectiveTransform(pts_game, target_pts)
        warped_game = cv2.warpPerspective(frame_game, M, (width, height)).astype(np.float32)
        
        b_ch, g_ch, r_ch = cv2.split(frame_green.astype(np.float32))
        max_g = (r_ch + b_ch) * 0.49
        spill_intensity = np.maximum(0.0, g_ch - max_g)
        edge_zone = 4.0 * alpha * (1.0 - alpha)
        g_corrected = g_ch - (spill_intensity * spill_suppress * edge_zone)
        bg_despilled = np.clip(cv2.merge([b_ch, g_corrected, r_ch]), 0, 255)
        
        alpha_3d = cv2.merge([alpha] * 3)
        final_frame = warped_game * (1.0 - alpha_3d) + bg_despilled * alpha_3d
        return np.clip(final_frame, 0, 255).astype(np.uint8), last_valid_pts, history_pts
        
    return frame_green, last_valid_pts, history_pts

# ---------- 主界面 ----------
col1, col2 = st.columns([5, 5], gap="large")
with col1:
    st.markdown('<div class="section-header">1. 素材队列输入</div>', unsafe_allow_html=True)
    uploaded_green = st.file_uploader("手持高动态实拍绿幕素材 (MP4/MOV)", type=["mp4", "mov"])
    uploaded_game = st.file_uploader("游戏玩法高帧率录屏 (MP4)", type=["mp4"])

if uploaded_green is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_green.getbuffer())
    st.session_state.green_cache_path = tfile.name
if uploaded_game is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_game.getbuffer())
    st.session_state.game_cache_path = tfile.name

if "green_cache_path" in st.session_state and "game_cache_path" in st.session_state:
    try:
        cap_info = cv2.VideoCapture(st.session_state.green_cache_path)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_info.release()
        if total_frames > 0:
            with col2:
                st.markdown('<div class="section-header">2. 极限晃动与反光动态监视器</div>', unsafe_allow_html=True)
                idx = st.slider("🎞 检查动作剧烈帧的抠像表现", 0, total_frames - 1, int(total_frames * 0.5), label_visibility="collapsed")
                cap_green = cv2.VideoCapture(st.session_state.green_cache_path)
                cap_game  = cv2.VideoCapture(st.session_state.game_cache_path)
                cap_green.set(cv2.CAP_PROP_POS_FRAMES, idx)
                game_total = int(cap_game.get(cv2.CAP_PROP_FRAME_COUNT))
                if game_total <= 0: game_total = 1
                cap_game.set(cv2.CAP_PROP_POS_FRAMES, idx % game_total)
                ret1, f_green = cap_green.read()
                ret2, f_game  = cap_game.read()
                cap_green.release(); cap_game.release()
                if ret1 and ret2:
                    preview, _, _ = process_core_frame_master(f_green, f_game, None, [])
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width=470, caption=f"第 {idx} 帧自适应融合表现")
                else:
                    st.warning("⚠️ 帧解码中断。")
            with col1:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">3. 无损级渲染导出管线</div>', unsafe_allow_html=True)
                if st.button("🔥 开始全片大卡高规格合成", type="primary", use_container_width=True):
                    with st.spinner("影视级抗噪自适应算法深度处理中，请稍候..."):
                        cap_green = cv2.VideoCapture(st.session_state.green_cache_path)
                        cap_game  = cv2.VideoCapture(st.session_state.game_cache_path)
                        fps = cap_green.get(cv2.CAP_PROP_FPS)
                        width = int(cap_green.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height= int(cap_green.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        last_valid_pts = None
                        history_pts = []
                        frames = []
                        while cap_green.isOpened():
                            ret_g, frame_g = cap_green.read()
                            if not ret_g: break
                            ret_m, frame_m = cap_game.read()
                            if not ret_m:
                                cap_game.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret_m, frame_m = cap_game.read()
                            if frame_m is None: frame_m = np.zeros_like(frame_g)
                            processed, last_valid_pts, history_pts = process_core_frame_master(frame_g, frame_m, last_valid_pts, history_pts)
                            # 转换为RGB (moviepy需要)
                            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                            frames.append(processed_rgb)
                        cap_green.release(); cap_game.release()
                        
                        # 使用帧序列直接生成视频（无需临时文件）
                        video_clip = ImageSequenceClip(frames, fps=fps)
                        
                        # 音频处理
                        audio = None
                        game_clip = None
                        try:
                            game_clip = VideoFileClip(st.session_state.game_cache_path)
                            if game_clip.audio is not None:
                                audio = game_clip.audio
                        except Exception as e: pass
                        
                        output_final_path = os.path.join(tempfile.gettempdir(), "final_with_audio.mp4")
                        if os.path.exists(output_final_path): os.remove(output_final_path)
                        
                        if audio is not None:
                            video_clip = video_clip.with_audio(audio)
                            video_clip = video_clip.with_duration(min(video_clip.duration, audio.duration))
                        
                        video_clip.write_videofile(
                            output_final_path,
                            codec='libx264',
                            audio_codec='aac' if audio is not None else None,
                            logger=None
                        )
                        video_clip.close()
                        if audio is not None: audio.close()
                        if game_clip is not None: game_clip.close()
                        
                        st.balloons()
                        st.success("🎉 自动化抗抖完全体全片合成完毕！")
                        with open(output_final_path, "rb") as file:
                            st.download_button("📥 导出全自动高品质成片", file,
                                file_name="智能抗噪大卡完全体.mp4", mime="video/mp4", use_container_width=True)
    except Exception as e:
        st.error(f"视频处理异常：{e}")
else:
    with col2:
        st.info("👆 请在左侧同时上传素材，新系统具备大角度抗噪防御力，无需频繁调参。")
