import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="绿幕合成 Pro",
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
    }
    .main-title { font-size:24px !important; font-weight:600; }
    .section-header { border-left:3px solid #059669; padding-left:8px; font-weight:600; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-title">绿幕自动化合成系统 Pro</div>', unsafe_allow_html=True)
st.caption("一键锁定 · 极速合成 · 进度可见")

# ---------- 全局状态 ----------
if "locked_corners" not in st.session_state:
    st.session_state.locked_corners = None

# ---------- 侧边栏参数 ----------
with st.sidebar:
    st.subheader("🎛️ 合成参数")
    skin = st.slider("肤色保护", 0.3, 0.9, 0.52, 0.02)
    soft = st.slider("边缘软度", 1.0, 15.0, 7.5, 0.5)
    spill = st.slider("去溢色", 0.0, 1.0, 0.9, 0.05)

    st.subheader("⚡ 性能")
    scale_pct = st.select_slider("处理分辨率", options=[50, 75, 100], value=75,
                                 help="降低分辨率可大幅加速合成")
    scale_factor = scale_pct / 100.0

    st.subheader("🔊 音量")
    green_vol = st.slider("绿幕音量", 0, 200, 100)
    game_vol = st.slider("游戏音量", 0, 200, 100)

# ---------- 屏幕检测（预览用） ----------
def detect_screen(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_skin))
    ksize = max(7, int(min(w, h) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid = [c for c in contours if cv2.contourArea(c) > w*h*0.01]
        if valid:
            c = max(valid, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            return order_points(box)
    return None

def order_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ---------- 快速合成一帧（无检测） ----------
def compose_fast(green, game, pts, soft_val, spill_val):
    h, w = green.shape[:2]
    if pts is None:
        return green
    gh, gw = game.shape[:2]
    M, _ = cv2.findHomography(np.float32([[0,0],[gw,0],[gw,gh],[0,gh]]), pts, cv2.RANSAC, 5)
    if M is None:
        M = cv2.getPerspectiveTransform(np.float32([[0,0],[gw,0],[gw,gh],[0,gh]]), pts)
    warped = cv2.warpPerspective(game, M, (w, h))

    hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_skin))

    alpha = 1.0 - mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (int(soft_val)*2+1, int(soft_val)*2+1), 0)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)[:,:,np.newaxis]

    b,g,r = cv2.split(green.astype(np.float32))
    max_g = (r + b) * 0.49
    spill_intensity = np.maximum(0, g - max_g)
    edge_zone = 4 * alpha[:,:,0] * (1 - alpha[:,:,0])
    g_corr = g - spill_intensity * spill_val * edge_zone
    bg_despilled = np.clip(cv2.merge([b, g_corr, r]), 0, 255)

    final = warped * (1 - alpha) + bg_despilled * alpha
    return np.clip(final, 0, 255).astype(np.uint8)

# ---------- 音频工具 ----------
def safe_extract_audio(path):
    try:
        clip = VideoFileClip(path)
        aud = clip.audio
        clip.close()
        return aud
    except:
        return None

def set_volume(audio_clip, vol):
    try:
        return audio_clip.fx.volumex(vol)
    except:
        return audio_clip

# ---------- 主界面 ----------
col1, col2 = st.columns([5,5])
with col1:
    st.markdown('<div class="section-header">1. 素材上传</div>', unsafe_allow_html=True)
    ug = st.file_uploader("绿幕素材", type=["mp4","mov"])
    um = st.file_uploader("游戏录屏", type=["mp4"])

if ug:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(ug.getbuffer())
    st.session_state.green_path = tfile.name
if um:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(um.getbuffer())
    st.session_state.game_path = tfile.name

if "green_path" in st.session_state and "game_path" in st.session_state:
    try:
        cap_g = cv2.VideoCapture(st.session_state.green_path)
        fps = cap_g.get(cv2.CAP_PROP_FPS)
        total_f = int(cap_g.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = total_f / fps if fps else 0
        cap_g.release()
        cap_m = cv2.VideoCapture(st.session_state.game_path)
        m_fps = cap_m.get(cv2.CAP_PROP_FPS)
        m_total = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_m.release()

        if total_f > 0:
            with col2:
                st.markdown('<div class="section-header">2. 预览 & 锁定</div>', unsafe_allow_html=True)
                idx = st.slider("预览帧", 0, total_f-1, int(total_f*0.5))
                cap_g = cv2.VideoCapture(st.session_state.green_path)
                cap_m = cv2.VideoCapture(st.session_state.game_path)
                m_idx = int((idx/fps)*m_fps) % m_total if m_total else 0
                cap_m.set(cv2.CAP_PROP_POS_FRAMES, m_idx)
                cap_g.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret1, fg = cap_g.read()
                ret2, fm = cap_m.read()
                cap_g.release(); cap_m.release()
                if ret1 and ret2:
                    auto_pts = detect_screen(fg)
                    if auto_pts is not None:
                        preview = compose_fast(fg, fm, auto_pts, soft, spill)
                    else:
                        h, w = fg.shape[:2]
                        m = 0.2
                        auto_pts = np.float32([[w*m, h*m], [w*(1-m), h*m], [w*(1-m), h*(1-m)], [w*m, h*(1-m)]])
                        preview = compose_fast(fg, fm, auto_pts, soft, spill)
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"第 {idx} 帧")

                    colB1, colB2 = st.columns(2)
                    with colB1:
                        if st.button("🔒 锁定当前角点"):
                            pts = detect_screen(fg)
                            if pts is not None:
                                st.session_state.locked_corners = pts
                                st.success("角点已锁定！合成速度将大幅提升。")
                            else:
                                st.warning("未检测到屏幕，请手动调整后锁定。")
                    with colB2:
                        if st.button("🔓 解锁"):
                            st.session_state.locked_corners = None
                            st.info("已解锁")
                else:
                    st.warning("帧解码失败")

            with col1:
                st.markdown('<div class="section-header">3. 极速导出</div>', unsafe_allow_html=True)
                if st.button("🔥 开始合成", type="primary", use_container_width=True):
                    if st.session_state.locked_corners is None:
                        st.warning("请先锁定角点！")
                    else:
                        pts = st.session_state.locked_corners
                        cap_g = cv2.VideoCapture(st.session_state.green_path)
                        cap_m = cv2.VideoCapture(st.session_state.game_path)
                        w = int(cap_g.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # 缩放
                        if scale_factor != 1.0:
                            small_w = int(w * scale_factor)
                            small_h = int(h * scale_factor)
                            small_pts = pts * scale_factor
                        else:
                            small_w, small_h = w, h
                            small_pts = pts

                        silent = os.path.join(tempfile.gettempdir(), "silent.mp4")
                        if os.path.exists(silent): os.remove(silent)
                        out = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = None  # 用于估算时间，可选

                        for fi in range(total_f):
                            ret_g, fr_g = cap_g.read()
                            if not ret_g: break
                            m_idx = int((fi/fps)*m_fps) % m_total if m_total else 0
                            cap_m.set(cv2.CAP_PROP_POS_FRAMES, m_idx)
                            ret_m, fr_m = cap_m.read()
                            if not ret_m: fr_m = np.zeros_like(fr_g)

                            if scale_factor != 1.0:
                                fr_g = cv2.resize(fr_g, (small_w, small_h))
                                fr_m = cv2.resize(fr_m, (small_w, small_h))

                            proc = compose_fast(fr_g, fr_m, small_pts, soft, spill)

                            if scale_factor != 1.0:
                                proc = cv2.resize(proc, (w, h))

                            out.write(proc)

                            # 更新进度
                            percent = int((fi + 1) / total_f * 100)
                            progress_bar.progress(percent)
                            status_text.text(f"正在合成... {percent}%")

                        cap_g.release(); cap_m.release(); out.release()
                        status_text.text("合成完成，正在处理音频...")

                        # 等待文件写入
                        while not os.path.exists(silent):
                            pass

                        # 音频（简化处理：优先绿幕音频，若无则游戏音频）
                        green_aud = safe_extract_audio(st.session_state.green_path)
                        game_aud = safe_extract_audio(st.session_state.game_path)
                        mixed = None
                        if green_aud:
                            mixed = set_volume(green_aud, green_vol/100)
                        elif game_aud:
                            # 循环游戏音频到视频长度
                            full_dur = game_aud.duration
                            n = int(dur // full_dur) if full_dur > 0 else 1
                            clips = [game_aud] * n
                            if dur % full_dur > 0:
                                clips.append(game_aud.set_duration(dur % full_dur))
                            mixed = concatenate_audioclips(clips) if len(clips) > 1 else clips[0]
                            mixed = set_volume(mixed, game_vol/100)

                        out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
                        vid = VideoFileClip(silent)
                        if mixed:
                            vid = vid.with_audio(mixed)
                            vid = vid.with_duration(min(vid.duration, mixed.duration))
                        vid.write_videofile(out_path, codec='libx264', audio_codec='aac' if mixed else None, logger=None)
                        vid.close()
                        if mixed: mixed.close()
                        if os.path.exists(silent): os.remove(silent)

                        progress_bar.empty()
                        status_text.empty()
                        st.balloons()
                        st.success("合成完成！")
                        with open(out_path, "rb") as f:
                            st.download_button("📥 下载成品", f, file_name="合成视频.mp4", mime="video/mp4")
    except Exception as e:
        st.error(f"异常：{e}")
else:
    with col2:
        st.info("👈 请上传绿幕素材和游戏视频")
