import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from io import BytesIO
import zipfile
import time
import streamlit as st
import gdown

# ===============================================================
# Page config & header
# ===============================================================
st.set_page_config(page_title="Cell Nuclei Segmentation", layout="wide")
st.markdown("<h1 style='text-align:center; font-size:32px; font-weight:bold;'> Cell Nuclei Segmentation </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload one or more microscopy images to visualize the predicted nuclei masks.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ===============================================================
# 0) TransUNetCBA (ResNet50 encoder + ViT + Conv-Blur-Attention decoder)
#    Architecture copied from the training notebook. The .h5 file holds the weights
#    (legacy Keras format), so the model is rebuilt here and the weights loaded into it.
# ===============================================================
TU_IMG_SIZE = 512

class ConvBlurAttention(layers.Layer):
    def __init__(self, filters, kernel_size=3, reduction=16, activation=tf.nn.leaky_relu):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.blur = layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False,
                                           depthwise_initializer=tf.keras.initializers.Constant(1/9))
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // reduction, activation="relu")
        self.dense2 = layers.Dense(filters, activation="sigmoid")
        self.spatial_attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid")
        self.activation = activation

    def call(self, x):
        conv_out = self.bn(self.conv(x))
        blur_out = self.blur(conv_out)
        se = self.dense2(self.dense1(self.global_pool(blur_out)))
        se = tf.reshape(se, [-1, 1, 1, se.shape[-1]])
        ch_attn = conv_out * se
        sp_attn = ch_attn * self.spatial_attn(ch_attn)
        return self.activation(sp_attn * 0.7 + blur_out * 0.3)

class PatchEmbedding(layers.Layer):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.proj = layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding="valid")
        self.flatten = layers.Reshape((-1, embed_dim))
    def call(self, x):
        return self.flatten(self.proj(x))

class PositionalEncoding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = self.add_weight(name="pos_embed", shape=(1, num_patches, embed_dim),
                                         initializer="random_normal")
    def call(self, x):
        return x + self.pos_embed

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=192, num_heads=12, mlp_dim=768, dropout=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])
    def call(self, x):
        attn_output = self.attn(self.norm1(x), self.norm1(x))
        x = x + self.drop1(attn_output)
        return x + self.mlp(self.norm2(x))

def build_transformer_encoder(x, num_layers=4, embed_dim=192, num_heads=12, mlp_dim=768):
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim)(x)
    return x

def build_transunetcba(input_shape=(512, 512, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=inputs)
    skip1 = base_model.get_layer("conv1_relu").output
    skip2 = base_model.get_layer("conv2_block3_out").output
    skip3 = base_model.get_layer("conv3_block4_out").output
    cnn_feature = base_model.get_layer("conv4_block6_out").output

    patches = PatchEmbedding(embed_dim=192)(cnn_feature)
    num_patches = patches.shape[1]
    patches = PositionalEncoding(num_patches, 192)(patches)
    transformer_out = build_transformer_encoder(patches)

    h, w = cnn_feature.shape[1], cnn_feature.shape[2]
    x = layers.Reshape((h, w, 192))(transformer_out)

    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip3])
    x = ConvBlurAttention(512)(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip2])
    x = ConvBlurAttention(256)(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip1])
    x = ConvBlurAttention(128)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = ConvBlurAttention(64)(x)
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", dtype="float32")(x)
    return models.Model(inputs, outputs, name="TransUNetCBA")

# ===============================================================
# 0b) NuSegFormer (EfficientNetV2-S encoder + gated-attention / ReGLU ViT bottleneck)
#     Architecture copied from the NuSegFormer training notebook. Only the weights are stored
#     in the .h5 file, so the model is rebuilt here and the weights loaded into it.
# ===============================================================
VIT_IMG_SIZE = 512
VIT_EMBED_DIM = 256
VIT_NUM_HEADS = 8
VIT_MLP_DIM = 512
VIT_NUM_LAYERS = 4
VIT_DROPOUT = 0.1


def vit_conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(VIT_DROPOUT)(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def vit_upconv_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    return vit_conv_block(x, filters)

class AddPositionalEmbedding(layers.Layer):
    def __init__(self, n_patches, dim, **kwargs):
        super().__init__(**kwargs)
        self.n_patches = n_patches
        self.dim = dim
    def build(self, input_shape):
        self.pos_emb = self.add_weight(name="pos_emb", shape=(1, self.n_patches, self.dim),
                                       initializer="random_normal", trainable=True)
        super().build(input_shape)
    def call(self, x):
        return x + tf.cast(self.pos_emb, dtype=x.dtype)

class GatedMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, name_suffix="", **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = num_heads * key_dim
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f"mha_{name_suffix}")
        def _gate(tag):
            return layers.Dense(self.embed_dim, activation="sigmoid", name=f"{tag}_gate_{name_suffix}")
        self.query_gate = _gate("query"); self.key_gate = _gate("key"); self.value_gate = _gate("value")
        self.sdpa_gate = _gate("sdpa"); self.dense_gate = _gate("dense")
        self.query_mul = layers.Multiply(name=f"qmul_{name_suffix}")
        self.key_mul = layers.Multiply(name=f"kmul_{name_suffix}")
        self.value_mul = layers.Multiply(name=f"vmul_{name_suffix}")
        self.sdpa_mul = layers.Multiply(name=f"smul_{name_suffix}")
        self.dense_mul = layers.Multiply(name=f"dmul_{name_suffix}")
    def call(self, x, training=None):
        orig = x
        q = self.query_mul([x, self.query_gate(x)])
        k = self.key_mul([x, self.key_gate(x)])
        v = self.value_mul([x, self.value_gate(x)])
        out = self.attention(query=q, key=k, value=v, training=training)
        out = self.sdpa_mul([out, self.sdpa_gate(orig)])
        out = self.dense_mul([out, self.dense_gate(orig)])
        return out

class ReGLUFeedForward(layers.Layer):
    def __init__(self, embed_dim, mlp_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.up_proj = layers.Dense(mlp_dim * 2, use_bias=True)
        self.down_proj = layers.Dense(embed_dim, use_bias=True)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
    def call(self, x, training=None):
        value, gate = tf.split(self.up_proj(x), 2, axis=-1)
        x = value * tf.nn.relu(gate)
        x = self.drop1(x, training=training)
        x = self.down_proj(x)
        return self.drop2(x, training=training)

def vit_transformer_block(x, block_idx):
    residual = x
    attn = GatedMultiHeadAttention(num_heads=VIT_NUM_HEADS, key_dim=VIT_EMBED_DIM // VIT_NUM_HEADS,
                                   name_suffix=f"blk{block_idx}")(x)
    x = layers.Add()([residual, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    residual = x
    mlp = ReGLUFeedForward(embed_dim=VIT_EMBED_DIM, mlp_dim=VIT_MLP_DIM, dropout=VIT_DROPOUT,
                           name=f"reglu_ffn_blk{block_idx}")(x)
    x = layers.Add()([residual, mlp])
    return layers.LayerNormalization(epsilon=1e-6)(x)

def vit_bottleneck_fn(x, n_layers=VIT_NUM_LAYERS):
    _, H, W, C = x.shape
    if C != VIT_EMBED_DIM:
        x = layers.Conv2D(VIT_EMBED_DIM, 1, padding="same", name="bottleneck_proj")(x)
        C = VIT_EMBED_DIM
    n_patches = H * W
    flat = layers.Reshape((n_patches, VIT_EMBED_DIM))(x)
    flat = AddPositionalEmbedding(n_patches, VIT_EMBED_DIM, name="pos_emb")(flat)
    for i in range(n_layers):
        flat = vit_transformer_block(flat, block_idx=i)
    out = layers.Reshape((H, W, VIT_EMBED_DIM))(flat)
    return layers.Conv2D(C, 1, padding="same", name="bottleneck_unproj")(out)

def build_nusegformer():
    inputs = layers.Input((VIT_IMG_SIZE, VIT_IMG_SIZE, 3), name="input_image")
    effnet = tf.keras.applications.EfficientNetV2S(include_top=False, weights=None, input_tensor=inputs)
    e1 = effnet.get_layer("block1b_add").output
    e2 = effnet.get_layer("block2d_add").output
    e3 = effnet.get_layer("block4a_expand_activation").output
    e4 = effnet.get_layer("block6a_expand_activation").output
    p4 = effnet.get_layer("top_activation").output
    b = vit_bottleneck_fn(p4)
    d4 = vit_upconv_block(b, e4, 272); d3 = vit_upconv_block(d4, e3, 160)
    d2 = vit_upconv_block(d3, e2, 64); d1 = vit_upconv_block(d2, e1, 48)
    d0 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d1)
    d0 = layers.Conv2D(32, 3, padding="same", use_bias=False)(d0)
    d0 = layers.BatchNormalization()(d0)
    d0 = layers.Activation("relu")(d0)
    x = layers.Conv2D(1, 1, name="pre_output")(d0)
    outputs = layers.Activation("sigmoid", name="output")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="NuSegFormer")


# ===============================================================
# 1) Model registry, weight download & loading
# ===============================================================
WEIGHTS_DIR = "model"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "TransUNetCBA": {
        # NOTE: this file is in the legacy Keras H5 format. It must NOT end with ".weights.h5"
        # (Keras 3 would then read it with the wrong loader), so it is saved as plain ".h5".
        "path": os.path.join(WEIGHTS_DIR, "transunet_hybridCBA_fold3.h5"),
        # Upload the weights file to Google Drive (share: "Anyone with the link")
        # and paste its file ID here. Leave empty if you place the file in ./model manually.
        "gdrive_id": "1r07XH2eaU3Ah6CIfKh8lPjtOZkRIAqlI",
        "size": "~312 MB",
        "input_size": (TU_IMG_SIZE, TU_IMG_SIZE),
    },
    "NuSegFormer": {
        # NOTE: Keras 3 only reads this format if the filename ends with ".weights.h5"
        "path": os.path.join(WEIGHTS_DIR, "ViT_ReGeLU_EfficientNetv2_fold1.weights.h5"),
        # Upload the weights file to Google Drive (share: "Anyone with the link")
        # and paste its file ID here. Leave empty if you place the file in ./model manually.
        "gdrive_id": "1dYtKeg77Xo4j44qWsOhuA3XMTSrL8Vc3",
        "size": "~360 MB",
        "input_size": (VIT_IMG_SIZE, VIT_IMG_SIZE),
    },
}

def ensure_weights(cfg):
    if os.path.exists(cfg["path"]):
        return
    if not cfg["gdrive_id"]:
        raise FileNotFoundError(
            f"Weights file not found at '{cfg['path']}'. Place it there, or set "
            f"'gdrive_id' for this model in MODEL_CONFIGS."
        )
    with st.spinner(f"Downloading model weights ({cfg['size']})... this may take a few minutes ⏳"):
        gdown.download(f"https://drive.google.com/uc?id={cfg['gdrive_id']}", cfg["path"], quiet=False)

@st.cache_resource
def load_transunetcba():
    cfg = MODEL_CONFIGS["TransUNetCBA"]
    ensure_weights(cfg)
    model = build_transunetcba(input_shape=(TU_IMG_SIZE, TU_IMG_SIZE, 3))  # weights=None: overwritten anyway
    model.load_weights(cfg["path"])
    return model

@st.cache_resource
def load_nusegformer():
    cfg = MODEL_CONFIGS["NuSegFormer"]
    ensure_weights(cfg)
    model = build_nusegformer()      # weights=None: ImageNet init is overwritten anyway
    model.load_weights(cfg["path"])
    return model

MODEL_LOADERS = {
    "TransUNetCBA": load_transunetcba,
    "NuSegFormer": load_nusegformer,
}

# ===============================================================
# 2) Sidebar controls (model choice must come before loading)
# ===============================================================
st.sidebar.header("⚙️ Controls")
model_name = st.sidebar.selectbox("Segmentation Model", list(MODEL_CONFIGS.keys()), index=1)
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.7, 0.05)
threshold = st.sidebar.slider("Mask Threshold", 0.05, 0.95, 0.5, 0.05)
colormap = st.sidebar.selectbox("Mask Colormap", ["JET", "VIRIDIS", "PLASMA", "HOT", "COOL"])
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
st.sidebar.info(f"💻 Running on: **{device}**")

# Load the selected model (each is cached separately)
try:
    model = MODEL_LOADERS[model_name]()
    st.success(f"✅ {model_name} loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()
target_size = MODEL_CONFIGS[model_name]["input_size"]

# ===============================================================
# 3) Prediction utilities
# ===============================================================
def preprocess_for_model(img_bgr, target_size=(224,224)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_norm = img_resized / 255.0
    return img_norm

def predict_mask(model, img_bgr, target_size=(224,224), threshold=0.5):
    img_norm = preprocess_for_model(img_bgr, target_size)
    inp = np.expand_dims(img_norm, axis=0)
    pred = model.predict(inp, verbose=0)
    pred_mask = (pred[0,:,:,0] > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(pred_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def mask_to_bytes(mask):
    _, buffer = cv2.imencode(".png", mask)
    return BytesIO(buffer).getvalue()

# ===============================================================
# 4) Upload & batch segmentation
#    Results are kept in st.session_state and rendered outside the button
#    block, so they survive reruns (e.g. clicking Download) and images are
#    not dropped by Streamlit's media-file cleanup.
# ===============================================================
def png_bytes(rgb_or_gray):
    arr = rgb_or_gray
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes()

uploaded_files = st.file_uploader(
    "📁 Upload microscopy images (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown(f"### 📂 {len(uploaded_files)} file(s) uploaded")

    if st.button("🔍 Segment All Images", key="segment_all"):
        start_time = time.time()
        results = []
        zip_io = BytesIO()
        with zipfile.ZipFile(zip_io, "w", zipfile.ZIP_DEFLATED) as zipf:
            progress = st.progress(0.0)
            for n, uploaded_file in enumerate(uploaded_files, 1):
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    st.warning(f"⚠️ Could not read {uploaded_file.name}, skipping.")
                    continue

                mask = predict_mask(model, img, target_size=target_size, threshold=threshold)
                cmap = getattr(cv2, f"COLORMAP_{colormap}")
                mask_color = cv2.cvtColor(cv2.applyColorMap(mask, cmap), cv2.COLOR_BGR2RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_rgb, alpha, mask_color, 1 - alpha, 0)

                stem = os.path.splitext(uploaded_file.name)[0]
                zipf.writestr(f"{stem}_mask.png", png_bytes(mask))
                zipf.writestr(f"{stem}_overlay.png", png_bytes(overlay))

                results.append({
                    "name": uploaded_file.name,
                    "original": png_bytes(img_rgb),
                    "mask": png_bytes(mask),
                    "overlay": png_bytes(overlay),
                })
                progress.progress(n / len(uploaded_files))
            progress.empty()

        st.session_state["results"] = results
        st.session_state["zip"] = zip_io.getvalue()
        st.session_state["elapsed"] = time.time() - start_time

    if st.session_state.get("results"):
        st.success(f"Batch segmentation finished in {st.session_state['elapsed']:.1f}s")
        st.download_button(
            "💾 Download All Results (ZIP)",
            data=st.session_state["zip"],
            file_name="segmentation_results.zip",
            mime="application/zip"
        )
        for r in st.session_state["results"]:
            with st.expander(f"🔍 {r['name']}", expanded=True):
                col1, col2, col3 = st.columns(3, gap="medium")
                for col, key, label in ((col1, "original", "Original"),
                                        (col2, "mask", "Predicted Mask"),
                                        (col3, "overlay", "Overlay")):
                    with col:
                        st.image(r[key], use_container_width=True)
                        st.markdown(f"<p style='text-align:center; font-size:20px; font-weight:bold;'>{label}</p>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Cell Nuclei Segmentation</p>", unsafe_allow_html=True)