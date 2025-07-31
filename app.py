def collect_images_labels(base_dir):
    images = []
    labels = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if file.endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(label_dir, file))
                labels.append(label)
    return images, labels

st.write("正在加载数据...")
dermnet_images, dermnet_labels = collect_images_labels(dermnet_dir)
st.write(f"共找到 {len(dermnet_images)} 张图片, 类别数量 {len(set(dermnet_labels))}")

# 标签编码
label_to_index = {label: idx for idx, label in enumerate(sorted(set(dermnet_labels)))}
index_to_label = {v: k for k, v in label_to_index.items()}

# -------------------------
# 图片预处理函数
# -------------------------
IMG_SIZE = 128

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img

# -------------------------
# 构建CNN模型
# -------------------------
def create_cnn_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# Streamlit 界面
# -------------------------
st.title("皮肤病识别 Demo")

# 训练按钮
if st.button("识别"):
    st.write("正在识别中...")

    # 转 numpy
    X = np.array([preprocess_image(p) for p in dermnet_images[:500]])   # 取前500张快速训练
    y = np.array([label_to_index[l] for l in dermnet_labels[:500]])

    # 划分训练测试
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    st.write("开始识别...")
    model = create_cnn_model(num_classes=len(label_to_index))
    model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))
    model.save("skin_model.h5")
    st.success("识别完成，模型已保存！")

# 上传预测
uploaded_file = st.file_uploader("上传皮肤照片", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    img = preprocess_image("temp.jpg")
    img = np.expand_dims(img, axis=0)

    if os.path.exists("skin_model.h5"):
        model = tf.keras.models.load_model("skin_model.h5")
        pred = model.predict(img)
        class_idx = np.argmax(pred)
        st.image("temp.jpg", caption="上传图片", use_container_width=True)
        st.write(f"预测结果: **{index_to_label[class_idx]}**")
    else:
        st.warning("请先点击上方按钮训练模型。")
