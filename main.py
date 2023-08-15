import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import torch
import requests
import io
import math


def _object_detection(image: Image, threshold: float = 0.25) -> Image:
    # 物体検出
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    pred = model(image)

    # 色の一覧を作成
    cmap = plt.cm.get_cmap("hsv", len(model.model.names))

    # フォント設定
    truetype_url = "https://github.com/JotJunior/PHP-Boleto-ZF2/blob/master/public/assets/fonts/arial.ttf?raw=true"
    r = requests.get(truetype_url, allow_redirects=True)
    sqrt = math.sqrt(image.size[0] * image.size[1] / 10000)
    size = int(sqrt * 5)
    rec_width = int(sqrt / 2)
    font = ImageFont.truetype(io.BytesIO(r.content), size=size)

    # 検出結果の描画
    for detections in pred.xyxy:
        for detection in detections:
            class_id = int(detection[5])
            class_name = str(model.model.names[class_id])
            bbox = [int(x) for x in detection[:4].tolist()]
            conf = float(detection[4])
            # 閾値以上のconfidenceの場合のみ描画
            if conf >= threshold:
                color = cmap(class_id, bytes=True)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline=color, width=rec_width)
                draw.text(
                    [bbox[0] + 5, bbox[1] + 10], class_name, fill=color, font=font
                )

    return image


def main():
    classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    st.title("物体検出")
    st.write(
        "COCO データセットに含まれる「車」「人」「犬」「自転車」などの物体を検出するアプリです。"
        "\n画像ファイルをアップロードし、「実行」ボタンを押すと、物体検出結果を表示します。"
    )

    with st.form(key="detect_form"):
        # 画像のロード
        uploaded_file = st.file_uploader("ファイルアップロード", type=["jpg", "png"])
        # 閾値の設定
        threshold = st.number_input(
            "threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01
        )
        # 送信ボタン
        submit_button = st.form_submit_button(label="実行")

    if submit_button:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        # 予測と描画
        image = _object_detection(image, threshold=threshold)
        st.image(image, caption="結果画像", use_column_width=True)

    st.write("COCOデータセットに含まれるクラス一覧", classes)


if __name__ == "__main__":
    main()
