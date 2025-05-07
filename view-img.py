import cv2
import numpy as np
import os

weights_path = "conf/yolov3.weights"
cfg_path = "conf/yolov3.cfg"
names_path = "conf/coco.names"

for file_path in [weights_path, cfg_path, names_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}. Verifique o diretório.")

net = cv2.dnn.readNet(weights_path, cfg_path)

layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

imagem = cv2.imread("assets/imagens/copo.png")
# imagem = cv2.imread("assets/imagens/praia1.png")
# imagem = cv2.imread("assets/imagens/praia2.png")
# imagem = cv2.imread("assets/imagens/rato.jpg")

if imagem is None:
    raise FileNotFoundError("Imagem 'copo.png' não encontrada. Verifique o nome ou o caminho.")

height, width, channels = imagem.shape

blob = cv2.dnn.blobFromImage(imagem, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Verde
        cv2.rectangle(imagem, (x, y), (x + w, y + h), color, 2)
        cv2.putText(imagem, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

screen_width = 700
screen_height = 500

scale_width = screen_width / width
scale_height = screen_height / height
scale = min(scale_width, scale_height)

new_width = int(width * scale)
new_height = int(height * scale)

imagem_resized = cv2.resize(imagem, (new_width, new_height), interpolation=cv2.INTER_AREA)

tela = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

x_offset = (screen_width - new_width) // 2
y_offset = (screen_height - new_height) // 2

tela[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = imagem_resized

cv2.imshow("Minha Imagem", tela)
cv2.waitKey(0)
cv2.destroyAllWindows()