import cv2
import face_recognition
import numpy as np
import json
import os

class Pessoa:
    def __init__(self, nome, encoding):
        self.nome = nome
        self.encoding = encoding

    @staticmethod
    def from_dict(data):
        return Pessoa(data["nome"], np.array(data["encoding"]))

    def to_dict(self):
        return {"nome": self.nome, "encoding": self.encoding.tolist()}

def carregar_pessoas(arquivo="pessoas.json"):
    if not os.path.exists(arquivo):
        return []
    with open(arquivo, "r") as f:
        dados = json.load(f)
        return [Pessoa.from_dict(p) for p in dados]

def main():
    # Carrega pessoas cadastradas
    pessoas = carregar_pessoas()
    encodings_conhecidos = [p.encoding for p in pessoas]
    nomes_conhecidos = [p.nome for p in pessoas]

    # Inicia a webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converte para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta rostos
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compara rostos
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            nome = "Desconhecido"
            if encodings_conhecidos:
                matches = face_recognition.compare_faces(encodings_conhecidos, face_encoding)
                face_distances = face_recognition.face_distance(encodings_conhecidos, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        nome = nomes_conhecidos[best_match_index]

            # Desenha retângulo e nome
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Exibe o frame
        cv2.imshow("Reconhecimento Facial", frame)

        # Sai com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()