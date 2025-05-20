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

def salvar_pessoa(pessoa, arquivo="pessoas.json"):
    pessoas = []
    if os.path.exists(arquivo):
        with open(arquivo, "r") as f:
            pessoas = json.load(f)
    pessoas.append(pessoa.to_dict())
    with open(arquivo, "w") as f:
        json.dump(pessoas, f, indent=4)

def main():
    nome = input("Digite o nome da pessoa a cadastrar: ")

    # Inicia a webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Olhe para a câmera. Pressione 's' para capturar o rosto ou 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converte para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta rostos
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Desenha retângulos nos rostos detectados
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Rosto Detectado", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Cadastro de Rosto", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and face_encodings:
            encoding = face_encodings[0]
            pessoa = Pessoa(nome, encoding)
            salvar_pessoa(pessoa)
            print(f"{nome} cadastrado com sucesso!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()