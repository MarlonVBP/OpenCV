# OpenCV Object Detection and Facial Recognition

## 📖 Descrição

Este projeto demonstra o uso do **OpenCV** e **YOLOv3** para detecção de objetos em imagens e vídeos, além de reconhecimento facial com a biblioteca `face_recognition`. Desenvolvido como parte de um seminário de Inteligência Artificial, o projeto explora aplicações práticas de visão computacional, incluindo identificação de objetos (como carros, pessoas e itens do cotidiano), reconhecimento facial personalizado e análise de imagens. O objetivo é fornecer uma base acessível para estudantes, desenvolvedores e entusiastas explorarem o potencial do OpenCV em soluções inovadoras.

Os scripts incluem exemplos para:
- Exibir e detectar objetos em imagens estáticas.
- Processar vídeos em tempo real via webcam.
- Reconhecer pessoas específicas (ex.: "Luiz") a partir de fotos de treinamento.

Os arquivos de *weights* do YOLOv3, devido ao seu tamanho, estão hospedados no Google Drive para facilitar o acesso.

## 🚀 Como Implementar

### Pré-requisitos
- **Python 3.7+** instalado.
- Uma webcam (para scripts de detecção em tempo real).
- Acesso à internet para baixar as bibliotecas e os arquivos de *weights*.

### Bibliotecas Necessárias
Instale as seguintes bibliotecas usando `pip`:

```bash
pip install opencv-python
pip install numpy
pip install face_recognition
```

> **Nota:** Para instalar `face_recognition`, pode ser necessário instalar `cmake` e `dlib` primeiro:
> ```bash
> pip install cmake
> pip install dlib
> ```

### Arquivos de *Weights*
O arquivo de *weights* do YOLOv3 (`yolov3.weights`) é muito grande para o GitHub e está hospedado no Google Drive. Baixe aqui:

🔗 [Download dos Arquivos](https://drive.google.com/file/d/1Va4WjQ4ppNjGFzgJreXU-LxAcbQmhuha/view?usp=sharing)

Após o download, extraia os arquivos e coloque-os na pasta raiz do projeto.

### Estrutura do Projeto
- `detect_image.py`: Detecta objetos em uma imagem estática usando YOLOv3.
- `detect_webcam.py`: Realiza detecção de objetos em tempo real via webcam.
- `face_recognition.py`: Reconhece pessoas específicas (ex.: "Luiz") a partir de fotos de treinamento.
- `fotos_luiz/`: Pasta para armazenar fotos de treinamento para reconhecimento facial (ex.: `luiz1.jpg`, `luiz2.jpg`).

### Passos para Executar
1. **Clone o Repositório**:
   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
   cd seu_repositorio
   ```

2. **Baixe os Arquivos de *Weights***:
   - Acesse o [link do Google Drive](https://drive.google.com/file/d/1Va4WjQ4ppNjGFzgJreXU-LxAcbQmhuha/view?usp=sharing).
   - Coloque `yolov3.weights`, `yolov3.cfg` e `coco.names` na pasta raiz.

3. **Instale as Dependências**:
   ```bash
   pip install -r requirements.txt
   ```
   > Crie um arquivo `requirements.txt` com:
   ```
   opencv-python
   numpy
   face_recognition
   ```

4. **Execute os Scripts**:
   - Para detecção em imagem:
     ```bash
     python detect_image.py
     ```
     > Edite o nome da imagem no script (ex.: `"image.png"`).
   - Para detecção via webcam:
     ```bash
     python detect_webcam.py
     ```
   - Para reconhecimento facial:
     - Coloque 5+ fotos do Luiz na pasta `fotos_luiz/`.
     - Execute:
       ```bash
       python face_recognition.py
       ```

## 📝 Licença

Este projeto é licenciado sob a **Licença MIT**. Você pode usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do software, desde que inclua o aviso de copyright e a permissão de uso em todas as cópias ou partes substanciais do software. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🛠️ Notas Adicionais
- **Performance**: Em CPUs, o FPS pode ser baixo com YOLOv3. Para melhor desempenho, use uma GPU com CUDA.
- **Personalização**: Para reconhecer outras pessoas ou objetos, adicione mais fotos na pasta `fotos_luiz/` ou treine um modelo YOLO customizado.
- **Problemas Comuns**:
  - Se a webcam não funcionar, tente mudar o índice em `cv2.VideoCapture(0)` para `1` ou `2`.
  - Verifique se os arquivos de *weights* estão no diretório correto.

## 📚 Referências
- [OpenCV Official Documentation](https://docs.opencv.org)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Face Recognition GitHub](https://github.com/ageitgey/face_recognition)

---
