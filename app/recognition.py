"""
Modulo de reconhecimento facial usando InsightFace (ArcFace)
Mais preciso que dlib/face_recognition (~99.83% vs ~99.38%)
"""
import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Singleton para o modelo InsightFace
_face_analyzer = None


def get_face_analyzer():
    """
    Retorna instancia singleton do analisador de faces.
    Carrega o modelo apenas uma vez.
    """
    global _face_analyzer

    if _face_analyzer is None:
        logger.info("Inicializando InsightFace...")
        try:
            from insightface.app import FaceAnalysis

            # buffalo_l = modelo mais preciso
            # buffalo_s = modelo mais leve (mais rapido, menos preciso)
            _face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']  # Use CUDAExecutionProvider para GPU
            )
            _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar InsightFace: {e}")
            raise

    return _face_analyzer


def detect_faces(image: np.ndarray) -> List[dict]:
    """
    Detecta rostos em uma imagem.

    Args:
        image: Imagem em formato numpy (BGR ou RGB)

    Returns:
        Lista de dicts com:
        - bbox: [x1, y1, x2, y2]
        - embedding: vetor 512-dim
        - det_score: confianca da deteccao
        - landmarks: pontos faciais
    """
    analyzer = get_face_analyzer()

    # InsightFace espera BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    faces = analyzer.get(image)

    results = []
    for face in faces:
        # Converte bbox para formato [top, right, bottom, left] (compativel com o sistema atual)
        # IMPORTANTE: Converter para int nativo do Python (nao numpy.int64) para serializacao JSON
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        location = [y1, x2, y2, x1]  # [top, right, bottom, left]

        results.append({
            'location': location,
            'embedding': face.embedding.tolist(),  # 512-dim
            'confidence': float(face.det_score),
            'landmarks': face.landmark_2d_106.tolist() if face.landmark_2d_106 is not None else None
        })

    return results


def load_image(file_path: str) -> np.ndarray:
    """
    Carrega uma imagem do disco, tratando orientacao EXIF.

    Args:
        file_path: Caminho para o arquivo de imagem

    Returns:
        Imagem em formato numpy BGR
    """
    from PIL import Image, ExifTags

    # Abre com PIL para tratar EXIF
    pil_image = Image.open(file_path)

    # Trata rotacao EXIF
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = pil_image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation_value == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation_value == 8:
                pil_image = pil_image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass

    # Converte para RGB e depois para BGR (OpenCV)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula similaridade entre dois embeddings usando distancia de cosseno.
    InsightFace usa embeddings normalizados, entao cosseno = dot product.

    Args:
        embedding1: Primeiro embedding (512-dim)
        embedding2: Segundo embedding (512-dim)

    Returns:
        Similaridade entre 0 e 1 (1 = identico)
    """
    # Normaliza os embeddings
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)

    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    # Similaridade de cosseno
    similarity = np.dot(e1, e2)

    return float(similarity)


def compute_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula distancia euclidiana entre dois embeddings.
    Mantido para compatibilidade com o sistema atual.

    Args:
        embedding1: Primeiro embedding
        embedding2: Segundo embedding

    Returns:
        Distancia (0 = identico, maior = mais diferente)
    """
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)

    # Normaliza antes de calcular distancia
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    return float(np.linalg.norm(e1 - e2))


def is_same_person(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.4) -> bool:
    """
    Verifica se dois embeddings sao da mesma pessoa.

    Args:
        embedding1: Primeiro embedding
        embedding2: Segundo embedding
        threshold: Limiar de similaridade (padrao 0.4 para InsightFace)

    Returns:
        True se forem a mesma pessoa
    """
    similarity = compute_similarity(embedding1, embedding2)
    return similarity >= threshold


# Thresholds recomendados para InsightFace
# Similaridade (cosseno):
#   >= 0.5 : muito provavel mesma pessoa
#   >= 0.4 : provavelmente mesma pessoa
#   >= 0.3 : possivelmente mesma pessoa
#   <  0.3 : provavelmente pessoas diferentes

# Distancia (euclidiana normalizada):
#   <= 0.8 : muito provavel mesma pessoa
#   <= 1.0 : provavelmente mesma pessoa
#   <= 1.2 : possivelmente mesma pessoa
#   >  1.2 : provavelmente pessoas diferentes

SIMILARITY_THRESHOLD = 0.4  # Para clustering
RECOGNITION_THRESHOLD = 0.5  # Para reconhecimento em tempo real (mais estrito)
