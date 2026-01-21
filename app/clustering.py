"""
Modulo de agrupamento de rostos (clustering)
Agrupa rostos similares em pessoas automaticamente
Atualizado para InsightFace (512-dim embeddings, similaridade de cosseno)
"""
import numpy as np
import logging
from sqlalchemy.orm import Session
from .models import Person, FaceEncoding, FaceExclusion

logger = logging.getLogger(__name__)

# Threshold de similaridade para considerar mesmo rosto (InsightFace)
# 0.35 = mais restritivo (mais pessoas separadas)
# 0.40 = balanceado
# 0.45 = mais permissivo (menos pessoas, mas pode juntar diferentes)
CLUSTERING_THRESHOLD = 0.40


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula similaridade de cosseno entre dois embeddings.
    """
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)

    # Normaliza
    e1 = e1 / (np.linalg.norm(e1) + 1e-10)
    e2 = e2 / (np.linalg.norm(e2) + 1e-10)

    return float(np.dot(e1, e2))


def find_or_create_person(face_encoding: np.ndarray, db: Session, exclude_face_id: int = None) -> Person:
    """
    Encontra uma pessoa existente que corresponda ao encoding do rosto,
    ou cria uma nova pessoa se não encontrar correspondência.

    Args:
        face_encoding: Vetor numpy com o encoding do rosto (512-dim)
        db: Sessão do banco de dados
        exclude_face_id: ID do rosto a ignorar nas exclusões (o próprio rosto sendo processado)

    Returns:
        Person: Pessoa existente ou nova criada
    """
    # Busca todas as pessoas com seus rostos
    persons = db.query(Person).all()

    best_match_person = None
    best_match_similarity = -1.0

    for person in persons:
        # Pega os rostos desta pessoa
        person_faces = [f for f in person.faces if f.encoding]

        if not person_faces:
            continue

        # Verifica se há exclusão para este rosto com esta pessoa
        if exclude_face_id:
            exclusion = db.query(FaceExclusion).filter(
                FaceExclusion.person_id == person.id,
                FaceExclusion.face_id == exclude_face_id
            ).first()
            if exclusion:
                continue  # Pula esta pessoa (foi marcada como "não é essa pessoa")

        # Calcula similaridade para os rostos desta pessoa
        similarities = []
        for pf in person_faces:
            pf_encoding = np.array(pf.encoding)
            sim = compute_similarity(pf_encoding, face_encoding)
            similarities.append(sim)

        # Usa a maior similaridade (melhor match)
        max_similarity = max(similarities)

        if max_similarity > best_match_similarity:
            best_match_similarity = max_similarity
            best_match_person = person

    # Se encontrou correspondência dentro do threshold
    if best_match_person and best_match_similarity >= CLUSTERING_THRESHOLD:
        logger.debug(f"Face matched to Person {best_match_person.id} (similarity: {best_match_similarity:.3f})")
        return best_match_person

    # Cria nova pessoa
    new_person = Person()
    db.add(new_person)
    db.flush()  # Para obter o ID
    logger.info(f"Created new Person {new_person.id}")
    return new_person


def assign_face_to_person(face: FaceEncoding, db: Session) -> Person:
    """
    Atribui um rosto a uma pessoa (existente ou nova).

    Args:
        face: FaceEncoding a ser atribuído
        db: Sessão do banco de dados

    Returns:
        Person: Pessoa à qual o rosto foi atribuído
    """
    if not face.encoding:
        return None

    face_encoding = np.array(face.encoding)
    person = find_or_create_person(face_encoding, db, exclude_face_id=face.id)

    face.person_id = person.id
    return person


def recluster_all_faces(db: Session) -> dict:
    """
    Re-agrupa rostos órfãos (sem pessoa associada).
    Mantém todas as associações existentes e nomes de pessoas.

    Returns:
        dict: Estatísticas do reagrupamento
    """
    logger.info("Starting re-clustering of orphan faces...")

    # Pega apenas rostos SEM pessoa associada
    orphan_faces = db.query(FaceEncoding).filter(
        FaceEncoding.encoding.isnot(None),
        FaceEncoding.person_id.is_(None)
    ).order_by(FaceEncoding.id).all()

    stats = {
        "orphan_faces": len(orphan_faces),
        "faces_assigned": 0,
        "new_persons_created": 0
    }

    logger.info(f"Found {len(orphan_faces)} orphan faces to cluster")

    persons_before = db.query(Person).count()

    for face in orphan_faces:
        person = assign_face_to_person(face, db)
        if person:
            stats["faces_assigned"] += 1

    db.commit()

    persons_after = db.query(Person).count()
    stats["new_persons_created"] = persons_after - persons_before

    # Remove pessoas vazias (sem rostos) que NÃO têm nome
    empty_persons = db.query(Person).filter(
        ~Person.faces.any(),
        (Person.name.is_(None)) | (Person.name == '') | (Person.name.like('Pessoa %'))
    ).all()

    for p in empty_persons:
        logger.info(f"Removing empty person {p.id} (name: {p.name})")
        db.delete(p)
    db.commit()

    logger.info(f"Re-clustering complete: {stats}")
    return stats


def get_person_representative_face(person: Person, db: Session) -> FaceEncoding:
    """
    Retorna o rosto mais representativo de uma pessoa.
    Usa o rosto mais próximo do centróide de todos os rostos.

    Args:
        person: Pessoa para encontrar o rosto representativo
        db: Sessão do banco de dados

    Returns:
        FaceEncoding: Rosto mais representativo
    """
    faces = [f for f in person.faces if f.encoding]

    if not faces:
        return None

    if len(faces) == 1:
        return faces[0]

    # Calcula o centróide
    encodings = np.array([f.encoding for f in faces])
    centroid = np.mean(encodings, axis=0)

    # Normaliza o centroide
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    # Encontra o rosto mais próximo do centróide (maior similaridade)
    best_face = None
    best_similarity = -1.0

    for face in faces:
        enc = np.array(face.encoding)
        enc = enc / (np.linalg.norm(enc) + 1e-10)
        sim = float(np.dot(enc, centroid))
        if sim > best_similarity:
            best_similarity = sim
            best_face = face

    return best_face


def exclude_face_from_person(face_id: int, person_id: int, db: Session) -> dict:
    """
    Marca que um rosto NÃO pertence a uma pessoa (feedback negativo).
    Remove o rosto da pessoa e cria uma exclusão.

    Args:
        face_id: ID do rosto
        person_id: ID da pessoa
        db: Sessão do banco de dados

    Returns:
        dict: Resultado da operação
    """
    face = db.query(FaceEncoding).filter(FaceEncoding.id == face_id).first()
    if not face:
        return {"error": "Face not found"}

    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        return {"error": "Person not found"}

    # Cria exclusão
    exclusion = FaceExclusion(person_id=person_id, face_id=face_id)
    db.add(exclusion)

    # Remove associação do rosto com a pessoa
    if face.person_id == person_id:
        face.person_id = None

    db.commit()

    # Tenta reatribuir o rosto a outra pessoa
    if face.encoding:
        new_person = assign_face_to_person(face, db)
        db.commit()

        return {
            "success": True,
            "message": f"Face removed from Person {person_id}",
            "new_person_id": new_person.id if new_person else None
        }

    return {"success": True, "message": f"Face removed from Person {person_id}"}


def merge_persons(source_id: int, target_id: int, db: Session) -> dict:
    """
    Mescla duas pessoas em uma (move todos os rostos da source para target).

    Args:
        source_id: ID da pessoa a ser mesclada (será removida)
        target_id: ID da pessoa destino
        db: Sessão do banco de dados

    Returns:
        dict: Resultado da operação
    """
    source = db.query(Person).filter(Person.id == source_id).first()
    target = db.query(Person).filter(Person.id == target_id).first()

    if not source or not target:
        return {"error": "Person not found"}

    if source_id == target_id:
        return {"error": "Cannot merge person with itself"}

    logger.info(f"MERGE: source_id={source_id} (name='{source.name}', faces={len(source.faces)})")
    logger.info(f"MERGE: target_id={target_id} (name='{target.name}', faces={len(target.faces)})")

    # Preserva nome: se target não tem nome real mas source tem, usa o nome do source
    target_has_real_name = target.name and not target.name.startswith("Pessoa ")
    source_has_real_name = source.name and not source.name.startswith("Pessoa ")

    if not target_has_real_name and source_has_real_name:
        target.name = source.name
        logger.info(f"MERGE: Preserved name '{source.name}' from Person {source_id} to Person {target_id}")

    # Conta faces antes de mover
    faces_count = len(source.faces)
    logger.info(f"MERGE: Moving {faces_count} faces from source to target using direct SQL")

    # Move rostos usando UPDATE direto com synchronize_session=False
    # para evitar conflitos com objetos já carregados na sessão
    from .models import FaceEncoding, FaceExclusion
    faces_moved = db.query(FaceEncoding).filter(
        FaceEncoding.person_id == source_id
    ).update({FaceEncoding.person_id: target_id}, synchronize_session=False)

    # Move exclusões usando UPDATE direto
    db.query(FaceExclusion).filter(
        FaceExclusion.person_id == source_id
    ).update({FaceExclusion.person_id: target_id}, synchronize_session=False)

    logger.info(f"MERGE: SQL UPDATE moved {faces_moved} faces")

    # Flush para garantir que UPDATEs foram executados antes do delete
    db.flush()

    # Expira o objeto source para evitar conflitos
    db.expire(source)

    # Remove pessoa source
    db.delete(source)
    db.commit()

    logger.info(f"MERGE: Complete. Moved {faces_moved} faces. Target now has {len(target.faces)} faces")

    return {
        "success": True,
        "faces_moved": faces_moved,
        "target_person_id": target_id
    }
