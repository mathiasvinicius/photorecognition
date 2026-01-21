from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime


class PersonGroup(Base):
    """Grupo de pessoas (ex: Família, Amigos, Trabalho)"""
    __tablename__ = "person_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # Nome do grupo (ex: "Família")
    color = Column(String, nullable=True)  # Cor do badge (ex: "#ff6b6b")
    icon = Column(String, nullable=True)  # Icone do grupo (ex: "fas fa-users")
    created_at = Column(DateTime, default=datetime.utcnow)

    persons = relationship("Person", back_populates="group")


class Person(Base):
    """Representa uma pessoa identificada pelo sistema"""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)  # Nome dado pelo usuário (ex: "Ariane")
    group_id = Column(Integer, ForeignKey("person_groups.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    group = relationship("PersonGroup", back_populates="persons")
    faces = relationship("FaceEncoding", back_populates="person")
    exclusions = relationship("FaceExclusion", back_populates="person", cascade="all, delete-orphan")


class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, unique=True, index=True)
    filename = Column(String)
    capture_date = Column(DateTime, index=True, nullable=True)
    camera_model = Column(String, index=True, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow)

    faces = relationship("FaceEncoding", back_populates="photo", cascade="all, delete-orphan")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, index=True)
    photo_id = Column(Integer, ForeignKey("photos.id"))
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)  # Pessoa associada
    encoding = Column(JSON)  # Storing numpy array as JSON list
    face_location = Column(JSON)  # [top, right, bottom, left] - stored during scan

    photo = relationship("Photo", back_populates="faces")
    person = relationship("Person", back_populates="faces")


class FaceExclusion(Base):
    """Marca que um rosto NÃO pertence a uma pessoa (feedback negativo)"""
    __tablename__ = "face_exclusions"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    face_id = Column(Integer, ForeignKey("face_encodings.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    person = relationship("Person", back_populates="exclusions")
    face = relationship("FaceEncoding")


class RecognitionHistory(Base):
    """Registro de reconhecimento ao vivo"""
    __tablename__ = "recognition_history"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    person_name = Column(String, nullable=True)
    similarity = Column(Float, nullable=True)
    recognized = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    person = relationship("Person")


class CaptureEvent(Base):
    """Registro de captura de frames no ao vivo"""
    __tablename__ = "capture_events"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    person_name = Column(String, nullable=True)
    label = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    person = relationship("Person")
