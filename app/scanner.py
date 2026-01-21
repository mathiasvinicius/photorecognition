import os
from sqlalchemy.orm import Session
from .models import Photo, FaceEncoding
from .utils import get_exif_data
from .database import SessionLocal
from .status import set_status, is_paused, should_stop, reset_scan_flags
import time
from .clustering import assign_face_to_person
from .recognition import detect_faces, load_image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def scan_paths(paths: list, db: Session):
    """
    Scans a list of directory paths sequentially.
    """
    reset_scan_flags()  # Reset control flags at start
    total = len(paths)
    for index, path in enumerate(paths):
        if should_stop():
            logger.info("Scan stopped by user")
            set_status("Idle", "Scan stopped by user")
            reset_scan_flags()
            return
        logger.info(f"Scanning folder {index + 1}/{total}: {path}")
        scan_directory(path, db)

    reset_scan_flags()
    set_status("Idle", "All selected scans complete")


def scan_directory(directory_path: str, db: Session):
    logger.info(f"Starting scan of {directory_path}")
    set_status("Scanning", f"Folder: {os.path.basename(directory_path)}")

    try:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Check for stop request
                if should_stop():
                    logger.info("Scan stopped by user during directory scan")
                    return

                # Check for pause
                while is_paused():
                    time.sleep(0.5)  # Wait while paused
                    if should_stop():
                        logger.info("Scan stopped while paused")
                        return

                if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    set_status("Scanning", f"Processing: {file}")
                    process_image(file_path, db)

    except Exception as e:
        logger.error(f"Scan failed: {e}")
        set_status("Idle", f"Error: {str(e)}")


def process_image(file_path: str, db: Session):
    # Check if already exists
    existing = db.query(Photo).filter(Photo.file_path == file_path).first()
    if existing:
        return  # Skip if already indexed

    try:
        # 1. Load Image using InsightFace loader (handles EXIF)
        image = load_image(file_path)
        height, width = image.shape[:2]

        # 2. Extract EXIF
        capture_date, camera_model = get_exif_data(file_path)

        # 3. Save Photo Metadata
        photo = Photo(
            file_path=file_path,
            filename=os.path.basename(file_path),
            capture_date=capture_date,
            camera_model=camera_model,
            width=width,
            height=height
        )
        db.add(photo)
        db.commit()
        db.refresh(photo)

        # 4. Detect Faces using InsightFace
        faces = detect_faces(image)

        for face_data in faces:
            face_entry = FaceEncoding(
                photo_id=photo.id,
                encoding=face_data['embedding'],  # 512-dim embedding
                face_location=face_data['location']  # [top, right, bottom, left]
            )
            db.add(face_entry)
            db.flush()  # Para obter o ID do face_entry

            # Atribui o rosto a uma pessoa (existente ou nova)
            person = assign_face_to_person(face_entry, db)
            if person:
                logger.debug(f"Face {face_entry.id} assigned to Person {person.id}")

        db.commit()
        logger.info(f"Indexed: {file_path} - Faces found: {len(faces)}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        db.rollback()
