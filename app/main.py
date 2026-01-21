from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from typing import List, Optional
from datetime import datetime, timezone
import shutil
import os
import json
import uuid
import base64
import cv2
import numpy as np
import logging
import asyncio
import threading
import time
from pydantic import BaseModel

from .database import engine, get_db, Base, SessionLocal
from .models import Photo, FaceEncoding, Person, FaceExclusion, PersonGroup, RecognitionHistory, CaptureEvent
from .scanner import scan_directory, scan_paths
from .organizer import organize_photos
from .status import get_status, set_status, pause_scan, resume_scan, stop_scan
from .clustering import (
    recluster_all_faces,
    exclude_face_from_person,
    merge_persons,
    get_person_representative_face,
    compute_similarity,
    CLUSTERING_THRESHOLD
)
from .recognition import detect_faces, load_image, compute_similarity as rec_compute_similarity
from .cache_utils import (
    ensure_cache_dirs,
    build_cache_path,
    load_cache_settings,
    update_cache_settings,
    create_thumbnail,
    save_webp,
    touch_cache_file,
    enforce_cache_limit,
)

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request Models
class ScanRequest(BaseModel):
    folders: List[str] = []


class DirectoryRequest(BaseModel):
    path: str
    name: Optional[str] = None



# Create Tables
Base.metadata.create_all(bind=engine)

# Auto-migrate group icon column
try:
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('person_groups')]
    if 'icon' not in columns:
        with engine.connect() as conn:
            conn.execute(text('ALTER TABLE person_groups ADD COLUMN icon VARCHAR'))
            conn.commit()
except Exception as e:
    logger.warning(f"Could not ensure group icon column: {e}")

ensure_cache_dirs()

# Camera storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
CAMERAS_FILE = os.getenv("CAMERAS_FILE", os.path.join(DATA_DIR, "cameras.json"))
CAPTURES_DIR = os.getenv("CAPTURES_DIR", "/mnt/Capturas")
os.makedirs(CAPTURES_DIR, exist_ok=True)


def load_saved_cameras() -> List[dict]:
    """Load saved RTSP camera configs from disk."""
    if not os.path.exists(CAMERAS_FILE):
        return []
    try:
        with open(CAMERAS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Could not load cameras file: {e}")
        return []


def save_saved_cameras(cameras: List[dict]) -> None:
    """Persist saved RTSP camera configs to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = CAMERAS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(cameras, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CAMERAS_FILE)


def resolve_capture_dir(location: Optional[str], custom_path: Optional[str] = None) -> str:
    """Resolve capture directory based on configured location."""
    if location == "photos":
        return os.path.join(PHOTOS_DIR, "Capturas")
    if location == "custom" and custom_path:
        return custom_path
    return CAPTURES_DIR


def to_utc_iso(value: datetime) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


# Initialize App
app = FastAPI(title="PhotoRecognition API")


@app.on_event("startup")
def startup_backend_workers():
    sync_backend_workers()

# Mount photos directory to serve images directly
PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/mnt/photos")
if os.path.exists(PHOTOS_DIR):
    app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")
else:
    logger.warning(f"Photos directory {PHOTOS_DIR} does not exist. Images will not be served.")

if os.path.exists(CAPTURES_DIR):
    app.mount("/capture-files", StaticFiles(directory=CAPTURES_DIR), name="captures")
else:
    logger.warning(f"Captures directory {CAPTURES_DIR} does not exist. Captures will not be served.")

# Mount Static Files for Frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")

@app.get("/folders")
def list_folders():
    """
    List immediate subdirectories in the PHOTOS_DIR.
    """
    logger.info(f"Listing folders in: {PHOTOS_DIR}")
    if not os.path.exists(PHOTOS_DIR):
        logger.error(f"Path does not exist: {PHOTOS_DIR}")
        return []

    try:
        items = os.listdir(PHOTOS_DIR)
        logger.info(f"Found items: {items}")
        folders = [item for item in items if os.path.isdir(os.path.join(PHOTOS_DIR, item))]
        folders.sort()
        return folders
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        return []


ALLOWED_BROWSE_ROOTS = ["/mnt"]


def normalize_browse_path(path: str) -> str:
    if not path:
        return CAPTURES_DIR
    real_path = os.path.realpath(path)
    if not any(real_path.startswith(root) for root in ALLOWED_BROWSE_ROOTS):
        raise HTTPException(status_code=400, detail="Invalid path")
    return real_path


@app.get("/directories")
def list_directories(path: Optional[str] = None):
    """List subdirectories for a given path (restricted)."""
    base = normalize_browse_path(path or CAPTURES_DIR)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="Path not found")
    if not os.path.isdir(base):
        raise HTTPException(status_code=400, detail="Not a directory")

    try:
        entries = sorted([
            name for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name))
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "path": base,
        "directories": [
            {"name": name, "path": os.path.join(base, name)} for name in entries
        ]
    }


@app.post("/directories")
def create_directory(request: DirectoryRequest):
    """Create a subdirectory in the given path (restricted)."""
    if not request.name:
        raise HTTPException(status_code=400, detail="Name required")
    base = normalize_browse_path(request.path)
    name = os.path.basename(request.name.strip())
    if not name or name in (".", ".."): 
        raise HTTPException(status_code=400, detail="Invalid folder name")

    target = os.path.join(base, name)
    try:
        os.makedirs(target, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"path": target}

@app.post("/scan")
def trigger_scan(request: ScanRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Triggers a background scan of the selected folders.
    If no folders provided, scans the root PHOTOS_DIR.
    """
    targets = []

    if not request.folders:
        # Scan everything
        targets = [PHOTOS_DIR]
    else:
        # Build absolute paths
        for folder in request.folders:
            # Basic security: prevent traversing up
            safe_folder = os.path.basename(folder)
            targets.append(os.path.join(PHOTOS_DIR, safe_folder))

    background_tasks.add_task(scan_paths, targets, db)
    return {"message": f"Scan started for {len(targets)} path(s)"}


@app.post("/scan/pause")
def pause_scan_endpoint():
    """
    Pausa o scan em andamento.
    """
    success = pause_scan()
    return {"success": success, "message": "Scan paused" if success else "No scan to pause"}


@app.post("/scan/resume")
def resume_scan_endpoint():
    """
    Continua o scan pausado.
    """
    success = resume_scan()
    return {"success": success, "message": "Scan resumed" if success else "No scan to resume"}


@app.post("/scan/stop")
def stop_scan_endpoint():
    """
    Para o scan completamente.
    """
    success = stop_scan()
    return {"success": success, "message": "Scan stop requested" if success else "Failed to stop scan"}


@app.post("/organize")
def trigger_organize(
    mode: str = Query("copy", pattern="^(copy|move)$"),
    dest: Optional[str] = Query(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Organize photos from Recovery Dir to Organized Dir.
    """
    recovery_dir = os.getenv("RECOVERY_DIR", "/mnt/recovery")
    organized_dir = os.getenv("ORGANIZED_DIR", "/mnt/organized")
    if dest:
        organized_dir = normalize_browse_path(dest)
        os.makedirs(organized_dir, exist_ok=True)

    background_tasks.add_task(organize_photos, recovery_dir, organized_dir, mode)
    return {"message": f"Organization ({mode}) started in background"}

@app.post("/search/face")
async def search_by_face(
    file: UploadFile = File(...),
    threshold: float = 0.4,  # InsightFace similarity threshold
    db: Session = Depends(get_db)
):
    """
    Upload an image to find similar faces in the database.
    Uses InsightFace for face detection and matching.
    """
    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load and detect faces using InsightFace
        image = load_image(temp_filename)
        faces = detect_faces(image)

        if not faces:
            return {"message": "No faces found in uploaded image", "matches": []}

        # Use the first face found as target
        target_embedding = np.array(faces[0]['embedding'])

        # Fetch all known encodings
        known_faces = db.query(FaceEncoding).all()

        matches = []
        for face_entry in known_faces:
            # Calculate similarity using cosine
            known_embedding = np.array(face_entry.encoding)
            similarity = rec_compute_similarity(known_embedding, target_embedding)

            if similarity >= threshold:
                matches.append({
                    "photo_id": face_entry.photo.id,
                    "file_path": face_entry.photo.file_path,
                    "filename": face_entry.photo.filename,
                    "capture_date": face_entry.photo.capture_date,
                    "similarity": float(similarity),
                    "distance": float(1 - similarity)  # For compatibility
                })

        # Sort by best match (highest similarity)
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return {"count": len(matches), "matches": matches}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/search/metadata")
def search_metadata(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_model: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    query = db.query(Photo)

    if start_date:
        query = query.filter(Photo.capture_date >= start_date)
    if end_date:
        query = query.filter(Photo.capture_date <= end_date)
    if camera_model:
        query = query.filter(Photo.camera_model.ilike(f"%{camera_model}%"))

    results = query.order_by(Photo.capture_date.desc()).limit(limit).all()
    return results

@app.get("/faces/all")
def get_all_faces(limit: int = 100, db: Session = Depends(get_db)):
    """
    Returns a list of photos that contain at least one detected face.
    """
    photos = db.query(Photo).join(FaceEncoding).distinct().limit(limit).all()

    results = []
    for p in photos:
        results.append({
            "id": p.id,
            "filename": p.filename,
            "file_path": p.file_path,
            "capture_date": p.capture_date,
            "face_count": len(p.faces)
        })
    return results

@app.get("/photo/{photo_id}/details")
def get_photo_details(photo_id: int, db: Session = Depends(get_db)):
    """
    Returns comprehensive metadata/EXIF details for a photo.
    Similar to Windows file properties.
    """
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    details = {
        "id": photo.id,
        "filename": photo.filename,
        "file_path": photo.file_path,
        "folder": os.path.dirname(photo.file_path).replace(PHOTOS_DIR, "").lstrip("/"),
        "capture_date": photo.capture_date.isoformat() if photo.capture_date else None,
        "camera_model": photo.camera_model,
        "width": photo.width,
        "height": photo.height,
        "added_at": photo.added_at.isoformat() if photo.added_at else None,
        "face_count": len(photo.faces),
        "exif": {}
    }
    
    # Get file stats
    if os.path.exists(photo.file_path):
        stat = os.stat(photo.file_path)
        details["file_size"] = stat.st_size
        details["file_size_human"] = f"{stat.st_size / 1024 / 1024:.2f} MB"
        details["modified_date"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    # Extract EXIF data
    try:
        with Image.open(photo.file_path) as img:
            details["width"] = img.width
            details["height"] = img.height
            details["format"] = img.format
            details["mode"] = img.mode
            
            exif_data = img._getexif()
            if exif_data:
                exif_readable = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Skip binary data and very long values
                    if isinstance(value, bytes):
                        continue
                    if isinstance(value, str) and len(value) > 200:
                        continue
                    
                    # Convert IFD references to readable format
                    if tag == "GPSInfo":
                        gps_data = {}
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            if not isinstance(gps_value, bytes):
                                gps_data[gps_tag] = str(gps_value)
                        exif_readable["GPS"] = gps_data
                    else:
                        # Convert to string for JSON serialization
                        try:
                            exif_readable[tag] = str(value) if not isinstance(value, (int, float)) else value
                        except:
                            pass
                
                details["exif"] = exif_readable
                
                # Extract common fields to top level
                if "Make" in exif_readable:
                    details["camera_make"] = exif_readable["Make"]
                if "Model" in exif_readable:
                    details["camera_model"] = exif_readable["Model"]
                if "LensModel" in exif_readable:
                    details["lens"] = exif_readable["LensModel"]
                if "FocalLength" in exif_readable:
                    details["focal_length"] = exif_readable["FocalLength"]
                if "FNumber" in exif_readable:
                    details["aperture"] = exif_readable["FNumber"]
                if "ISOSpeedRatings" in exif_readable:
                    details["iso"] = exif_readable["ISOSpeedRatings"]
                if "ExposureTime" in exif_readable:
                    details["exposure"] = exif_readable["ExposureTime"]
                if "Flash" in exif_readable:
                    details["flash"] = exif_readable["Flash"]
                if "Software" in exif_readable:
                    details["software"] = exif_readable["Software"]
                    
    except Exception as e:
        logger.warning(f"Could not read EXIF for photo {photo_id}: {e}")
    
    return details


@app.get("/search/folder")
def search_by_folder(folder: str, limit: int = 100, db: Session = Depends(get_db)):
    """
    Search for photos in a specific folder.
    """
    # Build the full path pattern
    folder_path = os.path.join(PHOTOS_DIR, folder)
    
    photos = db.query(Photo).filter(
        Photo.file_path.like(f"{folder_path}%")
    ).limit(limit).all()
    
    return [
        {
            "id": p.id,
            "photo_id": p.id,
            "filename": p.filename,
            "file_path": p.file_path,
            "capture_date": p.capture_date,
            "camera_model": p.camera_model
        }
        for p in photos
    ]


@app.get("/photo/{photo_id}/detect")
def detect_faces_in_photo(photo_id: int, db: Session = Depends(get_db)):
    """
    Returns face locations for a specific photo to draw interactive boxes.
    Uses stored locations from database (fast) with fallback to re-detection.
    """
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    stored_faces = photo.faces

    # Check if we have stored locations
    has_stored_locations = any(sf.face_location for sf in stored_faces)

    if has_stored_locations:
        # Fast path: use pre-computed locations from database
        results = []
        for sf in stored_faces:
            if sf.face_location:
                results.append({
                    "location": sf.face_location,
                    "face_id": sf.id
                })
        logger.info(f"Photo {photo_id}: returning {len(results)} stored face locations")
        return results

    # Fallback: re-detect faces using InsightFace
    try:
        logger.info(f"Photo {photo_id}: no stored locations, re-detecting faces...")
        image = load_image(photo.file_path)
        faces = detect_faces(image)

        results = []
        for face_data in faces:
            loc = face_data['location']
            enc = np.array(face_data['embedding'])

            best_match_id = None
            max_sim = 0.0

            for sf in stored_faces:
                sf_enc = np.array(sf.encoding)
                sim = rec_compute_similarity(sf_enc, enc)
                if sim > 0.4 and sim > max_sim:
                    max_sim = sim
                    best_match_id = sf.id

            results.append({
                "location": loc,
                "face_id": best_match_id
            })

        return results
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/by-id")
def search_by_face_id(
    face_id: int = Query(...),
    threshold: float = 0.4,  # InsightFace similarity threshold
    db: Session = Depends(get_db)
):
    """
    Search for similar faces using an existing face_id from the database.
    Uses cosine similarity (InsightFace).
    """
    target_face = db.query(FaceEncoding).filter(FaceEncoding.id == face_id).first()
    if not target_face:
        raise HTTPException(status_code=404, detail="Face ID not found")

    target_encoding = np.array(target_face.encoding)

    # Fetch all known encodings
    known_faces = db.query(FaceEncoding).all()

    logger.info(f"Searching for Face ID {face_id}. Total faces in DB: {len(known_faces)}")

    matches = []
    seen_photos = {}

    for face_entry in known_faces:
        encoding = np.array(face_entry.encoding)
        similarity = rec_compute_similarity(encoding, target_encoding)

        if similarity >= threshold:
            photo_id = face_entry.photo_id
            # Keep only best match per photo
            if photo_id not in seen_photos or similarity > seen_photos[photo_id]:
                seen_photos[photo_id] = similarity
                existing = next((m for m in matches if m["photo_id"] == photo_id), None)
                if existing:
                    existing["similarity"] = similarity
                    existing["distance"] = 1 - similarity
                else:
                    matches.append({
                        "photo_id": face_entry.photo.id,
                        "file_path": face_entry.photo.file_path,
                        "filename": face_entry.photo.filename,
                        "capture_date": face_entry.photo.capture_date,
                        "similarity": float(similarity),
                        "distance": float(1 - similarity)
                    })

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(f"Found {len(matches)} matches")
    return {"count": len(matches), "matches": matches}


# ==================== PERSONS ENDPOINTS ====================

@app.get("/persons")
def get_all_persons(db: Session = Depends(get_db)):
    """
    Lista todas as pessoas identificadas com foto representativa e contagem.
    """
    persons = db.query(Person).all()

    results = []
    for person in persons:
        face_count = len(person.faces)
        if face_count == 0:
            continue

        rep_face = get_person_representative_face(person, db)

        results.append({
            "id": person.id,
            "name": person.name or f"Pessoa {person.id}",
            "face_count": face_count,
            "representative_face_id": rep_face.id if rep_face else None,
            "created_at": person.created_at,
            "group_id": person.group_id,
            "group_name": person.group.name if person.group else None,
            "group_color": person.group.color if person.group else None,
            "group_icon": getattr(person.group, "icon", None) if person.group else None
        })

    results.sort(key=lambda x: x["face_count"], reverse=True)
    return results


@app.get("/persons/suggestions")
def get_merge_suggestions(threshold: float = 0.35, db: Session = Depends(get_db)):
    """
    Retorna sugestões de mesclagem entre pessoas.
    Usa similaridade de cosseno (InsightFace).
    """
    persons = db.query(Person).all()

    person_data = []
    for person in persons:
        faces = [f for f in person.faces if f.encoding]
        if not faces:
            continue

        encodings = np.array([f.encoding for f in faces])
        centroid = np.mean(encodings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

        rep_face = get_person_representative_face(person, db)
        person_data.append({
            "id": person.id,
            "name": person.name or f"Pessoa {person.id}",
            "face_count": len(faces),
            "centroid": centroid,
            "rep_face_id": rep_face.id if rep_face else None
        })

    suggestions = []
    for i in range(len(person_data)):
        for j in range(i + 1, len(person_data)):
            p1 = person_data[i]
            p2 = person_data[j]

            similarity = float(np.dot(p1["centroid"], p2["centroid"]))

            if similarity >= threshold:
                suggestions.append({
                    "person1": {
                        "id": p1["id"],
                        "name": p1["name"],
                        "face_count": p1["face_count"],
                        "rep_face_id": p1["rep_face_id"]
                    },
                    "person2": {
                        "id": p2["id"],
                        "name": p2["name"],
                        "face_count": p2["face_count"],
                        "rep_face_id": p2["rep_face_id"]
                    },
                    "similarity": int(similarity * 100),
                    "distance": float(1 - similarity)
                })

    suggestions.sort(key=lambda x: x["similarity"], reverse=True)
    return {"suggestions": suggestions, "count": len(suggestions)}


@app.get("/persons/{person_id}")
def get_person_detail(
    person_id: int,
    limit: Optional[int] = Query(None, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Retorna detalhes de uma pessoa específica.
    """
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    faces = db.query(FaceEncoding).join(Photo).filter(
        FaceEncoding.person_id == person_id
    ).order_by(
        Photo.capture_date.desc().nullslast(),
        Photo.id.desc()
    ).all()
    photos = []
    seen_photos = set()
    updated = False
    unique_index = 0

    all_photo_ids = {face.photo_id for face in faces}
    photo_count = len(all_photo_ids)

    for face in faces:
        if face.photo_id in seen_photos:
            continue
        if unique_index < offset:
            seen_photos.add(face.photo_id)
            unique_index += 1
            continue
        if limit is not None and len(photos) >= limit:
            break
        seen_photos.add(face.photo_id)
        unique_index += 1

        photo_obj = face.photo
        photo_width = photo_obj.width
        photo_height = photo_obj.height

        if (photo_width is None or photo_height is None) and os.path.exists(photo_obj.file_path):
            try:
                from PIL import Image
                with Image.open(photo_obj.file_path) as img:
                    photo_width, photo_height = img.size
                photo_obj.width = photo_width
                photo_obj.height = photo_height
                updated = True
            except Exception:
                pass

        photos.append({
            "photo_id": photo_obj.id,
            "face_id": face.id,
            "file_path": photo_obj.file_path,
            "filename": photo_obj.filename,
            "capture_date": photo_obj.capture_date,
            "face_location": face.face_location,
            "width": photo_width,
            "height": photo_height
        })

    if updated:
        db.commit()

    has_more = (offset + len(photos)) < photo_count

    return {
        "id": person.id,
        "name": person.name or f"Pessoa {person.id}",
        "face_count": len(faces),
        "photo_count": photo_count,
        "photos": photos,
        "created_at": person.created_at,
        "group_id": person.group_id,
        "group_name": person.group.name if person.group else None,
        "group_icon": getattr(person.group, "icon", None) if person.group else None,
        "has_more": has_more,
        "offset": offset,
        "limit": limit
    }


class PersonUpdateRequest(BaseModel):
    name: str


@app.put("/persons/{person_id}")
def update_person(person_id: int, request: PersonUpdateRequest, db: Session = Depends(get_db)):
    """
    Atualiza o nome de uma pessoa.
    """
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    person.name = request.name
    db.commit()

    return {"success": True, "id": person.id, "name": person.name}


class ExcludeRequest(BaseModel):
    face_id: int


@app.post("/persons/{person_id}/exclude")
def exclude_face(person_id: int, request: ExcludeRequest, db: Session = Depends(get_db)):
    """
    Marca que um rosto NÃO pertence a esta pessoa (feedback negativo).
    """
    result = exclude_face_from_person(request.face_id, person_id, db)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


class MergeRequest(BaseModel):
    source_id: int
    target_id: int


@app.post("/persons/merge")
def merge_persons_endpoint(request: MergeRequest, db: Session = Depends(get_db)):
    """
    Mescla duas pessoas em uma (move rostos da source para target).
    """
    result = merge_persons(request.source_id, request.target_id, db)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/cluster")
def trigger_clustering(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Re-agrupa todos os rostos do banco de dados.
    """
    background_tasks.add_task(recluster_all_faces, db)
    return {"message": "Re-clustering started in background"}


# ==================== GROUPS ====================

class GroupCreate(BaseModel):
    name: str
    color: Optional[str] = None
    icon: Optional[str] = None


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None


class CacheSettings(BaseModel):
    max_cache_gb: Optional[float] = None
    webp_quality: Optional[int] = None


class CacheWarmupRequest(BaseModel):
    photo_ids: List[int]
    size: Optional[int] = None


class CameraConfig(BaseModel):
    name: str
    rtsp_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    quality: Optional[str] = None
    low_latency: Optional[bool] = True
    backend_enabled: Optional[bool] = False
    capture_enabled: Optional[bool] = False
    capture_location: Optional[str] = "data"
    capture_path: Optional[str] = None


class CameraTestRequest(BaseModel):
    rtsp_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    quality: Optional[str] = None
    low_latency: Optional[bool] = True


@app.get("/cameras")
def list_saved_cameras():
    """List saved RTSP cameras."""
    return load_saved_cameras()


@app.post("/cameras")
def add_saved_camera(camera: CameraConfig):
    """Add a saved RTSP camera config."""
    cameras = load_saved_cameras()
    new_camera = {
        "id": str(uuid.uuid4()),
        "name": camera.name.strip() or "Camera",
        "rtsp_url": camera.rtsp_url.strip(),
        "username": camera.username,
        "password": camera.password,
        "quality": camera.quality,
        "low_latency": True if camera.low_latency is None else camera.low_latency,
        "backend_enabled": bool(camera.backend_enabled),
        "capture_enabled": bool(camera.capture_enabled),
        "capture_location": camera.capture_location or "data",
        "capture_path": camera.capture_path,
        "created_at": datetime.utcnow().isoformat()
    }
    cameras.append(new_camera)
    save_saved_cameras(cameras)
    sync_backend_workers()
    return new_camera


@app.delete("/cameras/{camera_id}")
def delete_saved_camera(camera_id: str):
    """Delete a saved RTSP camera config."""
    cameras = load_saved_cameras()
    updated = [cam for cam in cameras if cam.get("id") != camera_id]
    if len(updated) == len(cameras):
        raise HTTPException(status_code=404, detail="Camera not found")
    save_saved_cameras(updated)
    sync_backend_workers()
    return {"success": True}


@app.put("/cameras/{camera_id}")
def update_saved_camera(camera_id: str, camera: CameraConfig):
    """Update a saved RTSP camera config."""
    cameras = load_saved_cameras()
    updated = None
    for cam in cameras:
        if cam.get("id") == camera_id:
            cam.update({
                "name": camera.name.strip() or cam.get("name", "Camera"),
                "rtsp_url": camera.rtsp_url.strip(),
                "username": camera.username,
                "password": camera.password,
                "quality": camera.quality,
                "low_latency": True if camera.low_latency is None else camera.low_latency,
                "backend_enabled": bool(camera.backend_enabled),
                "capture_enabled": bool(camera.capture_enabled),
                "capture_location": camera.capture_location or "data",
                "capture_path": camera.capture_path
            })
            updated = cam
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Camera not found")

    save_saved_cameras(cameras)
    sync_backend_workers()
    return updated


@app.post("/cameras/test")
def test_camera(camera: CameraTestRequest):
    """Test RTSP camera connection and return a snapshot."""
    rtsp_url = build_rtsp_with_credentials(camera.rtsp_url, camera.username, camera.password)
    if camera.low_latency:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    apply_capture_settings(cap, VideoSourceRequest(
        source=rtsp_url,
        quality=camera.quality,
        low_latency=camera.low_latency
    ))

    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        raise HTTPException(status_code=400, detail="Não foi possível abrir a câmera")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Não foi possível ler um frame")

    ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem")

    encoded = base64.b64encode(buffer.tobytes()).decode('ascii')
    return {
        "success": True,
        "image": f"data:image/jpeg;base64,{encoded}"
    }


@app.post("/cameras/preview/start")
def start_camera_preview(camera: CameraTestRequest):
    """Start a preview stream for RTSP camera."""
    rtsp_url = build_rtsp_with_credentials(camera.rtsp_url, camera.username, camera.password)
    if camera.low_latency:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    if preview_state["cap"]:
        preview_state["cap"].release()

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    apply_capture_settings(cap, VideoSourceRequest(
        source=rtsp_url,
        quality=camera.quality,
        low_latency=camera.low_latency
    ))

    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        raise HTTPException(status_code=400, detail="Não foi possível abrir a câmera")

    preview_state["active"] = True
    preview_state["cap"] = cap
    return {"success": True}


@app.post("/cameras/preview/stop")
def stop_camera_preview():
    if preview_state["cap"]:
        preview_state["cap"].release()
    preview_state["active"] = False
    preview_state["cap"] = None
    return {"success": True}


@app.get("/cameras/preview/feed")
def preview_feed():
    if not preview_state["active"] or not preview_state["cap"]:
        raise HTTPException(status_code=400, detail="Preview not started")

    return StreamingResponse(
        generate_video_stream(preview_state),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/groups")
def list_groups(db: Session = Depends(get_db)):
    """Lista todos os grupos de pessoas."""
    groups = db.query(PersonGroup).all()
    result = []
    for g in groups:
        result.append({
            "id": g.id,
            "name": g.name,
            "color": g.color,
            "icon": getattr(g, "icon", None),
            "person_count": len(g.persons)
        })
    return result


@app.post("/groups")
def create_group(group: GroupCreate, db: Session = Depends(get_db)):
    """Cria um novo grupo de pessoas."""
    new_group = PersonGroup(name=group.name, color=group.color, icon=group.icon)
    db.add(new_group)
    db.commit()
    db.refresh(new_group)
    return {
        "id": new_group.id,
        "name": new_group.name,
        "color": new_group.color,
        "icon": getattr(new_group, "icon", None)
    }


@app.put("/groups/{group_id}")
def update_group(group_id: int, group: GroupUpdate, db: Session = Depends(get_db)):
    """Atualiza um grupo existente."""
    db_group = db.query(PersonGroup).filter(PersonGroup.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Group not found")

    if group.name is not None:
        db_group.name = group.name
    if group.color is not None:
        db_group.color = group.color
    if group.icon is not None:
        db_group.icon = group.icon

    db.commit()
    return {
        "id": db_group.id,
        "name": db_group.name,
        "color": db_group.color,
        "icon": getattr(db_group, "icon", None)
    }


@app.delete("/groups/{group_id}")
def delete_group(group_id: int, db: Session = Depends(get_db)):
    """Deleta um grupo (pessoas não são deletadas, apenas desvinculadas)."""
    db_group = db.query(PersonGroup).filter(PersonGroup.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Group not found")

    # Remove grupo das pessoas
    for person in db_group.persons:
        person.group_id = None

    db.delete(db_group)
    db.commit()
    return {"success": True}


@app.post("/persons/{person_id}/group")
def assign_person_to_group(person_id: int, group_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Atribui uma pessoa a um grupo (ou remove do grupo se group_id=None)."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    if group_id is not None:
        group = db.query(PersonGroup).filter(PersonGroup.id == group_id).first()
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

    person.group_id = group_id
    db.commit()
    return {"success": True, "person_id": person_id, "group_id": group_id}


@app.get("/groups/{group_id}/persons")
def get_group_persons(group_id: int, db: Session = Depends(get_db)):
    """Lista todas as pessoas de um grupo."""
    group = db.query(PersonGroup).filter(PersonGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    result = []
    for person in group.persons:
        rep_face = get_person_representative_face(person, db)
        result.append({
            "id": person.id,
            "name": person.name or f"Pessoa {person.id}",
            "face_count": len(person.faces),
            "representative_face_id": rep_face.id if rep_face else None
        })
    return result


@app.post("/reset")
def reset_database(db: Session = Depends(get_db)):
    """
    Limpa todo o historico do banco de dados.
    """
    try:
        db.query(CaptureEvent).delete()
        db.query(RecognitionHistory).delete()
        db.query(FaceExclusion).delete()
        db.query(FaceEncoding).delete()
        db.query(Photo).delete()
        db.query(Person).delete()
        db.commit()

        recognition_log_cache.clear()
        logger.info("Database reset complete")
        return {"success": True, "message": "Histórico limpo com sucesso"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats_v2(db: Session = Depends(get_db)):
    """
    Returns current statistics including persons count.
    """
    photo_count = db.query(Photo).count()
    face_count = db.query(FaceEncoding).count()
    person_count = db.query(Person).count()
    current_status = get_status()

    return {
        "photos_indexed": photo_count,
        "faces_indexed": face_count,
        "persons_identified": person_count,
        "status": current_status["state"],
        "current_detail": current_status["current_file"]
    }


@app.get("/bing-background")
def get_bing_background():
    """
    Proxy endpoint to fetch Bing daily image URL (avoids CORS issues).
    """
    import urllib.request
    import json

    try:
        url = "https://www.bing.com/HPImageArchive.aspx?format=js&idx=0&n=1&mkt=pt-BR"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("images") and len(data["images"]) > 0:
                image_url = "https://www.bing.com" + data["images"][0]["url"]
                return {"url": image_url, "copyright": data["images"][0].get("copyright", "")}
    except Exception as e:
        logger.warning(f"Could not fetch Bing background: {e}")

    return {"url": None, "error": "Could not fetch Bing image"}


@app.get("/face/{face_id}/crop")
def get_face_crop(face_id: int, db: Session = Depends(get_db)):
    """
    Returns a cropped image of just the face.
    """
    from PIL import Image, ExifTags
    from io import BytesIO

    face = db.query(FaceEncoding).filter(FaceEncoding.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    if not face.face_location:
        raise HTTPException(status_code=400, detail="Face location not available")

    try:
        img = Image.open(face.photo.file_path)

        # Get EXIF orientation
        orientation_value = None
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
        except (AttributeError, KeyError, IndexError):
            pass

        # Get face location [top, right, bottom, left]
        top, right, bottom, left = face.face_location

        # Add padding
        height = bottom - top
        width = right - left
        padding_h = int(height * 0.4)
        padding_w = int(width * 0.4)

        crop_left = max(0, left - padding_w)
        crop_top = max(0, top - padding_h)
        crop_right = min(img.width, right + padding_w)
        crop_bottom = min(img.height, bottom + padding_h)

        face_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Apply EXIF rotation
        if orientation_value == 3:
            face_img = face_img.rotate(180, expand=True)
        elif orientation_value == 6:
            face_img = face_img.rotate(270, expand=True)
        elif orientation_value == 8:
            face_img = face_img.rotate(90, expand=True)

        face_img = face_img.resize((200, 200), Image.LANCZOS)

        buffer = BytesIO()
        face_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error cropping face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/face/{face_id}/thumb")
def get_face_thumb(face_id: int, size: int = 80, db: Session = Depends(get_db)):
    """
    Returns a cached WebP thumbnail for a face crop.
    """
    face = db.query(FaceEncoding).filter(FaceEncoding.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    if not face.face_location:
        raise HTTPException(status_code=400, detail="Face location not available")

    size = max(60, min(size, 240))
    cache_settings = load_cache_settings()
    cache_path = build_cache_path("faces", str(face_id), size)

    if cache_path.exists():
        touch_cache_file(cache_path)
        return FileResponse(str(cache_path), media_type="image/webp")

    from PIL import Image, ExifTags

    try:
        img = Image.open(face.photo.file_path)

        orientation_value = None
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
        except (AttributeError, KeyError, IndexError):
            pass

        top, right, bottom, left = face.face_location
        height = bottom - top
        width = right - left
        padding_h = int(height * 0.4)
        padding_w = int(width * 0.4)

        crop_left = max(0, left - padding_w)
        crop_top = max(0, top - padding_h)
        crop_right = min(img.width, right + padding_w)
        crop_bottom = min(img.height, bottom + padding_h)

        face_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        if orientation_value == 3:
            face_img = face_img.rotate(180, expand=True)
        elif orientation_value == 6:
            face_img = face_img.rotate(270, expand=True)
        elif orientation_value == 8:
            face_img = face_img.rotate(90, expand=True)

        face_img = face_img.resize((size, size), Image.LANCZOS)
        save_webp(face_img, cache_path, int(cache_settings["webp_quality"]))
        enforce_cache_limit()
        return FileResponse(str(cache_path), media_type="image/webp")
    except Exception as e:
        logger.error(f"Error creating face thumbnail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thumb/{photo_id}")
def get_photo_thumb(photo_id: int, size: int = 480, db: Session = Depends(get_db)):
    """
    Returns a cached WebP thumbnail for a photo.
    """
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    size = max(160, min(size, 1280))
    cache_settings = load_cache_settings()
    cache_path = build_cache_path("photos", str(photo_id), size)

    if cache_path.exists():
        touch_cache_file(cache_path)
        return FileResponse(str(cache_path), media_type="image/webp")

    try:
        thumb_image = create_thumbnail(photo.file_path, size)
        save_webp(thumb_image, cache_path, int(cache_settings["webp_quality"]))
        enforce_cache_limit()
        return FileResponse(str(cache_path), media_type="image/webp")
    except Exception as e:
        logger.error(f"Error creating photo thumbnail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/settings")
def get_cache_settings_endpoint():
    """
    Returns current cache settings.
    """
    return load_cache_settings()


@app.post("/cache/settings")
def set_cache_settings(settings: CacheSettings):
    """
    Updates cache settings for WebP caching.
    """
    updated = update_cache_settings({
        "max_cache_gb": settings.max_cache_gb,
        "webp_quality": settings.webp_quality
    })
    enforce_cache_limit()
    return updated


# ==================== CACHE WARMUP ====================

@app.post("/cache/warmup")
def warm_cache(request: CacheWarmupRequest, db: Session = Depends(get_db)):
    """
    Warm cache for a list of photo IDs.
    """
    photo_ids = request.photo_ids
    if not photo_ids:
        return {"warmed": 0}

    size = request.size or 480
    size = max(160, min(size, 1280))
    cache_settings = load_cache_settings()
    warmed = 0

    for photo_id in photo_ids[:200]:
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            continue
        cache_path = build_cache_path("photos", str(photo_id), size)
        if cache_path.exists():
            touch_cache_file(cache_path)
            warmed += 1
            continue
        try:
            thumb_image = create_thumbnail(photo.file_path, size)
            save_webp(thumb_image, cache_path, int(cache_settings["webp_quality"]))
            warmed += 1
        except Exception as e:
            logger.warning(f"Cache warmup failed for {photo_id}: {e}")

    enforce_cache_limit()
    return {"warmed": warmed}


# ==================== VIDEO STREAMING ====================

# Global video state
video_state = {
    "active": False,
    "source": None,
    "cap": None,
    "capture_enabled": False,
    "capture_dir": None,
    "capture_last_time": None,
    "capture_last_label": None,
    "lock": threading.Lock()
}

preview_state = {
    "active": False,
    "cap": None,
    "lock": threading.Lock()
}

backend_workers = {}

recognition_log_cache = {}
RECOGNITION_LOG_INTERVAL_SEC = 10
MAX_RECOGNITION_HISTORY = 1000
CAPTURE_INTERVAL_SEC = 5


def prune_recognition_history(db: Session) -> None:
    if db.query(RecognitionHistory).count() <= MAX_RECOGNITION_HISTORY:
        return
    ids_to_remove = db.query(RecognitionHistory.id).order_by(
        RecognitionHistory.created_at.desc()
    ).offset(MAX_RECOGNITION_HISTORY).all()
    if ids_to_remove:
        db.query(RecognitionHistory).filter(
            RecognitionHistory.id.in_([row[0] for row in ids_to_remove])
        ).delete(synchronize_session=False)
        db.commit()


def log_recognition_event(db: Session, person: Optional[Person], similarity: float, timestamp: Optional[datetime] = None) -> None:
    if person is None:
        return
    now = timestamp or datetime.utcnow()
    last_seen = recognition_log_cache.get(person.id)
    if last_seen and (now - last_seen).total_seconds() < RECOGNITION_LOG_INTERVAL_SEC:
        return
    recognition_log_cache[person.id] = now
    entry = RecognitionHistory(
        person_id=person.id,
        person_name=person.name or f"Pessoa {person.id}",
        similarity=float(similarity),
        recognized=True,
        created_at=now
    )
    db.add(entry)
    db.commit()
    prune_recognition_history(db)


def normalize_capture_label(value: str) -> str:
    safe = "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))
    return safe[:60] if safe else "unknown"


def select_capture_label(results: List[dict]) -> str:
    best = None
    for item in results:
        if item.get("recognized") and item.get("person_name"):
            if best is None or item.get("similarity", 0) > best.get("similarity", 0):
                best = item
    if best:
        return normalize_capture_label(best.get("person_name", "unknown"))
    return "desconhecido"


def maybe_save_capture(frame: np.ndarray, results: List[dict], state: Optional[dict] = None, db: Optional[Session] = None) -> None:
    target = state or video_state
    if not target.get("capture_enabled") or not target.get("capture_dir"):
        return

    if not results:
        target["capture_last_label"] = None
        target["capture_last_time"] = None
        return

    label = select_capture_label(results)
    last_label = target.get("capture_last_label")

    now = datetime.utcnow()
    last_time = target.get("capture_last_time")
    if last_label == label and last_time and (now - last_time).total_seconds() < CAPTURE_INTERVAL_SEC:
        return

    target["capture_last_time"] = now
    target["capture_last_label"] = label
    day_folder = now.strftime("%Y-%m-%d")
    capture_dir = os.path.join(target["capture_dir"], day_folder)
    os.makedirs(capture_dir, exist_ok=True)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{label}.jpg"
    path = os.path.join(capture_dir, filename)
    try:
        cv2.imwrite(path, frame)
        if db:
            best_person = None
            best_similarity = 0.0
            for item in results:
                if item.get("recognized") and item.get("person_id"):
                    similarity = item.get("similarity", 0)
                    if similarity >= best_similarity:
                        best_similarity = similarity
                        best_person = item

            event = CaptureEvent(
                person_id=best_person.get("person_id") if best_person else None,
                person_name=best_person.get("person_name") if best_person else None,
                label=label,
                file_path=path,
                created_at=now
            )
            db.add(event)
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to save capture: {e}")


class VideoSourceRequest(BaseModel):
    source: str  # "webcam", "rtsp://...", or IP address
    quality: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    low_latency: Optional[bool] = True
    camera_id: Optional[str] = None


VIDEO_QUALITY_PRESETS = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}


def apply_capture_settings(cap: cv2.VideoCapture, request: VideoSourceRequest) -> None:
    if not cap:
        return

    width = request.width
    height = request.height
    if request.quality in VIDEO_QUALITY_PRESETS:
        width, height = VIDEO_QUALITY_PRESETS[request.quality]

    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if request.fps:
        cap.set(cv2.CAP_PROP_FPS, request.fps)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def build_rtsp_with_credentials(url: str, username: Optional[str], password: Optional[str]) -> str:
    if not username and not password:
        return url
    if "@" in url:
        return url
    parts = url.split("://", 1)
    if len(parts) != 2:
        return url
    user = username or ""
    pwd = password or ""
    auth = user
    if pwd:
        auth = f"{user}:{pwd}"
    return f"{parts[0]}://{auth}@{parts[1]}"


def recognize_faces_in_frame(db: Session, frame: np.ndarray) -> List[dict]:
    faces = detect_faces(frame)
    results = []
    known_faces = db.query(FaceEncoding).all()

    for face_data in faces:
        embedding = np.array(face_data['embedding'])
        best_match = None
        best_similarity = 0.0

        for kf in known_faces:
            if kf.encoding:
                sim = rec_compute_similarity(np.array(kf.encoding), embedding)
                if sim > best_similarity:
                    best_similarity = sim
                    if sim >= 0.5 and kf.person:
                        best_match = kf.person

        results.append({
            "location": face_data['location'],
            "person_id": best_match.id if best_match else None,
            "person_name": (best_match.name or f"Pessoa {best_match.id}") if best_match else None,
            "similarity": float(best_similarity) if best_match else 0,
            "recognized": best_match is not None,
            "group_id": best_match.group_id if best_match else None
        })

    return results


def detect_motion(prev_gray: Optional[np.ndarray], frame: np.ndarray) -> tuple[bool, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_gray is None:
        return False, gray
    frame_delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_ratio = float(np.sum(thresh > 0)) / float(thresh.size)
    return motion_ratio > 0.01, gray


def run_backend_camera(camera: dict, stop_event: threading.Event) -> None:
    rtsp_url = build_rtsp_with_credentials(camera.get("rtsp_url", ""), camera.get("username"), camera.get("password"))
    if not rtsp_url:
        return

    capture_dir = None
    if camera.get("capture_enabled"):
        capture_dir = resolve_capture_dir(camera.get("capture_location"), camera.get("capture_path"))
        try:
            os.makedirs(capture_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create capture dir {capture_dir}: {e}")
            capture_dir = None

    capture_state = {
        "capture_enabled": bool(camera.get("capture_enabled")) and bool(capture_dir),
        "capture_dir": capture_dir,
        "capture_last_time": None,
        "capture_last_label": None
    }

    idle_interval = 1.0
    active_interval = 0.5
    active_timeout = 3.0

    while not stop_event.is_set():
        if camera.get("low_latency", True):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        apply_capture_settings(cap, VideoSourceRequest(
            source=rtsp_url,
            quality=camera.get("quality"),
            low_latency=camera.get("low_latency", True)
        ))

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            time.sleep(5)
            continue

        last_gray = None
        last_motion_time = 0.0
        last_recognition_time = 0.0

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                small_frame = cv2.resize(frame, (640, 360))
                motion, last_gray = detect_motion(last_gray, small_frame)
                now = time.time()
                if motion:
                    last_motion_time = now

                is_active = (now - last_motion_time) < active_timeout
                interval = active_interval if is_active else idle_interval

                if now - last_recognition_time >= interval:
                    frame_for_recognition = frame if is_active else small_frame
                    db = SessionLocal()
                    try:
                        results = recognize_faces_in_frame(db, frame_for_recognition)
                        for item in results:
                            if item.get("recognized") and item.get("person_id"):
                                person = db.query(Person).filter(Person.id == item["person_id"]).first()
                                log_recognition_event(db, person, item.get("similarity", 0))
                        maybe_save_capture(frame, results, capture_state, db=db)
                    finally:
                        db.close()

                    last_recognition_time = now

                time.sleep(0.05)
        finally:
            cap.release()

    return


def camera_signature(camera: dict) -> tuple:
    return (
        camera.get("rtsp_url"),
        camera.get("username"),
        camera.get("password"),
        camera.get("quality"),
        camera.get("low_latency"),
        camera.get("capture_enabled"),
        camera.get("capture_location"),
        camera.get("capture_path")
    )


def sync_backend_workers():
    cameras = load_saved_cameras()
    enabled = {cam.get("id") for cam in cameras if cam.get("backend_enabled")}

    for cam_id in list(backend_workers.keys()):
        if cam_id not in enabled:
            worker = backend_workers.pop(cam_id)
            worker["stop"].set()

    for cam in cameras:
        cam_id = cam.get("id")
        if not cam_id or not cam.get("backend_enabled"):
            continue

        signature = camera_signature(cam)
        existing = backend_workers.get(cam_id)
        if existing and existing.get("signature") == signature:
            continue
        if existing:
            existing["stop"].set()
            backend_workers.pop(cam_id, None)

        stop_event = threading.Event()
        thread = threading.Thread(target=run_backend_camera, args=(cam, stop_event), daemon=True)
        backend_workers[cam_id] = {"thread": thread, "stop": stop_event, "signature": signature}
        thread.start()


class CameraDiscoverRequest(BaseModel):
    ip: str
    username: Optional[str] = None
    password: Optional[str] = None
    ports: Optional[List[int]] = None


def get_rtsp_url_from_ip(ip: str) -> Optional[str]:
    """
    Tenta descobrir URL RTSP de uma camera IP.
    Testa URLs comuns de fabricantes.
    """
    common_rtsp_urls = [
        f"rtsp://{ip}:554/stream1",
        f"rtsp://{ip}:554/live/ch00_0",
        f"rtsp://{ip}:554/h264_stream",
        f"rtsp://{ip}:554/cam/realmonitor?channel=1&subtype=0",
        f"rtsp://{ip}:554/Streaming/Channels/101",
        f"rtsp://{ip}/live/main",
        f"rtsp://admin:admin@{ip}:554/stream1",
    ]

    for url in common_rtsp_urls:
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return url
        except:
            pass

    return None


@app.post("/video/start")
def start_video(request: VideoSourceRequest, db: Session = Depends(get_db)):
    """
    Inicia captura de video de webcam ou camera IP.
    """
    global video_state

    if video_state["active"]:
        return {"error": "Video already active. Stop first."}

    source = request.source
    cap = None

    if source == "webcam" or source == "0":
        # Debug: List video devices
        import glob
        video_devices = glob.glob('/dev/video*')
        logger.info(f"Available video devices in container: {video_devices}")

        # Tenta indices comuns e backends
        params = [
            (0, cv2.CAP_V4L2),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_V4L2),
            (1, cv2.CAP_ANY),
            (2, cv2.CAP_V4L2)
        ]

        for idx, backend in params:
            logger.info(f"Trying to open webcam index {idx} with backend {backend}")
            candidate = cv2.VideoCapture(idx, backend)
            if candidate.isOpened():
                logger.info(f"Successfully opened webcam index {idx}")
                # Configura resolucao para performance
                if request.quality not in VIDEO_QUALITY_PRESETS and not request.width and not request.height:
                    candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                apply_capture_settings(candidate, request)
                cap = candidate
                break
            candidate.release()
    elif source.startswith("rtsp://"):
        if request.low_latency:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        apply_capture_settings(cap, request)
    else:
        # Assume it's an IP address, try to find RTSP URL
        rtsp_url = get_rtsp_url_from_ip(source)
        if not rtsp_url:
            return {"error": f"Could not find RTSP stream for IP {source}"}
        if request.low_latency:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        apply_capture_settings(cap, request)
        source = rtsp_url

    if not cap or not cap.isOpened():
        return {"error": f"Could not open video source: {source}"}

    video_state["active"] = True
    video_state["source"] = source
    video_state["cap"] = cap
    video_state["capture_enabled"] = False
    video_state["capture_dir"] = None
    video_state["capture_last_time"] = None
    video_state["capture_last_label"] = None

    if request.camera_id:
        cameras = load_saved_cameras()
        camera = next((cam for cam in cameras if cam.get("id") == request.camera_id), None)
        if camera and camera.get("capture_enabled"):
            capture_dir = resolve_capture_dir(camera.get("capture_location"), camera.get("capture_path"))
            try:
                os.makedirs(capture_dir, exist_ok=True)
                video_state["capture_enabled"] = True
                video_state["capture_dir"] = capture_dir
            except Exception as e:
                logger.warning(f"Could not enable capture dir {capture_dir}: {e}")

    return {"success": True, "source": source}


@app.post("/video/stop")
def stop_video():
    """
    Para a captura de video.
    """
    global video_state

    if video_state["cap"]:
        video_state["cap"].release()

    video_state["active"] = False
    video_state["source"] = None
    video_state["cap"] = None
    video_state["capture_enabled"] = False
    video_state["capture_dir"] = None
    video_state["capture_last_time"] = None
    video_state["capture_last_label"] = None

    return {"success": True}


@app.get("/video/status")
def video_status():
    """
    Retorna status do video.
    """
    return {
        "active": video_state["active"],
        "source": video_state["source"]
    }


def generate_video_frames(db: Session):
    """
    Gera frames de video com deteccao de faces.
    """
    global video_state

    while video_state["active"] and video_state["cap"]:
        with video_state["lock"]:
            ret, frame = video_state["cap"].read()
        if not ret:
            break

        # Detecta faces no frame
        try:
            faces = detect_faces(frame)

            # Busca correspondencias no banco
            for face_data in faces:
                top, right, bottom, left = face_data['location']
                embedding = np.array(face_data['embedding'])

                # Busca pessoa correspondente
                best_match = None
                best_similarity = 0.0

                persons = db.query(Person).all()
                for person in persons:
                    for pf in person.faces:
                        if pf.encoding:
                            pf_enc = np.array(pf.encoding)
                            sim = rec_compute_similarity(pf_enc, embedding)
                            if sim > best_similarity and sim >= 0.5:
                                best_similarity = sim
                                best_match = person

                # Desenha retangulo
                color = (0, 255, 0) if best_match else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Adiciona nome se reconhecido
                if best_match:
                    name = best_match.name or f"Pessoa {best_match.id}"
                    label = f"{name} ({int(best_similarity * 100)}%)"
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, "Desconhecido", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_video_stream(state: dict):
    while state["active"] and state["cap"]:
        with state["lock"]:
            ret, frame = state["cap"].read()
        if not ret:
            break

        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video/feed")
def video_feed(db: Session = Depends(get_db)):
    """
    Stream de video MJPEG com deteccao de faces.
    """
    if not video_state["active"]:
        raise HTTPException(status_code=400, detail="Video not started")

    return StreamingResponse(
        generate_video_frames(db),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video/stream")
def video_stream():
    """Stream de video MJPEG sem processamento pesado."""
    if not video_state["active"]:
        raise HTTPException(status_code=400, detail="Video not started")

    return StreamingResponse(
        generate_video_stream(video_state),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/video/recognize")
def recognize_from_video(db: Session = Depends(get_db)):
    """
    Captura um frame e identifica as pessoas.
    """
    if not video_state["active"] or not video_state["cap"]:
        raise HTTPException(status_code=400, detail="Video not started")

    with video_state["lock"]:
        ret, frame = video_state["cap"].read()
    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame")

    faces = detect_faces(frame)
    results = []
    now = datetime.utcnow()

    for face_data in faces:
        embedding = np.array(face_data['embedding'])
        location = face_data['location']

        # Busca correspondencias
        best_match = None
        best_similarity = 0.0

        known_faces = db.query(FaceEncoding).all()
        for kf in known_faces:
            if kf.encoding:
                sim = rec_compute_similarity(np.array(kf.encoding), embedding)
                if sim > best_similarity:
                    best_similarity = sim
                    if sim >= 0.5 and kf.person:
                        best_match = kf.person

        results.append({
            "location": location,
            "person_id": best_match.id if best_match else None,
            "person_name": (best_match.name or f"Pessoa {best_match.id}") if best_match else None,
            "similarity": float(best_similarity) if best_match else 0,
            "recognized": best_match is not None,
            "group_id": best_match.group_id if best_match else None
        })

        log_recognition_event(db, best_match, best_similarity, now)

    maybe_save_capture(frame, results, db=db)

    return {"faces_detected": len(results), "results": results}


@app.get("/recognition/history")
def get_recognition_history(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Return recent recognition history."""
    entries = db.query(RecognitionHistory).order_by(RecognitionHistory.created_at.desc()).limit(limit).all()
    return [
        {
            "id": entry.id,
            "person_id": entry.person_id,
            "person_name": entry.person_name,
            "similarity": entry.similarity,
            "recognized": entry.recognized,
            "created_at": to_utc_iso(entry.created_at)
        }
        for entry in entries
    ]


@app.delete("/recognition/history")
def clear_recognition_history(db: Session = Depends(get_db)):
    """Clear recognition history."""
    db.query(RecognitionHistory).delete()
    db.commit()
    recognition_log_cache.clear()
    return {"success": True}


@app.get("/captures/recent")
def get_recent_captures(limit: int = Query(5, ge=1, le=50), db: Session = Depends(get_db)):
    entries = db.query(CaptureEvent).order_by(CaptureEvent.created_at.desc()).limit(limit).all()
    return [
        {
            "id": entry.id,
            "person_id": entry.person_id,
            "person_name": entry.person_name,
            "label": entry.label,
            "file_path": entry.file_path,
            "created_at": to_utc_iso(entry.created_at)
        }
        for entry in entries
    ]


# ==================== PHOTO OPERATIONS ====================

class PhotoOperationRequest(BaseModel):
    photo_ids: List[int]
    destination: str = ""
    mode: str = "copy"  # copy or move


@app.post("/video/process_frame")
async def process_frame(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Recebe um frame (imagem) enviado pelo navegador e realiza reconhecimento facial.
    Nao salva a imagem em disco.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    faces = detect_faces(frame)
    results = []
    now = datetime.utcnow()

    for face_data in faces:
        embedding = np.array(face_data['embedding'])
        best_match = None
        best_similarity = 0.0

        known_faces = db.query(FaceEncoding).all()
        for kf in known_faces:
            if kf.encoding:
                sim = rec_compute_similarity(np.array(kf.encoding), embedding)
                if sim > best_similarity:
                    best_similarity = sim
                    if sim >= 0.5 and kf.person:
                        best_match = kf.person

        results.append({
            "location": face_data['location'],
            "person_id": best_match.id if best_match else None,
            "person_name": (best_match.name or f"Pessoa {best_match.id}") if best_match else None,
            "similarity": float(best_similarity) if best_match else 0,
            "recognized": best_match is not None,
            "group_id": best_match.group_id if best_match else None
        })

        log_recognition_event(db, best_match, best_similarity, now)

    maybe_save_capture(frame, results, db=db)

    return {"faces_detected": len(results), "results": results}


@app.post("/photos/download")
def download_photos(request: PhotoOperationRequest, db: Session = Depends(get_db)):
    """
    Cria um arquivo ZIP com as fotos selecionadas.
    """
    import zipfile
    from io import BytesIO
    from datetime import datetime

    photos = db.query(Photo).filter(Photo.id.in_(request.photo_ids)).all()

    if not photos:
        raise HTTPException(status_code=404, detail="No photos found")

    # Cria ZIP em memoria
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for photo in photos:
            if os.path.exists(photo.file_path):
                # Usa o nome original do arquivo
                zip_file.write(photo.file_path, photo.filename)

    zip_buffer.seek(0)

    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photos_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/photos/copy-move")
def copy_move_photos(request: PhotoOperationRequest, db: Session = Depends(get_db)):
    """
    Copia ou move fotos para uma pasta de destino.
    """
    if not request.destination:
        raise HTTPException(status_code=400, detail="Destination folder required")

    # Destino relativo ao PHOTOS_DIR
    dest_path = os.path.join(PHOTOS_DIR, request.destination)

    # Cria pasta se nao existir
    os.makedirs(dest_path, exist_ok=True)

    photos = db.query(Photo).filter(Photo.id.in_(request.photo_ids)).all()

    if not photos:
        raise HTTPException(status_code=404, detail="No photos found")

    results = {"success": 0, "failed": 0, "files": []}

    for photo in photos:
        try:
            src = photo.file_path
            dst = os.path.join(dest_path, photo.filename)

            # Evita sobrescrever
            if os.path.exists(dst):
                base, ext = os.path.splitext(photo.filename)
                counter = 1
                while os.path.exists(dst):
                    dst = os.path.join(dest_path, f"{base}_{counter}{ext}")
                    counter += 1

            if request.mode == "move":
                shutil.move(src, dst)
                # Atualiza caminho no banco
                photo.file_path = dst
                db.commit()
            else:
                shutil.copy2(src, dst)

            results["success"] += 1
            results["files"].append(os.path.basename(dst))
        except Exception as e:
            logger.error(f"Error {request.mode} {photo.file_path}: {e}")
            results["failed"] += 1

    return results


@app.get("/photos/destinations")
def list_destinations():
    """
    Lista pastas disponiveis como destino.
    """
    destinations = []

    def scan_dirs(path, prefix=""):
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    rel_path = os.path.join(prefix, item) if prefix else item
                    destinations.append(rel_path)
                    # Scan subpastas (limite de profundidade)
                    if prefix.count(os.sep) < 2:
                        scan_dirs(item_path, rel_path)
        except PermissionError:
            pass

    scan_dirs(PHOTOS_DIR)
    destinations.sort()

    return {"destinations": destinations}


@app.post("/cameras/discover")
def discover_cameras(request: CameraDiscoverRequest):
    """
    Tenta descobrir cameras ONVIF na rede local com credenciais fornecidas.
    """
    cameras = []
    ip = request.ip
    username = request.username or "admin"
    password = request.password or ""
    ports = request.ports or [80, 8899, 8000, 8080, 5000]

    def with_auth(rtsp_url: str) -> str:
        if not username:
            return rtsp_url
        if rtsp_url.startswith("rtsp://") and "@" not in rtsp_url:
            return rtsp_url.replace("rtsp://", f"rtsp://{username}:{password}@", 1)
        return rtsp_url

    try:
        from onvif import ONVIFCamera

        for port in ports:
            try:
                cam = ONVIFCamera(ip, port, username, password)
                device_info = cam.devicemgmt.GetDeviceInformation()

                media = cam.create_media_service()
                profiles = media.GetProfiles()
                if profiles:
                    stream_uri = media.GetStreamUri({
                        'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                        'ProfileToken': profiles[0].token
                    })

                    cameras.append({
                        "ip": ip,
                        "name": device_info.Model,
                        "manufacturer": device_info.Manufacturer,
                        "rtsp_url": with_auth(stream_uri.Uri),
                        "onvif_port": port
                    })
                    break
            except Exception as e:
                logger.debug(f"Could not connect to {ip}:{port}: {e}")

        if not cameras:
            rtsp_url = get_rtsp_url_from_ip(ip)
            if rtsp_url:
                cameras.append({
                    "ip": ip,
                    "name": "Unknown Camera",
                    "manufacturer": "Unknown",
                    "rtsp_url": with_auth(rtsp_url)
                })
    except ImportError:
        logger.warning("onvif-zeep not installed, ONVIF discovery not available")

    return {"cameras": cameras}
