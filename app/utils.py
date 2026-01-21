from PIL import Image, ExifTags
from datetime import datetime

def get_exif_data(image_path):
    """
    Extracts date and camera model from image EXIF.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return None, None

        exif = {
            ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in ExifTags.TAGS
        }

        # Get Date
        date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
        capture_date = None
        if date_str:
            try:
                capture_date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

        # Get Camera Model
        model = exif.get("Model")
        
        return capture_date, model

    except Exception:
        return None, None
