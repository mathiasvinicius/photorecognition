import os
import shutil
from datetime import datetime
from .utils import get_exif_data
import logging

logger = logging.getLogger(__name__)

def organize_photos(source_dir: str, dest_dir: str, mode: str = "copy"):
    """
    Organizes photos from source_dir to dest_dir based on EXIF date.
    mode: 'copy' or 'move'
    """
    logger.info(f"Starting organization: {mode} from {source_dir} to {dest_dir}")
    
    results = {
        "processed": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0 # No date found
    }

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            results["processed"] += 1
            
            try:
                # Get Date
                capture_date, _ = get_exif_data(file_path)
                
                if not capture_date:
                    # Fallback to file modification time? 
                    # User specifically asked for "Real photo date", implies EXIF. 
                    # We might skip or put in "Unknown_Date" folder.
                    # Let's put in "Unknown_Date"
                    folder_name = "Unknown_Date"
                    file_prefix = "Unknown"
                else:
                    folder_name = capture_date.strftime("%Y/%m") # YYYY/MM
                    file_prefix = capture_date.strftime("%Y-%m-%d")

                # Create Dest Folder
                target_folder = os.path.join(dest_dir, folder_name)
                os.makedirs(target_folder, exist_ok=True)

                # Generate Target Filename
                base, ext = os.path.splitext(file)
                new_filename = f"{file_prefix}_{base}{ext}"
                target_path = os.path.join(target_folder, new_filename)

                # Handle Duplicates
                counter = 1
                while os.path.exists(target_path):
                    new_filename = f"{file_prefix}_{base}_{counter}{ext}"
                    target_path = os.path.join(target_folder, new_filename)
                    counter += 1

                # Execute Action
                if mode == "move":
                    shutil.move(file_path, target_path)
                else:
                    shutil.copy2(file_path, target_path)
                
                results["success"] += 1

            except Exception as e:
                logger.error(f"Failed to organize {file_path}: {e}")
                results["failed"] += 1

    logger.info(f"Organization complete: {results}")
    return results
