# Simple in-memory status store
status = {
    "state": "Idle",  # Idle, Scanning, Organizing, Paused
    "current_file": "",
    "last_error": None,
    "paused": False,
    "stop_requested": False
}

def set_status(state, file=""):
    status["state"] = state
    status["current_file"] = file

def get_status():
    return status

def pause_scan():
    """Pausa o scan atual."""
    if status["state"] == "Scanning":
        status["paused"] = True
        status["state"] = "Paused"
        return True
    return False

def resume_scan():
    """Continua o scan pausado."""
    if status["paused"]:
        status["paused"] = False
        status["state"] = "Scanning"
        return True
    return False

def stop_scan():
    """Para o scan completamente."""
    status["stop_requested"] = True
    status["paused"] = False
    return True

def reset_scan_flags():
    """Reseta flags de controle."""
    status["paused"] = False
    status["stop_requested"] = False

def is_paused():
    """Verifica se o scan está pausado."""
    return status["paused"]

def should_stop():
    """Verifica se deve parar o scan."""
    return status["stop_requested"]
