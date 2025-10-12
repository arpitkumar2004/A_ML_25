"""Logging & experiment tracking helpers."""
def get_logger(name='project'):
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

