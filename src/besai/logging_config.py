import logging
import os

def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'besai.log')),
            logging.StreamHandler()
        ]
    )
    logging.config.dictConfig({
        'version': 1,
        'handlers': {
            'retry_handler': {
                'class': 'logging.FileHandler',
                'filename': 'logs/retry_processor.log',
                'formatter': 'detailed',
            }
        },
        'loggers': {
            'retry_processor': {
                'handlers': ['retry_handler'],
                'level': 'INFO',
            }
        }
    })
