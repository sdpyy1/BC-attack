import colorlog
import logging
def init_logger(Level,fileName):
    logger = logging.getLogger('run')
    logger.setLevel(Level)

    # 控制台handler（带颜色）
    console = logging.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
    )
    console.setFormatter(console_formatter)

    # 文件handler（不带颜色）
    file = logging.FileHandler(f'./{fileName}/run.log', encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file.setFormatter(file_formatter)

    logger.addHandler(console)
    logger.addHandler(file)
    return logger
