def install_nanodet():
    import os, kpick
    cwd = os.getcwd()
    print(f'Current workdir: {cwd}')
    HOME_DIR  = os.path.expanduser("~")
    NANODET_DIR = os.path.join(HOME_DIR, '000_nanodet')
    print(f'Nanodet dir: {NANODET_DIR}')

    link = 'https://github.com/keti-ai/nanodet'
    print(f'Nanodet was not installed yet. \nCloning from {link} into {NANODET_DIR}')
    if os.path.exists(NANODET_DIR):
        import shutil
        shutil.rmtree(NANODET_DIR)
    os.makedirs(NANODET_DIR)
    os.chdir(NANODET_DIR)
    os.system(f'git clone {link} .')

    print(f'Installing Nanodet...')
    os.system(f'pip install -e .')
    # os.system(f'python setup.py develop')

    os.system(f'cd {cwd}')
    print(f'Nanodet installed...')
    print('Program stopped. PLEASE RE-RUN YOUR CODE ...')
    exit()