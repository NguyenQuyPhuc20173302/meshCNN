import os
import zipfile

if not os.path.exists('datasets'):
    os.mkdir('datasets')
    fantasy_zip = zipfile.ZipFile('shrec_16.zip')
    fantasy_zip.extractall('datasets')
    fantasy_zip.close()

