pip install dataclasses ftfy regex tqdm timm diffdist spacy
pip install git+https://github.com/lvis-dataset/lvis-api.git
python -m spacy download en_core_web_sm

CURRENT_DIR=${PWD##*/}
cd ../
python -m pip install -e $CURRENT_DIR
cd $CURRENT_DIR

# for tsv loading
pip install Pillow==7.1.2

ln -s /mnt/data_storage/coco datasets/coco
ln -s /mnt/data_storage/lvis datasets/lvis