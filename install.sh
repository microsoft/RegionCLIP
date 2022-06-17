pip install dataclasses ftfy regex tqdm timm diffdist spacy
pip install git+https://github.com/lvis-dataset/lvis-api.git

CURRENT_DIR=${PWD##*/}
cd ../
python -m pip install -e $CURRENT_DIR
cd $CURRENT_DIR

# for tsv loading
pip install Pillow==7.1.2

#ln -s DIR_to_COCO datasets/coco
#ln -s DIR_to_LVIS datasets/lvis