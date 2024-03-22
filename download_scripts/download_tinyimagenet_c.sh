cd data
wget -O Tiny-ImageNet-C.tar https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1
mkdir -p TinyImageNet/corrupt
tar -xvf Tiny-ImageNet-C.tar -C TinyImageNet/corrupt/ --strip-components=1
rm -rf Tiny-ImageNet-C.tar
cd ..