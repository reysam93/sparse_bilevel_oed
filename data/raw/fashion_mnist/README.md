# Fashion-MNIST local snapshot

`fashion_mnist_28x28_u8.npz` (key `images`, uint8, (70000, 28, 28)):
train+test images of Fashion-MNIST (Xiao, Rasul, Vollgraf, 2017),
read from the idx files of https://github.com/zalandoresearch/fashion-mnist
(MIT license), concatenated train(60000)+t10k(10000), no labels.
Created 2026-08-31 for the ICASSP RF experiment; the digits_dct
adapter prefers this snapshot over fetch_openml so runs are
network-free.
