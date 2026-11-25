# Kurulum — Gerekli Kütüphaneler

Bu proje iki kütüphaneye ihtiyaç duyar: **OpenCV** (görüntü işleme) ve **zxing-cpp**
(barkod okuma). Aşağıda hem Windows hem Linux için kurulum adımları var.

## Windows (MSYS2 / UCRT64)

1. MSYS2'yi kur: https://www.msys2.org/
2. Başlat menüsünden **"MSYS2 UCRT64"** terminalini aç (sade "MSYS2" değil, UCRT64 olanı).
3. Derleyici, CMake ve OpenCV'yi kur:
   ```bash
   pacman -Syu
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake \
             mingw-w64-ucrt-x86_64-opencv git
   ```
4. zxing-cpp için hazır paket yoksa (aynı terminalde):
   ```bash
   cd ~
   git clone https://github.com/zxing-cpp/zxing-cpp.git
   cd zxing-cpp
   mkdir build && cd build
   cmake .. -G "MSYS Makefiles"
   mingw32-make -j$(nproc)
   mingw32-make install
   ```
5. Proje klasörüne gel ve derle:
   ```bash
   ./build.sh
   ```
   (CMD'den deniyorsan `build.bat` dosyasına çift tıkla.)

## Linux (Rock 5B / Armbian, Debian tabanlı)

1. Derleyici, CMake ve OpenCV'yi kur:
   ```bash
   sudo apt update
   sudo apt install build-essential cmake git libopencv-dev
   ```
2. zxing-cpp'yi kaynaktan derle (apt deposunda hazır paket yok):
   ```bash
   cd ~
   git clone https://github.com/zxing-cpp/zxing-cpp.git
   cd zxing-cpp
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   sudo ldconfig
   ```
3. Projeyi derle:
   ```bash
   cd src
   mkdir build && cd build
   cmake ..
   make
   ```

## Kontrol

Her iki platformda da şu komutlarla kütüphanelerin göründüğünü doğrulayabilirsin:

```bash
pkg-config --modversion opencv4
find /usr/local -iname "*zxing*"      # Linux
```
