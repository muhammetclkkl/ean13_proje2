// ============================================================================
//  EAN-13 barkod tespit + okuma cekirdegi  (C++17, OpenCV + zxing-cpp)
//
//  Boru hatti (cascade = kademeli):
//    1. asama : y bandini al, zxing ile dogrudan tara.   Kolay karelerde biter.
//    2. asama : sadece 1. asama bos donerse calisir.
//               |gx|-|gy| gradyan haritasi ile barkod-benzeri bolgeyi bul,
//               kucuk bir parca kirp, kontrasti ac (CLAHE), buyut, zxing ile oku.
//
//  Cekirdek, dosya adi DEGIL ham bellek alir (data, w, h, stride). Boylece
//  sahada kameradan gelen buffer hicbir degisiklik olmadan verilebilir.
// ============================================================================
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct EngineParams {
    // --- taranacak satir araligi (barkod bu veri setinde hep bu bantta) ---
    int bandY0 = 50;
    int bandY1 = 185;

    // --- beklenen barkod boyutu (px). Senaryo 1'de kullanicinin cizdigi
    //     kutudan OGRENILIR, senaryo 2'de bu varsayilanlar kullanilir. ---
    int barW = 212;
    int barH = 45;

    // --- 2. asama ---
    int   roiCandidates = 3;      // kac aday bolge denenecek (sureye gore kesilir)
    float roiMinScore   = 170.f;  // bu skorun altinda "barkod-benzeri bolge yok" -> 2. asamayi atla
    float roiExtentFrac = 0.6f;   // ROI merkezleme: tepenin bu orani ustundeki seridin ortasi
    int   cropH         = 150;    // kirpim yuksekligi; genislik barW'den turetilir

    // --- 3. asama: DIKEY (90 derece donuk) barkod ---
    //     Bazi urunlerde barkod dik duruyor: cubuklar YATAY uzaniyor. 2. asamanin
    //     |gx|-|gy| skoru boyle bir barkodu eksiye dusurup eledigi icin hic
    //     bulunamiyordu. 3. asama ayni Sobel ciktisini ters isaretle kullanir.
    int   vMinRows    = 300;    // goruntu bundan kisaysa dikey barkod sigamaz -> atla
    int   vCropW      = 200;    // dikey kirpim genisligi  (cubuk boyu yonu)
    int   vCropH      = 440;    // dikey kirpim yuksekligi (barkod boyu yonu)
    int   vCandidates = 3;      // kac aday bolge denenecek
    float vMinScore   = 170.f;  // bu skorun altinda dikey barkod-benzeri bolge yok

    // --- zaman butcesi ---
    double budgetMs   = 10.0;
    double budgetFrac = 0.70;     // 2. asama butcenin bu oraninda kesilir (kalan pay cizime)

    // --- yanlis okumaya karsi: bos degilse sadece bu kod kabul edilir ---
    std::string expect;
};

struct BarcodeResult {
    bool        found = false;
    std::string text;             // okunan EAN-13
    cv::Rect    box;              // tam goruntu koordinatinda sinirlayici kutu
    int         stage = 0;        // 1 = hizli yol, 2 = kurtarma yolu,
                                  // 3 = dikey barkod yolu, 0 = bulunamadi
    float       roiScore = 0.f;   // 2. asamadaki en iyi ROI skoru (bilgi amacli)
    double      msStage1 = 0, msStage2 = 0;
};

class BarcodeEngine {
public:
    static constexpr int kS2Variants = 3;   // 2. asamada denenen CLAHE ayari sayisi
    EngineParams p;

    BarcodeEngine();

    // Asil is: ham gri goruntude (8 bit, tek kanal) EAN-13 bul + oku.
    BarcodeResult find(const uint8_t* data, int width, int height, int stride);

    // Senaryo 1: kullanicinin cizdigi kutudan barkod geometrisini ogren.
    // Kutunun icinde barkod okunabilirse kodu da ogrenir (p.expect).
    // Donus: kutunun icinde barkod okunabildi mi?
    bool teach(const cv::Mat& gray, cv::Rect userBox);

    // Sonucu goruntuye ciz (kutu + kod). canvas tekrar kullanilir (yeniden
    // bellek ayirmamak icin) - bu sure de 10 ms butcesine dahildir.
    void draw(const cv::Mat& gray, const BarcodeResult& r, cv::Mat& canvas) const;

    // Sure etiketini goruntuye bas. Kronometre durduktan SONRA cagrilir
    // (olculen sureyi yazabilmek icin once olcumun bitmesi gerekir).
    static void stampTime(cv::Mat& canvas, const BarcodeResult& r,
                          double totalMs, double budgetMs);

    // Ilk cagri maliyetlerini (bellek ayirma, tembel ilklendirme) olcumden
    // once harcamak icin. Sahada sistem surekli calistigi icin "isinmis"
    // durum gercegi yansitir.
    void warmup(cv::Mat& canvas);

    static bool ean13Valid(const std::string& s);

private:
    bool stage1(const cv::Mat& gray, BarcodeResult& r);
    bool stage2(const cv::Mat& gray, BarcodeResult& r, double deadlineMs,
                const std::chrono::steady_clock::time_point& t0);
    // 3. asama. stage2'den SONRA cagrilir: Sobel ciktilarini (ax_, ay_) ondan devralir.
    bool stage2v(const cv::Mat& gray, BarcodeResult& r, double deadlineMs,
                 const std::chrono::steady_clock::time_point& t0);
    cv::Rect refineBox(const cv::Mat& gray, int x0, int x1, int ySeed) const;
    bool accept(const std::string& code) const;

    // tekrar kullanilan ara bellekler (her karede malloc yapmamak icin)
    cv::Ptr<cv::CLAHE> clahe_[kS2Variants];
    cv::Mat small_, gx_, gy_, ax_, ay_, diff_, score_, crop_, eq_, up_;
    cv::Mat vdiff_, vscore_, vcrop_, vrot_, veq_, vup_;   // 3. asama
};
