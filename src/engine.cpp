#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <ZXing/ReadBarcode.h>

using Clock = std::chrono::steady_clock;
static inline double msSince(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// zxing ayarlari: sadece EAN-13, tek yon. try_rotate / try_downscale /
// try_invert bu veri setinde SIFIR ek okuma getiriyor ama sureyi ~2 katina
// cikariyor (900 karede olculdu) -> hepsi kapali.
static ZXing::ReaderOptions makeZxOptions() {
    ZXing::ReaderOptions o;
    o.setFormats(ZXing::BarcodeFormat::EAN13);
    o.setTryRotate(false);
    o.setTryDownscale(false);
    o.setTryInvert(false);
    o.setTryHarder(true);
    o.setBinarizer(ZXing::Binarizer::LocalAverage);
    o.setMaxNumberOfSymbols(1);
    return o;
}
static const ZXing::ReaderOptions kZx = makeZxOptions();

// zxing'in verdigi dortgenin x araligi ve orta y'si
static void quadExtent(const ZXing::Position& q, int& xa, int& xb, int& yc) {
    xa = std::min({q.topLeft().x, q.bottomLeft().x, q.topRight().x, q.bottomRight().x});
    xb = std::max({q.topLeft().x, q.bottomLeft().x, q.topRight().x, q.bottomRight().x});
    yc = (q.topLeft().y + q.bottomLeft().y + q.topRight().y + q.bottomRight().y) / 4;
}

// 2. asama CLAHE ayarlari: (clip limiti, x karo sayisi, y karo sayisi).
// 900 kare uzerinde ~40 ayar tarandi; x yonunde tek/az karo belirgin iyi
// (barkod boyunca kontrast ayni haritayla aciliyor, cubuk oranlari bozulmuyor).
static const struct { double clip; int tx, ty; } kClahe[BarcodeEngine::kS2Variants] = {
    {4.0, 1, 4}, {3.0, 1, 4}, {3.0, 2, 6}};
static const double kUpX = 1.5, kUpY = 2.0;           // 2. asama buyutme oranlari

BarcodeEngine::BarcodeEngine() {
    for (int i = 0; i < kS2Variants; ++i)
        clahe_[i] = cv::createCLAHE(kClahe[i].clip, cv::Size(kClahe[i].tx, kClahe[i].ty));
}

// 13 hane + gecerli kontrol hanesi (EAN-13 kurali: 1,3,1,3... agirlikli
// toplam 10'un kati olmali). Bozuk okumalarin cogunu eler.
bool BarcodeEngine::ean13Valid(const std::string& s) {
    if (s.size() != 13) return false;
    int sum = 0;
    for (int i = 0; i < 13; ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        int d = s[i] - '0';
        sum += (i % 2 == 0) ? d : 3 * d;      // 1,3,1,3,... agirliklari
    }
    return sum % 10 == 0;
}

bool BarcodeEngine::accept(const std::string& code) const {
    return ean13Valid(code) && (p.expect.empty() || code == p.expect);
}

// zxing sadece taradigi birkac satirin konumunu veriyor (kutu cok ince
// cikiyor). Barkodun gercek dikey sinirlarini bulmak icin: tohum satirin
// profilini al, yukari/asagi dogru "bu satir hala ayni cubuk desenine
// benziyor mu" diye korelasyonla bak. ~100 satir x ~200 px -> mikrosaniyeler.
cv::Rect BarcodeEngine::refineBox(const cv::Mat& g, int x0, int x1, int ySeed) const {
    x0 = std::clamp(x0, 0, g.cols - 2);
    x1 = std::clamp(x1, x0 + 1, g.cols - 1);
    ySeed = std::clamp(ySeed, 1, g.rows - 2);
    const int n = x1 - x0 + 1;

    std::vector<float> seed(n);
    double mean = 0;
    for (int x = 0; x < n; ++x) {
        seed[x] = (g.at<uchar>(ySeed - 1, x0 + x) + g.at<uchar>(ySeed, x0 + x) +
                   g.at<uchar>(ySeed + 1, x0 + x)) / 3.f;
        mean += seed[x];
    }
    mean /= n;
    double sn = 0;
    for (auto& v : seed) { v -= (float)mean; sn += v * v; }
    sn = std::sqrt(sn) + 1e-6;

    auto similar = [&](int y) {
        const uchar* row = g.ptr<uchar>(y) + x0;
        double m = 0;
        for (int x = 0; x < n; ++x) m += row[x];
        m /= n;
        double dot = 0, rn = 0;
        for (int x = 0; x < n; ++x) {
            double v = row[x] - m;
            dot += v * seed[x];
            rn += v * v;
        }
        return dot / (sn * (std::sqrt(rn) + 1e-6)) > 0.55;
    };

    int top = ySeed, bot = ySeed;
    while (top > 0 && similar(top - 1)) --top;
    while (bot < g.rows - 1 && similar(bot + 1)) ++bot;
    return cv::Rect(x0, top, n, bot - top + 1);
}

// ---------------------------------------------------------------- 1. asama
// Bant uzerinde dogrudan zxing. Kopya YOK: zxing'e bandin baslangic adresi
// ve stride verilir (zero-copy).
bool BarcodeEngine::stage1(const cv::Mat& g, BarcodeResult& r) {
    int y0 = std::clamp(p.bandY0, 0, g.rows - 1);
    int y1 = std::clamp(p.bandY1, y0 + 1, g.rows);
    ZXing::ImageView iv(g.ptr<uchar>(y0), g.cols, y1 - y0,
                        ZXing::ImageFormat::Lum, (int)g.step);
    auto b = ZXing::ReadBarcode(iv, kZx);
    if (!b.isValid() || !accept(b.text())) return false;

    int xa, xb, yc;
    quadExtent(b.position(), xa, xb, yc);
    r.found = true;
    r.text = b.text();
    r.stage = 1;
    r.box = refineBox(g, xa, xb, yc + y0);
    return true;
}

// ---------------------------------------------------------------- 2. asama
bool BarcodeEngine::stage2(const cv::Mat& g, BarcodeResult& r, double deadlineMs,
                           const Clock::time_point& t0) {
    // --- (a) barkod-benzeri bolgeyi bul -----------------------------------
    // Barkod = dikey cubuklar  ->  yatay gradyan |gx| cok yuksek,
    //                              dikey  gradyan |gy| cok dusuk.
    // |gx| - |gy| farki barkodu one cikarir; arka plandaki yatay bantlarin
    // kenarlari (|gy| yuksek) eksiye duser ve kendiliginden bastirilir.
    // 1/4 olcekte calisilir: 16 kat az piksel, konum icin fazlasiyla yeterli.
    cv::resize(g, small_, cv::Size(g.cols / 4, g.rows / 4), 0, 0, cv::INTER_AREA);
    cv::Sobel(small_, gx_, CV_16S, 1, 0, 3);
    cv::Sobel(small_, gy_, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(gx_, ax_);
    cv::convertScaleAbs(gy_, ay_);
    cv::subtract(ax_, ay_, diff_);                       // negatifler 0'a doyar
    // Kutu filtresi: barkod boyutunda bir pencerenin ortalama "cubukluk" skoru.
    // Pencere genisligi beklenen barkod genisliginden turetilir (212 px -> 30).
    int kw = std::max(9, (int)std::lround(p.barW / 7.0));
    cv::boxFilter(diff_, score_, CV_32F, cv::Size(kw, 8));

    const int cw = std::min(g.cols, (int)std::lround(p.barW * 1.9 / 4.0) * 4);  // 212 -> 400
    const int ch = std::min(g.rows, p.cropH);

    // --- (b) kirpimi iyilestir + buyut + zxing ----------------------------
    // 1. asamanin kacirdigi barkodlar "sinirda" olanlar: kontrast dusuk ve
    // modul ~2 px. Iki ilac:
    //   CLAHE   : yerel kontrast esitleme. Yatay golge bantlari parlakligi
    //             surekli degistirdigi icin kontrasti BOLGE BOLGE aciyor.
    //   Buyutme : x1.5 / y2 (kubik). Cubuk kenarlari piksel aralarina
    //             dusmekten kurtuluyor; zxing'e daha cok tarama satiri kaliyor.
    // Bu ikili tum kare yerine sadece 400x150'lik kirpima uygulandigi icin
    // ucuz (~1.5 ms). 900 karede tek basina OpenCV barkod cozucusunun ~2
    // kati okuma getirdi (120'ye karsi 61) ve daha hizli.
    //
    // Ayni kirpim 3 farkli CLAHE ayariyla denenir (okumalar sinirda oldugu
    // icin kucuk bir ayar farki sonucu degistirebiliyor).
    //
    // YANLIS OKUMAYA KARSI:
    //   - Kod biliniyorsa (senaryo 1, p.expect): tek okuma yeter, cunku
    //     beklenen kodla birebir eslesmek zorunda.
    //   - Kod bilinmiyorsa (senaryo 2): IKI ayri deneme AYNI kodu vermeli.
    //     Kontrol hanesi tek basina yetmiyor - olcumde checksum'i gecen
    //     yanlis okumalar goruldu; cift onayla 900 karede sifira indi.
    std::string cands[kS2Variants + 1];
    int nCands = 0;
    auto vote = [&](const std::string& code) {
        if (!ean13Valid(code)) return false;
        if (!p.expect.empty()) return code == p.expect;
        for (int i = 0; i < nCands; ++i)
            if (cands[i] == code) return true;
        cands[nCands++] = code;
        return false;
    };

    // --- (c) en yuksek skorlu birkac bolgeyi sirayla dene ------------------
    // TEK adaya bakmak yetmiyor: seritte ayni anda 3-4 kutu var ve etiket
    // yazilari / tehlike isaretleri de dikey kenar uretiyor. Olculdu - bir
    // karede barkod 2. sirada ciktigi icin kaciriliyordu (tepe skoru 200 bos
    // bir bolgeydi, barkodun skoru 191). Her denemeden sonra o serit skor
    // haritasindan silinip bir sonraki tepe aranir.
    for (int cand = 0; cand < p.roiCandidates; ++cand) {
        if (msSince(t0) > deadlineMs) break;

        double mx; cv::Point loc;
        cv::minMaxLoc(score_, nullptr, &mx, nullptr, &loc);
        if (cand == 0) r.roiScore = (float)mx;
        if (mx < p.roiMinScore) break;                   // barkod-benzeri bolge kalmadi

        // Tepe noktasi barkodun ORTASI olmayabilir. Tepenin etrafinda skorun
        // %60'inin ustunde kalan seridin sinirlarini bul, ortasini al. Kirpimin
        // barkoda hizali olmasi okuma basarisini belirgin etkiliyor.
        const float* row = score_.ptr<float>(loc.y);
        float thr = (float)mx * p.roiExtentFrac;
        int lo = loc.x, hi = loc.x;
        while (lo > 0 && row[lo - 1] >= thr) --lo;
        while (hi < score_.cols - 1 && row[hi + 1] >= thr) ++hi;
        int x0 = std::clamp((lo + hi) * 2 - cw / 2, 0, g.cols - cw);
        int y0 = std::clamp(loc.y * 4 - ch / 2, 0, g.rows - ch);

        // bu seridi bastir ki sonraki tur baska bir yere baksin
        cv::rectangle(score_, cv::Rect(lo - 8, 0, hi - lo + 17, score_.rows), cv::Scalar(0),
                      cv::FILLED);

        g(cv::Rect(x0, y0, cw, ch)).copyTo(crop_);       // kucuk, bitisik bellek
        for (int v = 0; v < kS2Variants; ++v) {
            if (v > 0 && msSince(t0) > deadlineMs) return false;
            // senaryo 2: iki deneme de bos dondu -> kalan tek deneme onay getiremez
            if (p.expect.empty() && v == kS2Variants - 1 && nCands == 0) break;

            clahe_[v]->apply(crop_, eq_);
            cv::resize(eq_, up_, cv::Size(), kUpX, kUpY, cv::INTER_CUBIC);
            ZXing::ImageView iv(up_.data, up_.cols, up_.rows, ZXing::ImageFormat::Lum, (int)up_.step);
            auto b = ZXing::ReadBarcode(iv, kZx);
            if (!b.isValid() || !vote(b.text())) continue;

            int xa, xb, yc;
            quadExtent(b.position(), xa, xb, yc);
            r.found = true; r.text = b.text(); r.stage = 2;
            r.box = refineBox(g, x0 + (int)(xa / kUpX), x0 + (int)(xb / kUpX), y0 + (int)(yc / kUpY));
            return true;
        }
        nCands = 0;                                      // oylama her bolgede bastan
    }

    // NOT: OpenCV'nin kendi barkod cozucusu (cv::barcode) de denendi. Tek basina
    // 61 kare okuyor (bu yontem 120), ~2x yavas, ve kalan sure butcesinde
    // yalnizca 0-1 ek kare getiriyor -> kullanilmadi.
    return false;
}

// ---------------------------------------------------------------- 3. asama
// DIKEY barkod (90 derece donuk): cubuklar YATAY uzanir.
//
// 2. asamanin skoru |gx|-|gy| idi; boyle bir barkod orada EKSIYE duser ve
// konveyor bandinin yatay cizgileriyle birlikte elenir. Burada isaret ters
// cevrilir: |gy|-|gx|.
//
// MALIYET: Sobel YENIDEN HESAPLANMAZ. 2. asamada uretilen ax_/ay_ aynen
// kullanilir (en pahali kisim oydu). Geriye sadece boxFilter + kirpim +
// dondurme + CLAHE + zxing kaliyor -> olcumde p99'a +1.2 ms.
//
// Bant yanlis pozitif uretmiyor, cunku bu asama yalnizca 2. asama basarisiz
// olunca calisir ve o karelerde yatay ROI skoru zaten esigin altindadir
// (olculdu: 86-144 < 170), yani 2. asama erken cikip butceyi bos birakir.
//
// Kirpim olcusu 200x440 olarak OLCULDU (4 yukseklik x 3 genislik tarandi).
// Daha genis kirpimlar barkodu kesip yanlis okumaya yol aciyor: 260x520 ve
// 680x260 ayarlarinda sabit bir yanlis okuma gorulduz, 200x440'ta sifir.
bool BarcodeEngine::stage2v(const cv::Mat& g, BarcodeResult& r, double deadlineMs,
                            const Clock::time_point& t0) {
    if (ax_.empty() || ay_.empty()) return false;        // stage2 calismamis

    cv::subtract(ay_, ax_, vdiff_);                      // negatifler 0'a doyar
    // Kutu filtresi 2. asamanin DEVRIGI: uzun eksen dikeyde.
    int kl = std::max(9, (int)std::lround(p.barW / 7.0));
    cv::boxFilter(vdiff_, vscore_, CV_32F, cv::Size(8, kl));

    const int cw = std::min(g.cols, p.vCropW);
    const int ch = std::min(g.rows, p.vCropH);

    // Yanlis okumaya karsi 2. asamadaki oylamanin aynisi.
    std::string cands[kS2Variants + 1];
    int nCands = 0;
    auto vote = [&](const std::string& code) {
        if (!ean13Valid(code)) return false;
        if (!p.expect.empty()) return code == p.expect;
        for (int i = 0; i < nCands; ++i)
            if (cands[i] == code) return true;
        cands[nCands++] = code;
        return false;
    };

    for (int cand = 0; cand < p.vCandidates; ++cand) {
        if (msSince(t0) > deadlineMs) break;

        double mx; cv::Point loc;
        cv::minMaxLoc(vscore_, nullptr, &mx, nullptr, &loc);
        if (mx < p.vMinScore) break;

        // 2. asamada serit x ekseninde araniyordu; dikey barkodda Y ekseninde.
        float thr = (float)mx * p.roiExtentFrac;
        int lo = loc.y, hi = loc.y;
        while (lo > 0 && vscore_.at<float>(lo - 1, loc.x) >= thr) --lo;
        while (hi < vscore_.rows - 1 && vscore_.at<float>(hi + 1, loc.x) >= thr) ++hi;
        int y0 = std::clamp((lo + hi) * 2 - ch / 2, 0, std::max(0, g.rows - ch));
        int x0 = std::clamp(loc.x * 4 - cw / 2, 0, std::max(0, g.cols - cw));

        // bu seridi bastir ki sonraki tur baska yere baksin
        cv::rectangle(vscore_, cv::Rect(0, lo - 8, vscore_.cols, hi - lo + 17),
                      cv::Scalar(0), cv::FILLED);

        int ww = std::min(cw, g.cols - x0), hh = std::min(ch, g.rows - y0);
        if (ww < 20 || hh < 20) continue;
        g(cv::Rect(x0, y0, ww, hh)).copyTo(vcrop_);
        cv::rotate(vcrop_, vrot_, cv::ROTATE_90_CLOCKWISE);   // artik YATAY barkod

        for (int v = 0; v < kS2Variants; ++v) {
            if (v > 0 && msSince(t0) > deadlineMs) return false;
            if (p.expect.empty() && v == kS2Variants - 1 && nCands == 0) break;

            clahe_[v]->apply(vrot_, veq_);
            cv::resize(veq_, vup_, cv::Size(), kUpX, kUpY, cv::INTER_CUBIC);
            ZXing::ImageView iv(vup_.data, vup_.cols, vup_.rows,
                                ZXing::ImageFormat::Lum, (int)vup_.step);
            auto b = ZXing::ReadBarcode(iv, kZx);
            if (!b.isValid() || !vote(b.text())) continue;

            // Dondurulmus kirpimda barkod YATAY durumda, yani 2. asamadaki
            // refineBox aynen kullanilabilir (zxing 1D'de cubuk BOYUNU vermez,
            // quad bir cizgiye cokuyor - kutuyu refineBox genisletir).
            int xa, xb, yc;
            quadExtent(b.position(), xa, xb, yc);
            cv::Rect rb_ = refineBox(vrot_, (int)(xa / kUpX), (int)(xb / kUpX),
                                     (int)(yc / kUpY));
            // ROTATE_90_CLOCKWISE geri donusu:  dst(x,y) -> kirpim(x = y, y = hh-1-x)
            //   dikdortgen:  (rx,ry,rw,rh) -> (ry, hh-rx-rw, rh, rw)
            cv::Rect box(x0 + rb_.y, y0 + (hh - rb_.x - rb_.width),
                         rb_.height, rb_.width);
            r.found = true;
            r.text  = b.text();
            r.stage = 3;
            r.box   = box & cv::Rect(0, 0, g.cols, g.rows);
            return true;
        }
        nCands = 0;                                      // oylama her bolgede bastan
    }
    return false;
}

// ------------------------------------------------------------------- find
BarcodeResult BarcodeEngine::find(const uint8_t* data, int width, int height, int stride) {
    BarcodeResult r;
    // Kopyasiz sarmalayici: cv::Mat sadece baslik, pikseller yerinde kalir.
    const cv::Mat g(height, width, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

    auto t0 = Clock::now();
    bool ok = stage1(g, r);
    r.msStage1 = msSince(t0);
    if (!ok) {
        // 2. asama butcenin %75'inde kesilir: sure dolarsa "bulunamadi"
        // denir ama 10 ms siniri asilmaz (gercek zamanli sistemde dogru olan).
        // 2. asama yatay barkodu arar; bulamazsa ve goruntu dikey bir barkodu
        // alacak kadar uzunsa 3. asama AYNI butce siniri icinde denenir.
        if (!stage2(g, r, p.budgetMs * p.budgetFrac, t0) && g.rows >= p.vMinRows)
            stage2v(g, r, p.budgetMs * p.budgetFrac, t0);
        r.msStage2 = msSince(t0) - r.msStage1;
    }
    return r;
}

// ------------------------------------------------------------------ teach
bool BarcodeEngine::teach(const cv::Mat& gray, cv::Rect ub) {
    ub &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (ub.width < 20 || ub.height < 8) return false;

    // Kullanicinin kutusu kabadir. Kutunun (biraz genisletilmis) icinde
    // barkodu okumayi dene; okunursa GERCEK cubuk genisligini ve kodu ogren.
    // Okunamazsa kutunun kendi olculeri kullanilir.
    cv::Rect ex = ub;
    ex.x -= 30; ex.width += 60; ex.y -= 10; ex.height += 20;
    ex &= cv::Rect(0, 0, gray.cols, gray.rows);

    cv::Mat up;
    cv::resize(gray(ex), up, cv::Size(), 2, 2, cv::INTER_CUBIC);
    ZXing::ImageView iv(up.data, up.cols, up.rows, ZXing::ImageFormat::Lum, (int)up.step);
    auto b = ZXing::ReadBarcode(iv, kZx);

    bool decoded = false;
    int w = ub.width, h = ub.height;
    p.expect.clear();
    if (b.isValid() && ean13Valid(b.text())) {
        int xa, xb, yc;
        quadExtent(b.position(), xa, xb, yc);
        cv::Rect fine = refineBox(gray, ex.x + xa / 2, ex.x + xb / 2, ex.y + yc / 2);
        w = fine.width; h = std::max(fine.height, 10);
        p.expect = b.text();
        decoded = true;
    }
    p.barW = std::clamp(w, 60, gray.cols);
    p.barH = std::clamp(h, 10, gray.rows);
    return decoded;
}

// ------------------------------------------------------------------- draw
void BarcodeEngine::draw(const cv::Mat& gray, const BarcodeResult& r, cv::Mat& canvas) const {
    cv::cvtColor(gray, canvas, cv::COLOR_GRAY2BGR);     // canvas bellegi tekrar kullanilir
    if (!r.found) return;
    const cv::Scalar green(0, 255, 0);
    const cv::Rect all(0, 0, canvas.cols, canvas.rows);
    cv::Rect b = r.box;
    b.x -= 6; b.y -= 6; b.width += 12; b.height += 12;
    b &= all;
    cv::rectangle(canvas, b, green, 2);

    int base = 0;
    cv::Size ts = cv::getTextSize(r.text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    int ly = b.y - 6;                                   // etiket kutunun ustune
    if (ly - ts.height < 2) ly = std::min(canvas.rows - 4, b.y + b.height + ts.height + 6);
    int lx = std::clamp(b.x, 0, std::max(0, canvas.cols - ts.width - 8));
    cv::rectangle(canvas, cv::Rect(lx, ly - ts.height - 4, ts.width + 8, ts.height + 8) & all,
                  green, cv::FILLED);
    cv::putText(canvas, r.text, cv::Point(lx + 4, ly), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
}

void BarcodeEngine::stampTime(cv::Mat& canvas, const BarcodeResult& r,
                              double totalMs, double budgetMs) {
    char buf[160];
    if (r.found) std::snprintf(buf, sizeof buf, "OK %s | %.2f ms", r.text.c_str(), totalMs);
    else         std::snprintf(buf, sizeof buf, "BARKOD YOK | %.2f ms", totalMs);
    cv::Scalar col = r.found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    if (totalMs > budgetMs) col = cv::Scalar(0, 255, 255);          // butce asildi -> sari
    int base = 0;
    cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    // sag alt kose: kameranin kendi (sol ust) yazisiyla cakismasin
    int x = std::max(0, canvas.cols - ts.width - 16), y = canvas.rows;
    cv::rectangle(canvas, cv::Rect(x, y - ts.height - 12, ts.width + 16, ts.height + 12) &
                          cv::Rect(0, 0, canvas.cols, canvas.rows),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, buf, cv::Point(x + 8, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                cv::LINE_AA);
}

void BarcodeEngine::warmup(cv::Mat& canvas) {
    cv::Mat noise(200, 4200, CV_8UC1);
    cv::randu(noise, 0, 255);
    // dikey cizgili bir bolge ekle ki 2. asamanin tum dallari da calissin
    for (int x = 1000; x < 1220; x += 4) noise(cv::Rect(x, 80, 2, 60)).setTo(0);
    for (int i = 0; i < 5; ++i) {
        BarcodeResult r = find(noise.data, noise.cols, noise.rows, (int)noise.step);
        draw(noise, r, canvas);
    }
}
