// ============================================================================
//  EAN-13 barkod kontrol uygulamasi
//
//  Senaryo 1 (ogretmeli): program acilinca BIR KEZ bir fotografta barkodu
//      mouse ile kutu icine alirsin. Program barkodun boyutunu ve kodunu
//      ogrenir. Sonra error klasorunden sectigin her fotografta barkodu
//      bulur, okur, kutu icine alir.
//  Senaryo 2 (direkt): kutu cizmek yok; fotograf sec, bulsun.
//
//  Kullanim:
//      barkod.exe              -> senaryo sorulur
//      barkod.exe 1 | 2        -> senaryo dogrudan
//      barkod.exe --batch [klasor] [--teach foto x,y,w,h] [--out klasor]
//                              -> pencere acmadan tum kareleri olc (rapor)
//      barkod.exe --file foto [--out cikti.png]
//
//  Sure olcumu: 10 ms butcesi = tespit + okuma + goruntuye cizim.
//  JPEG'i diskten okuyup acmak butceye DAHIL DEGIL (sahada goruntu kameradan
//  dogrudan bellege gelir) ama olculur ve ekranda gosterilir.
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

#include "engine.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double msBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static const char* WIN = "EAN-13 barkod kontrol";
static const int GAP = 6;            // serit satirlari arasindaki bosluk (px)

// --selftest: pencereleri kullanici olmadan dener (gelistirme/test icin).
static bool g_selfTest = false;
static cv::Rect g_autoBox;           // selftest'te "cizilmis" sayilan kutu (tam goruntu koord.)
static std::string g_dumpDir = ".";  // selftest'te ekran goruntulerinin yazilacagi klasor

// ---------------------------------------------------------------- yardimcilar

static std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// Dosyayi oku + gri olarak ac. Yol Turkce karakter icerse de calissin diye
// baytlar kendimiz okunur, cv::imdecode ile acilir. ms = okuma + JPEG acma.
static bool loadGray(const fs::path& path, cv::Mat& gray, double& ms) {
    auto t0 = Clock::now();
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    if (n <= 0) return false;
    std::vector<uchar> buf((size_t)n);
    f.seekg(0);
    f.read((char*)buf.data(), n);
    gray = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    if (gray.empty() && n > 4 && buf[0] == 0xFF && buf[1] == 0xD8) {
        // Yarim kaydedilmis JPEG (bitis isareti FFD9 yok): veri setinde 10 tane
        // var. Sonuna dolgu + bitis isareti ekleyince eksik satirlar gri kalir
        // ama goruntunun geri kalani acilir.
        buf.insert(buf.end(), 4096, 0);
        buf.push_back(0xFF); buf.push_back(0xD9);
        gray = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    }
    ms = msBetween(t0, Clock::now());
    return !gray.empty();
}

// Windows'un standart "dosya ac" penceresi.
static fs::path pickFile(const wchar_t* title, const fs::path& startDir) {
    wchar_t file[4096] = L"";
    std::wstring dir = startDir.wstring();
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof ofn;
    ofn.lpstrFilter = L"Goruntu (*.jpg;*.jpeg;*.png;*.bmp)\0*.jpg;*.jpeg;*.png;*.bmp\0Tum dosyalar\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = 4096;
    ofn.lpstrTitle = title;
    ofn.lpstrInitialDir = dir.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (!GetOpenFileNameW(&ofn)) return {};
    return fs::path(file);
}

static fs::path findErrorDir() {
    wchar_t exe[4096];
    GetModuleFileNameW(nullptr, exe, 4096);
    fs::path ed = fs::path(exe).parent_path();
    for (const fs::path& c : {fs::current_path() / "error_3", ed / "error_3", ed.parent_path() / "error_3"})
        if (fs::is_directory(c)) return c;
    return fs::current_path();
}

// Genis seridi (4200 px) ekrana sigsin diye parcalara bolup alt alta dizer.
// step < tileW verilirse parcalar ust uste biner (kutu cizerken barkodun
// iki parcanin arasina dusmemesi icin).
static cv::Mat tile(const cv::Mat& img, int tileW, int step) {
    std::vector<cv::Mat> rows;
    for (int x = 0;; x += step) {
        int w = std::min(tileW, img.cols - x);
        cv::Mat row(img.rows, tileW, img.type(), cv::Scalar(40, 40, 40));
        img(cv::Rect(x, 0, w, img.rows)).copyTo(row(cv::Rect(0, 0, w, img.rows)));
        if (!rows.empty()) rows.push_back(cv::Mat(GAP, tileW, img.type(), cv::Scalar(70, 70, 70)));
        rows.push_back(row);
        if (x + tileW >= img.cols) break;
    }
    cv::Mat out;
    cv::vconcat(rows, out);
    return out;
}

struct Line { std::string text; cv::Scalar color; };

static cv::Mat header(int width, const std::vector<Line>& lines) {
    cv::Mat bar(26 * (int)lines.size() + 10, width, CV_8UC3, cv::Scalar(25, 25, 25));
    for (size_t i = 0; i < lines.size(); ++i)
        cv::putText(bar, lines[i].text, cv::Point(10, 22 + 26 * (int)i), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, lines[i].color, 1, cv::LINE_AA);
    return bar;
}

// Pencereyi ekrana sigacak sekilde boyutlandir.
static void fitWindow(const cv::Mat& img) {
    int sw = GetSystemMetrics(SM_CXSCREEN) - 60, sh = GetSystemMetrics(SM_CYSCREEN) - 140;
    double s = std::min({1.0, sw / (double)img.cols, sh / (double)img.rows});
    cv::resizeWindow(WIN, (int)(img.cols * s), (int)(img.rows * s));
}

static bool windowClosed() {
    try { return cv::getWindowProperty(WIN, cv::WND_PROP_VISIBLE) < 1; }
    catch (...) { return true; }
}

static const cv::Scalar WHITE(230, 230, 230), GREY(150, 150, 150), GREEN(0, 255, 0),
                        RED(60, 60, 255), YELLOW(0, 230, 255);

// ------------------------------------------------- senaryo 1: kutu cizdirme

struct DragState {
    bool dragging = false, has = false;
    cv::Point a, b;          // ekran (birlesik goruntu) koordinati
    int headerH = 0, rowH = 0, rows = 0, tileW = 0;
    int rowOf(int y) const { return std::clamp((y - headerH) / (rowH + GAP), 0, rows - 1); }
};

static void onMouse(int event, int x, int y, int, void* ud) {
    auto* d = static_cast<DragState*>(ud);
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (y < d->headerH) return;
        d->dragging = true; d->has = false;
        d->a = d->b = cv::Point(x, y);
    } else if (event == cv::EVENT_MOUSEMOVE && d->dragging) {
        // kutu, baslandigi serit satirinin disina tasamaz
        int r = d->rowOf(d->a.y);
        int top = d->headerH + r * (d->rowH + GAP);
        d->b = cv::Point(std::clamp(x, 0, d->tileW - 1), std::clamp(y, top, top + d->rowH - 1));
    } else if (event == cv::EVENT_LBUTTONUP && d->dragging) {
        d->dragging = false;
        d->has = std::abs(d->a.x - d->b.x) > 15 && std::abs(d->a.y - d->b.y) > 6;
    }
}

// Kullaniciya barkodu kutu icine aldirir ve motoru egitir. false = iptal.
static bool teachInteractive(BarcodeEngine& eng, const cv::Mat& gray) {
    const int tileW = 1600, step = 1300;       // 300 px bindirme
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    cv::Mat strip = tile(bgr, tileW, step);

    DragState d;
    d.rowH = gray.rows; d.tileW = tileW;
    d.rows = (strip.rows + GAP) / (gray.rows + GAP);

    std::string warn;
    bool warned = false;
    cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    bool first = true, ok = false;
    for (int iter = 0;; ++iter) {
        std::vector<Line> lines = {
            {"SENARYO 1 - OGRETME: barkodu mouse ile kutu icine al (sol tus basili surukle)", WHITE},
            {"ENTER/SPACE: onayla    R: kutuyu sil    ESC: iptal", GREY},
            {warn.empty() ? std::string(" ") : warn, YELLOW},
        };
        cv::Mat head = header(tileW, lines);
        d.headerH = head.rows;
        cv::Mat view;
        cv::vconcat(head, strip, view);
        if (d.dragging || d.has) cv::rectangle(view, d.a, d.b, cv::Scalar(0, 200, 255), 2);
        cv::imshow(WIN, view);
        if (first) { fitWindow(view); cv::setMouseCallback(WIN, onMouse, &d); first = false; }

        int k = cv::waitKey(20) & 0xFF;
        if (g_selfTest) {                       // kullanici yerine: kutuyu koy, sonra ENTER
            int r = std::min(d.rows - 1, g_autoBox.x / step);
            int top = d.headerH + r * (d.rowH + GAP);
            d.a = cv::Point(g_autoBox.x - r * step, top + g_autoBox.y);
            d.b = d.a + cv::Point(g_autoBox.width, g_autoBox.height);
            d.has = true;
            if (iter == 10) cv::imwrite(g_dumpDir + "/selftest_ogretme.png", view);
            if (iter >= 10) k = 13;
        }
        if (k == 27 || windowClosed()) break;
        if (k == 'r' || k == 'R') { d.has = false; warn.clear(); warned = false; }
        if ((k == 13 || k == ' ') && d.has) {
            // ekran koordinati -> tam goruntu koordinati
            int r = d.rowOf(d.a.y);
            int top = d.headerH + r * (d.rowH + GAP);
            cv::Rect box(cv::Point(std::min(d.a.x, d.b.x) + r * step, std::min(d.a.y, d.b.y) - top),
                         cv::Point(std::max(d.a.x, d.b.x) + r * step, std::max(d.a.y, d.b.y) - top));
            bool decoded = eng.teach(gray, box);
            if (decoded || warned) {
                std::printf("\nOgrenildi -> barkod boyutu: %d x %d px", eng.p.barW, eng.p.barH);
                if (decoded) std::printf(", kod: %s\n", eng.p.expect.c_str());
                else std::printf(" (kutuda kod okunamadi, sadece boyut ogrenildi)\n");
                ok = true;
                break;
            }
            warn = "Kutunun icinde barkod OKUNAMADI. Tekrar ciz ya da ENTER ile sadece boyutu kullan.";
            warned = true;
        }
    }
    cv::setMouseCallback(WIN, nullptr, nullptr);     // d yigin degiskeni: geri cagrimi birak
    return ok;
}

// ------------------------------------------------------- tek fotograf isleme

struct Timing { double load = 0, find = 0, draw = 0; double total() const { return find + draw; } };

// Butceye giren kisim: find (tespit + okuma) + draw (goruntuye cizim).
static BarcodeResult process(BarcodeEngine& eng, const cv::Mat& gray, cv::Mat& canvas, Timing& t) {
    auto t0 = Clock::now();
    BarcodeResult r = eng.find(gray.data, gray.cols, gray.rows, (int)gray.step);
    auto t1 = Clock::now();
    eng.draw(gray, r, canvas);
    auto t2 = Clock::now();
    t.find = msBetween(t0, t1);
    t.draw = msBetween(t1, t2);
    // sure etiketi kronometre durduktan sonra basilir
    BarcodeEngine::stampTime(canvas, r, t.total(), eng.p.budgetMs);
    return r;
}

static std::string shortName(const fs::path& p) {
    return narrow(p.parent_path().filename().wstring()) + "/" + narrow(p.filename().wstring());
}

// Sonuc ekrani. Donus: true = yeni foto sec, false = cikis.
static bool showResult(const BarcodeEngine& eng, const fs::path& path, const cv::Mat& gray,
                       const cv::Mat& canvas, const BarcodeResult& r, const Timing& t, int mode) {
    const int tileW = 1400;
    char b1[256], b2[256], b3[256];
    if (r.found) std::snprintf(b1, sizeof b1, "BULUNDU  %s   (%d. asama)", r.text.c_str(), r.stage);
    else         std::snprintf(b1, sizeof b1, "BARKOD YOK   (ROI skoru %.0f)", r.roiScore);
    std::snprintf(b2, sizeof b2,
                  "tespit+okuma %.2f ms  +  cizim %.2f ms  =  TOPLAM %.2f ms  /  butce %.0f ms%s",
                  t.find, t.draw, t.total(), eng.p.budgetMs,
                  t.total() > eng.p.budgetMs ? "   << BUTCE ASILDI" : "");
    std::snprintf(b3, sizeof b3, "JPEG okuma+acma: %.2f ms  (butce disi: sahada goruntu bellege dogrudan gelir)",
                  t.load);
    std::string m = mode == 1 ? "Senaryo 1 (ogretilen: " + std::to_string(eng.p.barW) + "x" +
                                    std::to_string(eng.p.barH) + " px" +
                                    (eng.p.expect.empty() ? "" : ", " + eng.p.expect) + ")"
                              : "Senaryo 2 (direkt)";
    std::vector<Line> lines = {
        {shortName(path) + "    |    " + m, WHITE},
        {b1, r.found ? GREEN : RED},
        {b2, t.total() > eng.p.budgetMs ? YELLOW : WHITE},
        {b3, GREY},
        {"SPACE/ENTER: yeni foto sec    S: kaydet    ESC/Q: cikis", GREY},
    };
    std::vector<cv::Mat> parts = {header(tileW, lines), tile(canvas, tileW, tileW)};

    if (r.found) {                       // barkodun 3x buyutulmus hali
        cv::Rect z = r.box;
        z.x -= 40; z.width += 80; z.y -= 30; z.height += 60;
        z &= cv::Rect(0, 0, canvas.cols, canvas.rows);
        cv::Mat zoom;
        cv::resize(canvas(z), zoom, cv::Size(), 3, 3, cv::INTER_NEAREST);
        if (zoom.cols > tileW) zoom = zoom(cv::Rect(0, 0, tileW, zoom.rows));
        cv::Mat pad(zoom.rows + 12, tileW, CV_8UC3, cv::Scalar(25, 25, 25));
        zoom.copyTo(pad(cv::Rect(10, 6, zoom.cols, zoom.rows)));
        parts.push_back(pad);
    }
    cv::Mat view;
    cv::vconcat(parts, view);

    cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::setMouseCallback(WIN, nullptr, nullptr);
    cv::imshow(WIN, view);
    fitWindow(view);
    if (g_selfTest) {
        cv::waitKey(600);
        cv::imwrite(g_dumpDir + "/selftest_sonuc_" + path.stem().string() + ".png", view);
        return false;
    }
    for (;;) {
        int k = cv::waitKey(30) & 0xFF;
        if (k == 27 || k == 'q' || k == 'Q' || windowClosed()) return false;
        if (k == ' ' || k == 13) return true;
        if (k == 's' || k == 'S') {
            fs::create_directories("cikti");
            fs::path out = fs::path("cikti") / (path.parent_path().filename().wstring() + L"__" +
                                                path.stem().wstring() + L".png");
            std::vector<uchar> png;
            cv::imencode(".png", canvas, png);
            std::ofstream(out, std::ios::binary).write((const char*)png.data(), (std::streamsize)png.size());
            std::printf("kaydedildi: %s\n", narrow(fs::absolute(out).wstring()).c_str());
        }
    }
}

// ------------------------------------------------------------- toplu olcum

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(q * v.size()))];
}

static int runBatch(BarcodeEngine& eng, const fs::path& dir, const fs::path& outDir) {
    std::vector<fs::path> files;
    for (auto& e : fs::recursive_directory_iterator(dir)) {
        auto ext = e.path().extension().wstring();
        for (auto& c : ext) c = (wchar_t)towlower(c);
        if (e.is_regular_file() && (ext == L".jpg" || ext == L".jpeg" || ext == L".png" || ext == L".bmp"))
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::printf("goruntu bulunamadi: %s\n", narrow(dir.wstring()).c_str()); return 1; }
    if (!outDir.empty()) fs::create_directories(outDir);

    struct G { int n = 0, hit = 0; };
    std::map<std::string, G> groups;
    std::map<std::string, int> codes;
    std::vector<double> tTot, tFind, tDraw, tLoad, tS1, tS2;
    int failed = 0;
    const bool verbose = std::getenv("BK_VERBOSE") != nullptr;
    int hits = 0, s1 = 0, s2 = 0, over = 0;
    cv::Mat gray, canvas;
    for (auto& f : files) {
        Timing t;
        if (!loadGray(f, gray, t.load)) {
            failed++;
            std::printf("ACILAMADI: %s\n", shortName(f).c_str());
            continue;
        }
        BarcodeResult r = process(eng, gray, canvas, t);
        tS1.push_back(r.msStage1);
        if (r.msStage2 > 0) tS2.push_back(r.msStage2);
        if (r.found && verbose)
            std::printf("%s asama=%d %s skor=%.0f kutu=%d,%d,%d,%d\n", shortName(f).c_str(), r.stage,
                        r.text.c_str(), r.roiScore, r.box.x, r.box.y, r.box.width, r.box.height);
        tTot.push_back(t.total()); tFind.push_back(t.find); tDraw.push_back(t.draw); tLoad.push_back(t.load);
        auto& g = groups[narrow(f.parent_path().filename().wstring())];
        g.n++;
        if (r.found) { g.hit++; hits++; codes[r.text]++; (r.stage == 1 ? s1 : s2)++; }
        if (t.total() > eng.p.budgetMs) over++;
        if (!outDir.empty() && r.found) {
            std::vector<uchar> png;
            cv::imencode(".png", canvas, png);
            fs::path o = outDir / (f.parent_path().filename().wstring() + L"__" + f.stem().wstring() + L".png");
            std::ofstream(o, std::ios::binary).write((const char*)png.data(), (std::streamsize)png.size());
        }
    }
    int n = (int)tTot.size(), gOk = 0;
    std::printf("%-34s %5s %8s\n", "URUN (klasor)", "KARE", "OKUNAN");
    for (auto& [name, g] : groups) {
        std::printf("%-34s %5d %8d%s\n", name.c_str(), g.n, g.hit, g.hit ? "" : "   <-- hic okunamadi");
        gOk += g.hit > 0;
    }
    std::printf("------------------------------------------------------------------\n");
    std::printf("Kare bazli : %d/%d (%.1f%%)   [1. asama %d, 2. asama %d]\n", hits, n, 100.0 * hits / n, s1, s2);
    std::printf("Urun bazli : %d/%d\n", gOk, (int)groups.size());
    std::printf("Okunan kodlar:");
    for (auto& [c, k] : codes) std::printf("  %s x%d", c.c_str(), k);
    std::printf("\n");
    auto row = [](const char* name, const std::vector<double>& v) {
        double sum = 0; for (double x : v) sum += x;
        std::printf("%-22s ort %5.2f | ortanca %5.2f | p95 %5.2f | p99 %5.2f | max %5.2f  ms\n", name,
                    sum / v.size(), pct(v, .5), pct(v, .95), pct(v, .99), *std::max_element(v.begin(), v.end()));
    };
    row("  1. asama", tS1);
    if (!tS2.empty()) {
        std::printf("  (2. asama %d karede calisti)\n", (int)tS2.size());
        row("  2. asama", tS2);
    }
    row("tespit+okuma", tFind);
    row("cizim", tDraw);
    row("TOPLAM (butce)", tTot);
    row("JPEG okuma (disi)", tLoad);
    std::printf("Butceyi (%.0f ms) asan kare: %d/%d\n", eng.p.budgetMs, over, n);
    return 0;
}

// --------------------------------------------------------------------- main

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> a(argv + 1, argv + argc);
    auto has = [&](const char* f) { return std::find(a.begin(), a.end(), f) != a.end(); };
    auto val = [&](const char* f, int k = 1) -> std::string {
        auto it = std::find(a.begin(), a.end(), f);
        return (it != a.end() && it + k < a.end()) ? *(it + k) : std::string();
    };

    BarcodeEngine eng;
    if (has("--no-stage2")) eng.p.roiMinScore = 1e9f;
    if (!val("--cand").empty()) eng.p.roiCandidates = std::stoi(val("--cand"));
    if (!val("--budget").empty()) eng.p.budgetMs = std::stod(val("--budget"));
    if (!val("--expect").empty()) eng.p.expect = val("--expect");
    // Goruntuler kucuk; cok is parcacigi kazanc getirmiyor, uyanma gecikmesi
    // yuzunden sure dalgalaniyor (olculdu: tek parcacikta en kotu sure daha dusuk).
    cv::setNumThreads(std::getenv("BK_THREADS") ? std::atoi(std::getenv("BK_THREADS")) : 1);
    cv::Mat canvas;                 // cizim tuvali: bir kez ayrilir, hep tekrar kullanilir
    eng.warmup(canvas);

    // ---- pencere acmayan modlar (olcum / test) ----
    if (has("--teach")) {
        cv::Mat g; double ms; int x, y, w, h;
        if (!loadGray(fs::u8path(val("--teach")), g, ms) ||
            std::sscanf(val("--teach", 2).c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4) {
            std::printf("--teach <foto> <x,y,w,h>\n"); return 1;
        }
        bool dec = eng.teach(g, cv::Rect(x, y, w, h));
        std::printf("ogretme: boyut %dx%d px, kod %s\n", eng.p.barW, eng.p.barH,
                    dec ? eng.p.expect.c_str() : "(okunamadi)");
    }
    if (has("--batch")) {
        std::string d = val("--batch");
        fs::path dir = (d.empty() || d[0] == '-') ? findErrorDir() : fs::u8path(d);
        return runBatch(eng, dir, fs::u8path(val("--out")));
    }
    if (has("--file")) {
        cv::Mat g; Timing t;
        if (!loadGray(fs::u8path(val("--file")), g, t.load)) { std::printf("acilamadi\n"); return 1; }
        BarcodeResult r = process(eng, g, canvas, t);
        std::printf("%s  asama=%d  kutu=(%d,%d,%d,%d)  tespit+okuma=%.2f  cizim=%.2f  TOPLAM=%.2f ms  (JPEG %.2f ms)\n",
                    r.found ? r.text.c_str() : "BARKOD YOK", r.stage, r.box.x, r.box.y, r.box.width,
                    r.box.height, t.find, t.draw, t.total(), t.load);
        if (!val("--out").empty()) cv::imwrite(val("--out"), canvas);
        return 0;
    }

    // ---- pencereleri kullanicisiz dene:  --selftest <ogretme foto> <x,y,w,h> <test foto> [klasor] ----
    if (has("--selftest")) {
        cv::Mat g; Timing t; int x, y, w, h;
        if (!loadGray(fs::u8path(val("--selftest")), g, t.load) ||
            std::sscanf(val("--selftest", 2).c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4) return 1;
        g_selfTest = true;
        g_autoBox = cv::Rect(x, y, w, h);
        if (!val("--selftest", 4).empty()) g_dumpDir = val("--selftest", 4);
        bool ok = teachInteractive(eng, g);
        std::printf("selftest ogretme: %s\n", ok ? "tamam" : "BASARISIZ");
        fs::path f = fs::u8path(val("--selftest", 3));
        if (!loadGray(f, g, t.load)) return 1;
        BarcodeResult r = process(eng, g, canvas, t);
        showResult(eng, f, g, canvas, r, t, 1);
        std::printf("selftest sonuc: %s\n", r.found ? r.text.c_str() : "BARKOD YOK");
        return ok ? 0 : 1;
    }

    // ---- etkilesimli mod ----
    int mode = 0;
    if (!a.empty() && (a[0] == "1" || a[0] == "2")) mode = a[0][0] - '0';
    while (mode != 1 && mode != 2) {
        std::printf("\nEAN-13 barkod kontrol\n"
                    "  1) Senaryo 1: once barkodu kutu icine alarak ogret, sonra foto sec\n"
                    "  2) Senaryo 2: direkt foto sec, barkodu kendi bulsun\n"
                    "Secim (1/2): ");
        std::fflush(stdout);
        char line[64];
        if (!std::fgets(line, sizeof line, stdin)) return 0;
        mode = line[0] - '0';
    }

    fs::path dir = findErrorDir();
    cv::Mat gray;
    double ms;

    if (mode == 1) {
        std::printf("\nOgretme: barkodun NET gorundugu bir fotograf sec...\n");
        for (;;) {
            fs::path f = pickFile(L"OGRETME: barkodun net gorundugu bir fotograf sec", dir);
            if (f.empty()) return 0;
            dir = f.parent_path();
            if (!loadGray(f, gray, ms)) { std::printf("acilamadi, baska dosya sec\n"); continue; }
            if (teachInteractive(eng, gray)) break;
            std::printf("iptal edildi, baska fotograf sec (ya da pencereyi kapat)\n");
        }
        eng.warmup(canvas);  // ogrenilen parametrelerle bellekler yeniden isinsin
    }

    std::printf("\nSimdi barkodu aranacak fotografi sec (iptal = cikis)...\n");
    for (;;) {
        fs::path f = pickFile(L"Barkodu aranacak fotografi sec", dir);
        if (f.empty()) break;
        dir = f.parent_path();
        Timing t;
        // Dosya penceresinde beklerken islemci uykuya gecer, onbellekler sogur.
        // Sahada sistem araliksiz calisir; o durumu yansitmak icin olcumden once
        // SENTETIK bir goruntuyle isinma turu atilir (olculen kare ile degil).
        eng.warmup(canvas);
        if (!loadGray(f, gray, t.load)) { std::printf("acilamadi: %s\n", narrow(f.wstring()).c_str()); continue; }
        BarcodeResult r = process(eng, gray, canvas, t);
        std::printf("%-44s %-14s asama %d | tespit+okuma %.2f + cizim %.2f = %.2f ms | JPEG %.2f ms\n",
                    shortName(f).c_str(), r.found ? r.text.c_str() : "BARKOD YOK", r.stage, t.find,
                    t.draw, t.total(), t.load);
        if (!showResult(eng, f, gray, canvas, r, t, mode)) break;
    }
    cv::destroyAllWindows();
    return 0;
}
