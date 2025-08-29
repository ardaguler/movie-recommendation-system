# Movie Recommendation System (ALS)

![python](https://img.shields.io/badge/python-3.9%2B-blue)  
![Pandas](https://img.shields.io/badge/Pandas-data-informational)  
![NumPy](https://img.shields.io/badge/NumPy-array-informational)  
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)  
![implicit](https://img.shields.io/badge/implicit-ALS-green)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-plots-informational)

> Otomatik oluşturulma tarihi: **2025-08-29**

Bu repo, kullanıcı–film etkileşimlerinden (rating/izleme) **ALS (Alternating Least Squares)** tabanlı bir öneri sistemi kurar. Veri hazırlanır, sparse kullanıcı-öğe matrisi oluşturulur, **implicit ALS** ile model eğitilir ve **benzer film / kişiselleştirilmiş öneriler** üretilir.

## 📁 Önerilen Klasör Yapısı
```
.
├── data/
│   ├── ratings.csv        # userId, movieId, rating, timestamp
│   └── movies.csv         # movieId, title, genres
├── notebooks/
│   └── movie_recsys.ipynb # çalıştığın notebook
├── src/
│   ├── data_utils.py
│   ├── train_als.py
│   └── recommend.py
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Kurulum
```bash
# (önerilir) sanal ortam
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## 🧹 Veri Hazırlama (Örnek)
- `ratings.csv` → `userId, movieId, rating` alanlarını kullan.
- Ratingleri implicit ALS için **tercih skoru** olarak yeniden ölçeklemek istersen (örn. 0–1 arası), min-max ya da log-dönüşüm uygulayabilirsin.
- `scipy.sparse.csr_matrix` ile kullanıcı–film matrisi inşa edilir.

## 🎯 Model (ALS, implicit kütüphanesi)
**Temel eğitim akışı:**
1. Veriyi train/test ayır.  
2. CSR matris oluştur.  
3. `implicit.als.AlternatingLeastSquares` ile modeli eğit:  
   - `factors` (gizli boyut),  
   - `regularization` (regülerizasyon),  
   - `iterations`,  
   - `alpha` (confidence için, implicit veride).  
4. Değerlendirme:  
   - Precision@K, Recall@K, MAP@K gibi metriklerden en az birini raporla.  
5. Öneri üret:  
   - **Kullanıcıya öneri**: `model.recommend(user_id, user_items, N=10)`  
   - **Benzer film**: `model.similar_items(movie_id, N=10)`

> Not: Explicit rating’ler için önce implicit score’a dönüştürmek gerekebilir (ör. rating>3 ⇒ pozitif tercih).

## ▶️ Çalıştırma Örnekleri
### Notebook ile
```bash
jupyter notebook notebooks/movie_recsys.ipynb
```

### Python betikleriyle (opsiyonel)
```bash
python -m src.train_als --ratings data/ratings.csv --movies data/movies.csv --factors 64 --alpha 40 --iters 20
python -m src.recommend --user-id 123 --top-k 10
```

## 📊 Sonuçlar
- `results/` altına Precision@K/Recall@K tablolarını ve örnek öneri listelerini kaydedin.
- Örnek görseller (matplotlib) ve sample output’lar README’ye eklenebilir.

## 🧩 İpuçları
- **Cold-start** için içerik tabanlı (title/genre) benzerlikleri karıştıran **hibrid** bir skor hesaplayabilirsin.
- `tqdm` ile eğitim/öneri adımlarına progress bar ekleyebilirsin.
- Büyük veri setlerinde ALS için **GPU** gerekmez; fakat `implicit`’in `alternating_least_squares` fonksiyonunda numba hızlandırması işe yarar.

## 🤝 Katkı
PR’lar memnuniyetle. Lütfen düzgün bir açıklama ve tekrar üretilebilir adımlar ekleyin.

## 📜 Lisans
MIT (öneri). `LICENSE` dosyasını eklemeyi unutmayın.
