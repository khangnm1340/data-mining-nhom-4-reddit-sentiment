#set page(
  margin: 1.5cm,
)
#set text(
  font: "Times New Roman", // name of the font as Typst sees it
  size: 11pt,
)
#rect(
  width: 100%,
  height: 100%,
  stroke: 2pt, // outer line thickness
  inset: 5pt, // padding inside outer rect
)[
  #rect(
    width: 100%,
    height: 100%,
    stroke: 2pt, // inner line thickness
    inset: 9pt, // gap between inner border and your content
  )[

    #align(center)[
      #v(1cm)
      #text(size: 10pt, weight: "bold")[
        BỘ GIAO THÔNG VẬN TẢI \
        TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI THÀNH PHỐ HỒ CHÍ MINH \
      ]


      #image("images/UTH_logo_idrV1VcT-T_1_edited.jpeg", width: 40%)

      #v(1cm)

      #text(size: 20pt, weight: "bold")[
        KHAI THÁC DỮ LIỆU \
        BÁO CÁO CUỐI KỲ
      ]

      #v(0.5cm)

      #text(size: 20pt, weight: "bold")[
      PHÂN TÍCH NỘI DUNG TRÒ CHUYỆN \ TRÍCH XUẤT VẤN ĐỀ PHỔ
BIẾN VỀ TÀI SẢN
      ]

      #v(2cm)

      #align(center, block[
        #set align(left)
        #text(size: 14pt)[
          Giảng Viên Hướng Dẫn: *Trần Thế Vinh*\

          *Họ Và Tên Sinh Viên Thực Hiện Và MSSV:* \
          Phạm Hoàng Thiện - 051205000064\
          Ngô Minh Khang - 086250511340\
          Nguyễn Văn Mạnh - 027205000040\
          Nguyễn Văn Quang - 038205004237\
          Lê Huỳnh Cao Dương-079204051017\
          Nguyễn Ngọc Anh Thư\
          Nguyễn Đăng Khoa\
          Nguyễn Dương\

        ]
      ])

      #v(5cm)

      #text(size: 11pt, weight: "bold", style: "italic")[
        Thành phố Hồ Chí Minh, ngày 18 tháng 9 năm 2024
      ]
    ]
  ]]
  #set page(width: 21cm, height: 29.7cm, margin: (top: 2.5cm, bottom: 2.5cm,
  left: 2cm, right: 2cm))
  #set text(font: "Noto Serif", lang: "vi")
  #let accent = rgb("#0B7285")
  #let accent_secondary = rgb("#1864AB")
  #let accent_tertiary = rgb("#495057")
  #let meta = text.with(fill: accent_secondary,weight: "bold", size: 13pt)
  #let mono = text.with(font: "DejaVu Sans Mono")
  #set heading(numbering: "1.")
  #show heading.where(level: 1): set text(size: 22pt, weight: "bold", fill: accent)
  #show heading.where(level: 2): set text(size: 16pt, weight: "bold", fill: accent_secondary)
  #show heading.where(level: 3): set text(size: 13pt, weight: "semibold", fill: accent_tertiary)
  #show figure.caption: set text(size: 10pt, fill: accent_secondary)


  #outline(
    title: [#heading(level: 1, numbering: none)[Mục lục]],
    depth: 3,
  )


  #pagebreak()

  = Phân tích thảo luận công khai để rút ra insight sản phẩm
  // #meta[#smallcaps[Nhóm :] Group 4] \
  // #meta[#smallcaps[Học phần :] UTH Data Mining — Final Report] \
  #meta[#smallcaps[Pipeline :]] *Thu thập Reddit + Tiki → Chuẩn hóa Parquet
→ EDA → Topic modeling (TF-IDF → SVD → KMeans) + Aspect sentiment*
 *Link github*: https://github.com/khangnm1340/data-mining-nhom-4-reddit-sentiment

  == Tóm tắt điều hành
  - Chúng em khai thác 16 cộng đồng Reddit tập trung vào phần cứng (kèm các
  đánh giá Tiki bổ sung) để phát hiện vấn đề sản phẩm lặp lại, ưu/nhược
  điểm và xu hướng chủ đề.
  - Công cụ thu thập Arctic Shift vượt giới hạn API, cung cấp dữ liệu nhiều
  năm được lưu dưới dạng Parquet đồng nhất.
  - CLI (`run_pipeline.py`) chuyển tập văn bản đã lọc thành đặc trưng TF-
  IDF/SVD, cụm KMeans và các artefact sentiment theo khía cạnh.
  - Kế hoạch fine-tune transformer đã được lập nhưng tạm hoãn sau phân tích
  chi phí/lợi ích; VADER đảm nhiệm pipeline sentiment bàn giao.
  - Đầu ra gồm joblib/JSON tái sử dụng, dashboard PNG và tư liệu thuyết
  trình trong `extra/`.

  == Nguồn dữ liệu & Thu thập
  === Lý do chọn Reddit & Tiki
  - Reddit cung cấp chủ đề giàu văn bản và được cộng đồng kiểm duyệt
  - Tiki bổ sung phản hồi mua hàng xác thực từ thị trường Việt Nam.
  - Facebook và TikTok bị đặt thấp ưu tiên do thiếu API, nhiễu bot và thiên
  về nội dung ảnh và video( mà dự án này làm NPL).
  - Danh sách subreddit bao trùm laptop, điện thoại, âm thanh, nhà thông
  minh, nhiếp ảnh, lắp ráp PC và công thái học — nắm bắt góc nhìn người
  chơi và người xử lý sự cố.
#columns(2)[
  #figure(
    image("images/subreddits.png", height: 20%),
    caption: [Danh sách subreddit tiêu biểu bao phủ homelab, màn hình, cộng
  đồng audiophile và hệ sinh thái Apple.]
  )
#colbreak()
  #figure(
    image("images/tiki.png"),
    caption: [Mẫu đánh giá Tiki cho máy ảnh — chấm điểm có cấu trúc kết hợp
  với phản hồi tiếng Việt bổ sung cho Reddit.]
  )
]

  === Quy trình Arctic Shift
  - Các gói dữ liệu lịch sử (Academic Torrents) cấp nguồn cho bộ thu Arctic
  Shift, loại bỏ giới hạn ~1 000 bài/subreddit của PRAW.
  - Dump đánh giá Tiki bổ trợ Reddit để đối chiếu chéo khi khả dụng.
  - Dữ liệu được chuẩn hóa bằng Polars/Nushell từ 32 JSONL sang
  `posts.parquet` và `comments.parquet`, đảm bảo schema thống nhất.
  #figure(
    image("images/artic-shift.png",width: 50%),
    caption: [Công cụ Arctic Shift tải toàn bộ lịch sử r/headphones — vượt
  trần API để có chuỗi thời gian dài hạn.]
  )
  #figure(
    image("images/pushshift.png"),
    caption: [Pushshift giờ chỉ dành cho moderator, càng củng cố nhu cầu tự
  lưu trữ.]
  )
  #figure(
    image("images/data_for_ML.png"),
    caption: [Xem trước Parquet bài đăng và bình luận hợp nhất, đảm bảo ID
  thống nhất trước khi mô hình hóa.]
  )

  === Ảnh chụp corpus
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    [#strong[Split]], [#strong[Rows]], [#strong[Columns]], [#strong[Unique
  subs]],
    table.hline(),
    [Posts], [134121], [27], [16],
    [Comments], [1300190], [16], [16],
  )

  == Kỹ thuật dữ liệu & tích hợp
  - Lọc sử dụng khớp alias không phân biệt hoa thường (ví dụ “hd600”, “fiio
  ft1”), kèm tùy chọn giới hạn theo subreddit.
  - Token thương hiệu/alias được đưa vào danh sách stopword để không chiếm
  ưu thế trong TF-IDF.
  - Giữ cân bằng bài đăng/bình luận bằng cách đồng bộ các khóa (`name`,
  `subreddit`, `link_id`, `parent_id`) trước khi gộp.
  - #mono[`run_pipeline.py`] ghi log metadata tái lập (command-line, phiên
  bản package) và xuất toàn bộ artefact trung gian vào thư mục chỉ định.

#pagebreak()
  == Cách PRAW duyệt cây bình luận
  #columns(2, gutter: 1cm)[
    #list(
      [Reddit trả về cây bình luận với nút `View more comments`.],
      [Phải mở rộng lưới, duyệt và tuần tự hóa thủ công — không có chế độ tải
    một lần.],
      [Thu thập đủ mọi tầng sâu khi cần nhưng vẫn tránh vượt rate limit.],
    )

    #colbreak()

    #image("images/more-comments.png" )
  ]

  == Phân tích dữ liệu thăm dò (EDA)
  - Tài sản EDA nằm trong `eda/` (PNG + CSV) để tái sử dụng nhanh cho slide
  và dashboard.
  #figure(
    image("eda/01_posts_by_subreddit.png",width: 80%),
    caption: [Số bài theo subreddit — PcBuild, iPhone và GamingLaptops chiếm
  sản lượng lớn.]
  )
  #figure(
    image("eda/02_comments_by_subreddit.png"),
    caption: [Số bình luận: cộng đồng xử lý sự cố (PcBuild, homelab) dẫn đầu
  về tương tác.]
  )
  #figure(
    image("eda/04_avg_comment_length_by_subreddit.png"),
    caption: [Độ dài bình luận trung bình cho thấy thảo luận kỹ thuật sâu ở
  r/macbookpro và r/AppleWatch.]
  )
  #figure(
    image("eda/08_avg_comment_score_by_subreddit.png"),
    caption: [Điểm bình luận trung bình khá thấp (≤2.5), nhấn mạnh nhu cầu
  insight văn bản ngoài karma.]
  )
  #figure(
    image("eda/09b_posts_per_week.png"),
    caption: [Sản lượng bài đăng theo tuần phát hiện các đợt ra mắt phần
  cứng theo mùa.]
  )
  #figure(
    image("eda/15_weekday_hour_heatmap.png"),
    caption: [Heatmap theo ngày/giờ — buổi tối và cuối tuần thúc đẩy bàn
  luận.]
  )
  #figure(
    image("eda/11_top_post_authors.png", width: 75%),
    caption: [Tác giả bài đăng hàng đầu — chỉ ra phần lớn những account đăng
  nhiều nhất là bot và những account bị xóa sẽ quy chung về `[deleted]`.]
  )
  #figure(
    image("eda/12_top_comment_authors.png", width: 75%),
    caption: [Tác giả bình luận hàng đầu — nhận diện người thiên về hỗ trợ
  và điều phối.]
  )
  #figure(
    image("eda/13_wordcloud_titles.png", width: 75%),
    caption: [Word cloud tiêu đề nêu bật nhóm sản phẩm chủ đạo và chủ đề xử
  lý sự cố.]
  )
  #figure(
    image("eda/14_wordcloud_comments.png", width: 65%),
    caption: [Word cloud bình luận cho thấy các cụm từ cảm xúc lặp lại và
  thuật ngữ linh kiện.]
  )
  #figure(
    image("eda/16_score_vs_length_comments.png", width: 60%),
    caption: [Biểu đồ score so với độ dài bình luận — phản hồi dài hơn có xu
  hướng nhận karma cao trong chủ đề hỗ trợ.]
  )
  #figure(
    image("eda/17_score_vs_length_posts.png", width: 60%),
    caption: [Biểu đồ score so với độ dài bài đăng — tin đồn ngắn gọn và bài
  review chi tiết đều thu hút người xem.]
  )
  - Các hình bổ sung: word cloud, scatter plot score-theo-độ-dài, bảng xếp
  hạng tác giả và tập CSV cho phân tích sâu hơn (`eda/*.csv`).

  == Phân tích sentiment — Kế hoạch ban đầu & quyết định
  - Quy trình dự kiến: fine-tune `lxyuan/distilbert-base-multilingual-cased-
  sentiments-student` trên bộ nhãn phân tầng, hỗ trợ bởi Gemini (mục tiêu ≥
  5 000 annotate cho hiệu chỉnh và đánh giá).
  - Trở ngại: chi phí thuê GPU, tốc độ gán nhãn bị giới hạn và lợi ích biên
  thấp so với việc tận dụng topic insight không giám sát.
  - Cách triển khai: VADER (rule-based) tính polarity câu trong giai đoạn
  tổng hợp khía cạnh, giữ lại kế hoạch nâng cấp cho tương lai.
  #figure(
    image("images/distillbert.png"),
    caption: [Mục tiêu fine-tune `lxyuan/distilbert-base-multilingual` bị tạm
  hoãn do hạn chế GPU và nhãn.]
  )

  == Topic modeling & pipeline khía cạnh
  CLI vận hành vòng lặp xác định — làm sạch → TF-IDF → SVD → KMeans → sentiment
  theo khía cạnh — cân bằng tự động hóa cho lặp nhanh với các flag rõ ràng
  cho người dùng muốn tùy chỉnh.
  === Tiền xử lý
  - Chuẩn hóa Unicode (NFC), loại URL/khối code, chuyển chữ thường, loại ký
  tự không phải chữ cái nhưng giữ dấu nháy.
  - Lọc token giữ từ dài hơn hai ký tự và loại stopword mở rộng (bao gồm alias
  thương hiệu).
  - Bộ lọc alias/subreddit cho phép tạo corpus tập trung theo từng sản phẩm.

  === Chọn đặc trưng & giảm chiều
  - TF-IDF (`ngram_range = 1–2`) với `min_df` thích ứng (3 nếu N ≥ 300, ngược
  lại 2) và `max_df = 0.95`; vocabulary giới hạn ở `min(2000, 3N)` khi chạy auto
  (`run_pipeline.py:545`).
  - TruncatedSVD giới hạn thành phần ở `min(round(0.25N), 200)` với đáy 50, ghi
  lại explained variance để minh bạch (`run_pipeline.py:560`).
  - Bộ bảo vệ giới hạn SVD ở `min(V - 1, N - 1)` và mở override qua `--min-df`,
  `--max-feat`, `--svd`, giúp corpus nhỏ vẫn an toàn mà không mất kiểm soát
  chuyên gia.
  - Cơ chế bảo vệ vocabulary rỗng tự động thử lại TF-IDF với `min_df = 1`, tránh
  việc chạy alias góc cạnh bị thất bại.

  === Quy trình phân cụm
  - KMeans quét `k_min..k_max` (mặc định 3–8) dựa vào silhouette; `method.json`
  ghi lại phạm vi tìm kiếm, seed và score tốt nhất để phân tích biết vì sao chọn
  `k` đó.
  - Chẩn đoán cụm kết hợp trung bình TF-IDF (`top_terms_by_cluster`) với bài đại
  diện gần centroid (`representatives_by_cluster`) để tạo chủ đề dễ diễn giải và
  trích dẫn.
  - Tùy chọn `--save-plots` sinh bar chart kích thước cụm nhằm QA nhanh trước khi
  chia sẻ.
  #figure(
    image("extra/artifacts_ft1/cluster_sizes.png",width: 60%),
    caption: [Phân bố kích thước cụm (ví dụ Fiio FT1) sau khi giảm chiều bằng
  SVD và phân cụm KMeans.]
  )

  === Trích khía cạnh & kết hợp sentiment
  - Thành viên subreddit tự động chọn bộ seed khía cạnh (pin, nhiệt, độ thoải mái
  ...); tùy chọn TF-IDF expansion bổ sung thuật ngữ đặc thù corpus.
  - Ghép mẫu ở cấp câu; polarity VADER tạo bảng mentions, sentiment trung bình và
  danh sách trích dẫn đã lọc.
  - `assignments.jsonl` gắn nhãn cụm cho từng tài liệu cùng profile sentiment khía
  cạnh, trong khi `aspect_summary*.json` loại khía cạnh nhiễu dưới ngưỡng tần suất
  thích ứng (`max(5, round(0.01N))`).
  - Phân tích viên có thể mở rộng coverage với `--expand-seeds`, thêm top term TF-
  IDF (trừ alias thương hiệu) vào từ điển khía cạnh cho cách diễn đạt tự nhiên.
  - Bộ đệm trích dẫn chỉ giữ câu có compound polarity vượt ±0.05 và giới hạn tối đa
  năm ví dụ mỗi khía cạnh, cân bằng tín hiệu và ngắn gọn.
  // #figure(
  //   image("extra/artifacts_ft1/aspects_pos_neg.png"),
  //   caption: [Tổng hợp sentiment theo khía cạnh (chạy Fiio FT1): số trích dẫn
  // tích cực vs. tiêu cực.]
  // )

  === Artefact trọng tâm
  - `tfidf_vectorizer.joblib`, `svd_model.joblib`,
  `svd_explained_variance.json` — pipeline đặc trưng có thể tái sử dụng.
  - `kmeans_clusters.json` — kích thước cụm, top TF-IDF và bài đại diện gần
  centroid gói trong một tệp; `method.json` ghi lại siêu tham số đã chọn.
  - `assignments.jsonl`, `aspect_summary.json`,
  `aspect_summary_by_subreddit.json` — cụm cấp tài liệu kèm sentiment khía
  cạnh, lọc bằng ngưỡng tần suất thích ứng.
  - PNG tùy chọn: `cluster_sizes.png`, `aspects_pos_neg.png` để kiểm tra nhanh.

  == Deliverables & hướng dẫn sử dụng
  - Điểm vào CLI: `run_pipeline.py --data <parquet> --product <name> --aliases
  <comma-separated> --out <dir> [options]`.
  - Tham số cấu hình điều khiển vector hóa (`--min-df`, `--max-df`, `--max-
  feat`, `--ngram-*`), giảm chiều (`--svd`), phân cụm (`--method`, `--k-min`,
  `--k-max`) và hành vi khía cạnh (`--aspect-category`, `--expand-seeds`, `--min-
  aspect-freq`).
  - Artefact cho Fiio FT1, Sennheiser HD600 và Sony WF-1000XM4 được lưu tại
  `extra/artifacts_ft1`, `extra/artifacts_hd600` và `extra/artifacts_m4`.

  == Hạn chế & hướng phát triển
  - Sentiment vẫn dựa rule-based; thực thi kế hoạch fine-tune sẽ xử lý châm biếm và
  thuật ngữ đặc thù tốt hơn.
  - Nhánh NMF/LDA mới dựng khung, chưa sản phẩm hóa — bổ sung sẽ cho phép chủ đề
  chồng lấn.
  - Mở rộng tích hợp Tiki và hỗ trợ đa ngôn ngữ tốt hơn sẽ phản ánh sentiment phi
  tiếng Anh chính xác hơn.
  - Cần bổ sung đánh giá (topic coherence, độ ổn định khi lấy mẫu lại, kiểm chứng
  có con người) cho vòng lặp kế tiếp.
  #figure(
    image("images/non-english-comments.png", width: 60%),
    caption: [Ví dụ Reddit đa ngôn ngữ cho thấy nhu cầu xử lý tiếng Việt và châu
  Âu ở bản phát hành tiếp theo.]
  )
  #figure(
    image("images/vast_ai.png"),
    caption: [Ảnh chụp marketplace GPU Vast.ai — dự toán chi phí fine-tune khiến
  nhóm hoãn huấn luyện transformer.]
  )

  == Phụ lục
  === Tài sản cốt lõi
  - `plan.md` — kiến trúc chi tiết, phương án thay thế, heuristic đánh giá.
  - `eda/` — biểu đồ khám phá và tổng hợp CSV.
  - `run_pipeline.py` (thư mục gốc & `extra/`) — CLI chủ đề + sentiment khía cạnh
  tùy chỉnh.
  - `extra/bai_thuyet_trinh.typ` — slide Typst trình bày trên lớp.

  === Ví dụ tóm tắt console (rút gọn)
  #block[
    #mono[
      N=742, V=1625, min_df=3, max_df=0.95, max_features=2000 \
      Số thành phần SVD=150, cumulative explained variance=0.72 \
      Chọn K=5 với silhouette=0.41 \
      Aspect battery POS: "Battery life has been excellent..." (+0.68) \
      Aspect battery NEG: "Battery drains fast when streaming..." (-0.52)
    ]
  ]

  #pagebreak()
  #align(center)[#smallcaps[Kết thúc báo cáo]]
  #v(0.5cm)
  #block[
    Trong khuôn khổ môn Kỹ thuật Khai thác Dữ liệu, nhóm đã xây dựng một pipeline
    hiện đại để gom, làm sạch và phân tích thảo luận cộng đồng xoay quanh sản
    phẩm công nghệ. Chuỗi công việc này thể hiện khả năng kết nối nhiều nguồn dữ
    liệu, triển khai quy trình máy học có thể tái lập và rút ra insight thực tiễn
    cho doanh nghiệp. Chúng em tin rằng những cải tiến tương lai về sentiment
    và đa ngôn ngữ sẽ giúp pipeline trở thành nền tảng phân tích sản phẩm toàn
    diện hơn. Xin chân thành cảm ơn thầy đã hỗ trợ, góp ý suốt môn học vừa qua.
  ]
