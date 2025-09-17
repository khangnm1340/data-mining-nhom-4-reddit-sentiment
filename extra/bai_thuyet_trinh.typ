// Diatypst-based presentation assembled from presentation.typ and report.typ outline
#import "@preview/diatypst:0.7.0": *

#show: slides.with(
  title: "Data Mining ", // Required
  subtitle: "Phân tích nội dung trò chuyện để trích xuất vấn đề phổ biến về tài sản (hàng điện tử, hàng dân dụng)",
  authors: ("Nhóm 4"),
  // Optional Style Options
  title-color: purple.darken(50%),
  first-slide: true,
  ratio: 16/9, // aspect ratio of the slides, any valid number
  layout: "small", // one of "small", "medium", "large"
  toc: true,
  count: "number", // one of "dot", "number", or none
  footer: false,
  // footer-title: "Custom Title",
  // footer-subtitle: "Custom Subtitle",
  // theme: "full", // one of "normal", "full"
)


// Common colors
#let accent = rgb(0x2f, 0x6f, 0xf6)
#let ok = rgb(0x17, 0x9e, 0x63)
#let subtle = luma(96%)
// Toggle to display images if they exist under ./images
#let show_images = true

= .Tổng quát

=== Vấn đề và giá trị

- Sản phẩm nào hay hỏng, vì sao, và tần suất bao nhiêu — từ thảo luận công khai.
- Biến văn bản không cấu trúc từ Reddit và Tiki meaningful insights.
- Xuất ra insight mà nhóm có thể dùng: vấn đề, xu hướng, mức độ nghiêm trọng, ví dụ.

=== Mục tiêu

- Những vấn đề nào? Tầng suất ra sao? Xu hướng thế nào? Mức độ nghiêm trọng đến đâu
- Các vấn đề tập trung ở đâu (by product/subreddit/time)?
- Xây dựng một pipeline có thể lặp lại và đưa ra các phát hiện rõ ràng, có thứ tự ưu tiên
Reddit/Tiki collection → preprocess→ EDA → unify to Parquet → Topic modeling ( Feature selection/Dimensionality reduction = (LSA/SVD) -> K-mean cluster -> Artifacts, Metrics -> rồi đánh giá và lặp lại ) 

== Data Source Comparison

== Nguồn dữ liệu 

#table(
  columns: (auto, 2fr, 2fr),
  align: (left, top, top),
  stroke: 0.5pt + gray,
  inset: 6pt,

  // header row
  [#strong[Platform]], [#strong[Pros]], [#strong[Cons]],
  table.hline(),

  // Reddit row
  [Reddit],
  [#list(
    [API công khai (truy cập dễ dàng)],
    [Thảo luận văn bản phong phú],
    [Hỗ trợ mạnh mẽ cho NLP tiếng Anh],
    [Subreddit tập trung vào chủ đề, do cộng đồng điều hành],
  )],
  [#list(
    [Một số bài viết gây nhiễu và lạc đề],
    [Dữ liệu tiếng Việt hạn chế],
    [Hạn ngạch/hạn chế sử dụng API],
  )],

  // Facebook row
  [Facebook],
  [#list(
    [Cơ sở người dùng lớn tại Việt Nam],
    [Các nhóm và cộng đồng tích cực],
    [Chủ đề đa dạng],
  )],
  [#list(
    [Nhiều bot/tài khoản spam],
    [Hạn chế API],
    [Khó thu thập dữ liệu sạch],
    [Hỗ trợ yếu cho các thư viện NLP tiếng Anh],
  )],

  // TikTok row
  [TikTok],
  [#list(
    [Rất phổ biến với người dùng trẻ],
    [Xu hướng mạnh mẽ / Nhận định về meme],
  )],
  [#list(
    [Chủ yếu là phương tiện truyền thông (video, hình ảnh)],
    [Không có API chính thức],
    [Thiếu cấu trúc nhóm/cộng đồng],
    [Thu thập dữ liệu chậm (phân tích cú pháp HTML)],
    [Khó trích xuất dữ liệu văn bản có liên quan],
  )],
)

 
= .Thu thập dữ liệu 
  //#image()

== Phương pháp 

- Ban đầu: PRAW (Python Reddit API Wrapper) để thử nghiệm.
  - Giới hạn: ~1,000 bài, rate limit.
- Hiện tại (kết hợp):
  - Dữ liệu lịch sử Reddit (Academic Torrents) để bỏ giới hạn API.
  - Dump review Tiki để đối chiếu.
  - Kết quả: khung thời gian rộng hơn, dữ liệu nhiều hơn.


== How PRAW Traverses Comments 

#columns(2, gutter: 1cm)[
  #list(
    [Reddit trả về cây bình luận với chỗ `View more comments`.],
    [Mở rộng lưới, duyệt và tuần tự hóa — không có “tải xuống” một cú bấm.],
    [Bắt toàn bộ độ sâu khi cần; tránh bùng nổ rate limit.],
  )

  #colbreak()

  #image("images/more-comments.png", height: 100%)
]



== Các lựa chọn thay thế

#grid(columns: (1fr, 1fr), column-gutter: 1cm,
  [
    #strong[Pushshift.io (Pushshift API)]  
- Mạnh hơn PRAW, lọc bài theo thời gian
- Cần quyền moderator → không khả thi
  ],
  [
    #if show_images [#image("images/pushshift.png", width: 100%)]
    #if not(show_images) [#box(fill: subtle, inset: 8pt, radius: 6pt)[placeholder]]
  ],
)

#grid(columns: (1fr, 1fr), column-gutter: 1cm,
  [
    #strong[Personal Archive]  
- Thu thập liên tục trong nhiều tuần
- Không thực tế: phần cứng/thời gian; không lấy được bài cũ
  ],
  [
    #if show_images [#image("images/archive.png", width: 100%)]
    #if not(show_images) [#box(fill: subtle, inset: 8pt, radius: 6pt)[placeholder]]
  ],
)

#grid(columns: (1fr, 1fr), column-gutter: 1cm,
  [
    #strong[Academic Torrents (Arctic Shift)]  
- Có thể lọc posts và comments theo thời gian.
- Không bị giới hạn số lượng
],
  [
    #if show_images [#image("images/artic-shift.png", width: 100%)]
    #if not(show_images) [#box(fill: subtle, inset: 8pt, radius: 6pt)[placeholder]]
  ],
)

== Subreddit Đã Chọn

#grid(
  columns: (6fr, 4fr),   // tỉ lệ 6:4
  column-gutter: 1cm,

  [
    - Tổ hợp đa dạng các cộng đồng công nghệ và đời sống:
    - r/macbookpro, r/GamingLaptops
    - r/iphone, r/AppleWatch, r/Monitors, r/headphones, r/homelab, r/photography
    - ...và vài subreddit khác về gia đình, âm thanh, và lắp ráp PC.
    - tổng: `134121` bài viết, `1300190` bình luận, 2025-06-01 → 2025-07-31.
    // - [Hình: lưới huy hiệu subreddit tỷ lệ theo số mẫu]
  ],

  [
    #image("images/subreddits.png", width: 60%)
    #image("images/item_counts.png", width: 90%)
  ]
)


== Tiki

#if show_images [#image("images/tiki.png", width: 80%)]

= Preprocessing

== Pipeline tiền xử lý


// #text(size: 10pt, fill: gray)[Convert 32 JSONL → 2 Parquet (Nushell + Polars); keeping as many meaningful columns as possible.]

#grid(
  columns: (4fr, 1fr, 4fr),
  column-gutter: 14pt,
  [
   / Trước: 32 file jsonl \(posts và comments của 16 subreddits)\ posts: 106 cột \ comments : 69 cột
  ],
[  // ô giữa: canh giữa cả hai trục
    #align(center + horizon)[
      #text(size: 16pt, weight: "bold", fill: accent)[->]
    ]
  ],
  [
    / Sau: 2 file parquet của posts và comments.\ posts: 27 cột \ comments : 16 cột
  ],
)
Chỉ giữ những cột có ý nghĩa.
#columns(2)[
  - *Làm sạch*:
    - Xóa URL\
    - Chỉ giữ lại những comments bằng tiếng anh. Loại gần 6 ngàn những comment ngôn ngữ khác bằng `fastText`
    #colbreak()
  ]
    #place(
        dx: 8cm,   // shift right
        dy: -3cm,   // shift up
      image("images/2025-09-09-133417_hyprshot.png",width: 40%))
#if show_images [
 // #image("images/data_for_ML.png", width: 60%)
]

== Gộp dữ liệu: 32 JSONL → 2 Parquet → 1 Parquet

#columns(2)[
  Mục tiêu: Tạo một tệp Parquet duy nhất cho cả bài viết & bình luận

  #list(
    [Đọc *32 tệp JSONL* → phân loại theo loại dữ liệu],
    [Xuất *2 Parquet*: `posts.parquet` và `comments.parquet`],
    [Chuẩn hoá *schema* và *nối dọc* → `all.parquet`],
  )

  Sơ đồ nhanh:
  `32 JSONL` → `posts.parquet` + `comments.parquet` → *chuẩn hoá schema* → *concat* → `all.parquet`

  #colbreak()

  Schema mục tiêu (ví dụ)
  #list(
    [`name`, `author`, `subreddit`, `created_utc`, `score`, `body`,
     `link_id`, `parent_id`, `is_post`,
     `upvote_ratio`, `num_comments`, `num_crossposts`],
  )

  Quy tắc chuẩn hoá
  #list(
    [*Posts*: `body = title + "\n" + selftext`; `is_post = true`;
     `link_id = null`; `parent_id = null`; các cột thiếu → `null`],
    [*Comments*: giữ nguyên `body`; `is_post = false`;
     giữ `link_id`, `parent_id`; cột thiếu → `null`],
  )

  Nối & kiểm tra
  #list(
    [*Chọn đúng thứ tự cột* như *Schema mục tiêu* cho cả hai bảng],
    [*Ghép dọc* `posts` ⊕ `comments` → `all.parquet`],
    [*Kiểm tra*: số dòng, tỉ lệ `null`, lấy mẫu vài hàng],
  )
]
= EDA and Visualization

== Dataset Overview

#columns(2, gutter: 1cm)[
  #heading(level: 3)[Basic Stats]

  - *Các hàng bài post*: `134121`
  - *Các dòng bình luận*: `1300190`
  - *Các sub chứa bài đăng độc nhất*: `16`
  - *Các sub có bình luận duy nhất*: `16`

  #heading(level: 3)[Posts schema]

  #list(
    [archived, author, created_utc, domain, hide_score, id, is_crosspostable, is_self, is_video, name, num_comments, num_crossposts, media_only, over_18, permalink, pinned, retrieved_on, selftext, send_replies, stickied, subreddit, subreddit_id, subreddit_name_prefixed, subreddit_subscribers, thumbnail, title, score, upvote_ratio],
  )

  #heading(level: 3)[Comments schema]

  #list(
    [archived, author, body, created_utc, name, id, is_submitter, link_id, locked, parent_id, permalink, retrieved_on, score, score_hidden, subreddit, subreddit_id, subreddit_name_prefixed],
  )
]

  



== Posts & Comments by Subreddit
#columns(2)[
  #image("eda/01_posts_by_subreddit.png", width: 100%)
  #colbreak()
  #image("eda/02_comments_by_subreddit.png", width: 100%)
]


== Comment & Post Lengths
#columns(2)[
  #image("eda/03_comment_length_hist.png", width: 100%)
  #colbreak()
  #image("eda/04_avg_comment_length_by_subreddit.png", width: 100%)
]

#pagebreak()
#columns(2)[
  #image("eda/05_title_length_hist.png", width: 100%)
  #colbreak()
  #image("eda/06_selftext_length_hist.png", width: 100%)
]


== Post Scores
#columns(2)[
  #image("eda/07_post_scores_boxplot.png", width: 100%)
  #colbreak()
  #image("eda/08_avg_comment_score_by_subreddit.png", width: 100%)
]

== Activity Over Time
#columns(2)[
  #image("eda/09_posts_per_month.png", width: 100%)
  #colbreak()
  #image("eda/09b_posts_per_week.png", width: 100%)
]

== Temporal Patterns
#columns(2)[
  #image("eda/10_comments_by_hour.png", width: 100%)
  #colbreak()
  #image("eda/15_weekday_hour_heatmap.png", width: 100%)
]

== Top Authors
#columns(2)[
  #image("eda/11_top_post_authors.png", width: 100%)
  #colbreak()
  #image("eda/12_top_comment_authors.png", width: 100%)
]

== Wordclouds
#columns(2)[
  #image("eda/13_wordcloud_titles.png", width: 110%)
  #colbreak()
  #image("eda/14_wordcloud_comments.png", width: 110%)
]

== Score vs Length
#columns(2)[
  #image("eda/16_score_vs_length_comments.png", width: 100%)
  #colbreak()
  #image("eda/17_score_vs_length_posts.png", width: 100%)
]


//== 
//
//- So sánh phân loại theo ngữ cảnh và theo từng bình luận trên tập nhãn nhỏ.
//- Kết quả: phân loại theo từng bình luận giúp tránh việc cắt ngắn 512 token và hiện tượng nhiễu ngữ cảnh.
//#if show_images [
//  #columns(2)[
//    #image("images/contextualized_comment_1.png", width: 100%)
//    #colbreak()
//    #image("images/contextualized_comment_2.png", width: 100%)
//  ]
//]


= TOPIC MODELING (K-Means Cluster)

//Here’s the gist: this script is a CLI pipeline that
//
//#list(
//  [Cắt nhỏ tập dữ liệu Reddit xuống các dòng liên quan đến một sản phẩm cụ thể bằng cách khớp alias] 
//  [Tùy chọn: giảm chiều với SVD]
//  [Làm sạch văn bản và xây dựng đặc trưng TF-IDF] ,
//  [Quét KMeans để chọn số cụm K tốt nhất dựa trên silhouette][Khai thác sentiment ở mức khía cạnh bằng VADER (tự động suy luận từ vựng từ các subreddit)],
//)

== Mục tiêu và cách làm (flow)

- Mục tiêu: tóm tắt chủ đề (issues/topics) & cảm xúc (sentiment) từ thảo luận Reddit để hiểu pain points người dùng.

- Dữ liệu: 16 subreddit về công nghệ/điện tử tiêu dùng; đã làm sạch (lowercase, bỏ URL, chuẩn thời gian), lấy mẫu có kiểm soát.

- Sentiment Analysis (SA):

- Định fine-tune bằng nhãn từ Gemini nhưng bị rate limit (~20 cmt/lần); fine-tune có ý nghĩa cần ≥5k nhãn ⇒ tạm bỏ fine-tune.
#image("images/distillbert.png",height:70%)

- Dùng model có sẵn (pretrained) để gán nhãn pos/neu/neg và điểm cho mỗi bình luận.


== End-to-end flow của Topic modeling

#list(
[Nạp & lọc: Đọc tệp, yêu cầu cột body, khớp alias không phân biệt hoa/thường. Tùy chọn giao với --subs.],
[Làm sạch văn bản: Bỏ URL/mã, chuyển về chữ thường, loại ký tự không chữ cái, loại token ngắn + stopwords + từ thương hiệu.],
[Vector hóa (TF-IDF): n-gram (mặc định 1–2), min_df tự động, max_df=0.95, max_features=min(2000, 3N).],
[Giảm chiều (SVD): clamp(round(0.25N), 50, 200). Lưu mô hình + phương sai giải thích.],
[Quét & chọn KMeans: Thử K trong khoảng, chọn silhouette tốt nhất. Lưu kích thước cụm, từ khóa hàng đầu, đại diện.],
[Khai phá khía cạnh + cảm xúc: Tạo seed từ danh mục subreddit, mở rộng, khớp câu, chấm điểm bằng VADER. Tổng hợp lượt nhắc, cảm xúc trung bình, trích dẫn. Cắt bỏ khía cạnh dưới ngưỡng.],
[Đầu ra + tóm tắt console: In thống kê, silhouette, phương sai SVD, trích dẫn ±.],
)

== Kết quả chính (điền số & hình sau khi chạy)
#columns(2)[
- Topics/Issues nổi bật (ví dụ đặt tên):
- Battery/Pin, Display/Màn hình, Connectivity/Kết nối, Performance/Hiệu năng,
- Setup/Compatibility/Cài đặt – tương thích, Audio/Âm thanh, Price/Giá trị, Thermals/Nhiệt – tiếng ồn.
→ Mỗi topic: top từ khoá + 2–3 bình luận đại diện.
#colbreak()
Sau đó ghi ra artifacts (joblibs/JSON/plots)
#image("images/command_output.png" )
]
== What it takes in
```bash
python scripts/run_pipeline.py 
--data FILE_CHUNG_1TRIEU_MAU.parquet 
--product "Fiio FT1" --aliases "Fiio Ft1, ft1" -
-out artifacts_ft1  --ngram-min 1 --ngram-max 2 
--min-df auto --max-df 0.95 --max-feat auto 
--svd auto --svd-n-iter 7 --svd-random-state 42 
--method kmeans --k-min 3 --k-max 8 --kmeans-random-state 42 
--kmeans-n-init 10 
--aspect-category headphones 
--expand-seeds --expand-top-k 50 
--min-aspect-freq auto --save-plots

```
`--data`: parquet of posts or comments  
/ `--product` + `--aliases`: e.g. `"Sennheiser HD600"` and `"hd600,hd 600,hd6xx,sennheiser hd600"`  
/ Vectorization knobs: `--min-df/--max-df/--max-feat/--ngram-min/--ngram-max`  
/ SVD knobs: `--svd auto|N|skip`, `--svd-n-iter`  
/ Clustering: `--method kmeans`, `--k-min/--k-max` sweep  
#columns(2)[/ Aspect mining: `--aspect-category `, `--expand-seeds`, `--expand-top-k`
#colbreak()
#image("images/aspect.png")
]


== What it writes

#image("images/2025-09-10-080944_hyprshot.png")
