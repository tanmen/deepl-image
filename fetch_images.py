from icrawler.builtin import BingImageCrawler

# 猫の画像を100枚取得
crawler = BingImageCrawler(storage={"root_dir": "img/deep_learning/cat"})
crawler.crawl(keyword="猫", max_num=100)

crawler = BingImageCrawler(storage={"root_dir": "img/deep_learning/dog"})
crawler.crawl(keyword="犬", max_num=100)
