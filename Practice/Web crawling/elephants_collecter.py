from google_images_download import google_images_download


def imageCrawling(keyword, dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keyword,
                 "limit": 300,
                 "chromedriver": "/home/EIEN443/chromedriver",
                 "print_urls": False,
                 "no_directory": True,
                 "output_directory": dir}
    path = response.download(arguments)
    print(path)


# african elephant
imageCrawling('forest elephant', './datasets/crawl/forest elephant')
imageCrawling('African bush elephant', './datasets/crawl/African bush elephant')
imageCrawling('sub-Saharan Africa elephants', './datasets/crawl/sub-Saharan Africa elephants')

# asian elephant
imageCrawling('Asian elephants', './datasets/crawl/Asian elephants')
imageCrawling('Indian elephants', './datasets/crawl/Indian elephants')
imageCrawling('Elephas maximus', './datasets/crawl/Elephas maximus')
