from google_images_download import google_images_download

def imageCrawling(keyword, dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":'African elephants',
                 "limit" : 200,
                 "chromedriver" : "/"}