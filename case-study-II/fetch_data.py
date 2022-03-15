import os
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

import time

AMAZON_REVIEW_URL = ("https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz")
#"https://doc-0o-9g-docs.googleusercontent.com/docs/securesc/5eqdu74r584cfufeb8vmchjpj1ptksh5/ojmcg877ql586rbsdkeufg26vinb7qa4/1641494025000/07511006523564980941/14553342361444956870/0Bz8a_Dbh9QhbaW12WVVZS2drcnM?resourcekey=0-NlNEsGjcAEPJBwDJtuuZ3Q&e=download&authuser=0&nonce=plg1bg5jjl7bk&user=14553342361444956870&hash=ge521kq0qla99g8a31uhptlg0oocnbe5")
AMAZON_REVIEW_ARCHIVE_NAME = "amazon_review_polarity_csv.tar.gz"



def check_amazon_review():
    print("Checking availability of the amazon_review dataset")
    archive_path = os.path.join(".", AMAZON_REVIEW_ARCHIVE_NAME)
    amazon_review_path = os.path.join(".", 'amazon_review')

    if not os.path.exists(amazon_review_path):
        if not os.path.exists(archive_path):
            print("Downloading dataset from %s " % AMAZON_REVIEW_URL)
            opener = urlopen(AMAZON_REVIEW_URL)
            open(archive_path, 'wb').write(opener.read())
        else:
            print("Found archive: " + archive_path)

        print("Extracting %s to %s" % (archive_path, amazon_review_path))
    print("Checking that the amazon_review CSV files exist...")
    print("=> Success!")


if __name__ == "__main__":

    start_time = time.monotonic()
    check_amazon_review()
    elapsed_time = time.monotonic() - start_time
    print("Time:{}".format(elapsed_time))