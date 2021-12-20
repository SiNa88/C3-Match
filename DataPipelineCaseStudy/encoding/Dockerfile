FROM ubuntu

RUN apt-get update && apt-get install -y ffmpeg python3 wget

# Set the working directory
WORKDIR /work

# Add python-script
#COPY run-encoding.py /work
#COPY 20.mp4 /work
#COPY 20-20000.mp4 /work


#RUN wget -O /usr/local/bin/kubetools https://github.com/kubemq-io/kubemq-community/releases/download/v2.2.10/kubemq-linux-armv7.tar.gz
#RUN chmod +x /usr/local/bin/kubetools

ENTRYPOINT ["python3", "run-encoding.py", "x264"]

